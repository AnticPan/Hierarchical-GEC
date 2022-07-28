import pickle
import os
import logging
from dataset import GECDataset
from model.patcher import Patcher, PatcherOutput
import argparse
import torch
import math
import random
from transformers import AdamW, AutoTokenizer
from transformers.optimization import get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import Batch, EpochState, set_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def handle_a_batch(model: Patcher, batch: Batch, epoch_state: EpochState):
    output: PatcherOutput
    output = model(input_ids=batch.input_ids.cuda(),
                   attention_mask=batch.attention_mask.cuda(),
                   word_offsets=batch.word_offsets.cuda(),
                   tf_labels=batch.tf_labels.cuda() if batch.tf_labels is not None else None,
                   bio_tags=batch.bio_tags.cuda() if batch.bio_tags is not None else None,
                   patch_idx=batch.patch_idx.cuda() if batch.patch_idx is not None else None,
                   patch_start_pos=batch.patch_start_pos.cuda() if batch.patch_start_pos is not None else None,
                   patch_mid_pos=batch.patch_mid_pos.cuda() if batch.patch_mid_pos is not None else None,
                   patch_end_pos=batch.patch_end_pos.cuda() if batch.patch_end_pos is not None else None,
                   patch_ids=batch.patch_ids.cuda() if batch.patch_ids is not None else None)

    if args.discriminating:
        output.discriminator_loss = output.discriminator_loss.mean()
        predict_tfs = (output.discriminator_logits > args.discriminating_threshold).cpu()
        epoch_state.dis_total += predict_tfs.size(0)
        epoch_state.dis_correct += int((predict_tfs == batch.tf_labels).sum())
        epoch_state.dis_loss += output.discriminator_loss.item()

    if args.detecting:
        predict_labels = output.detector_logits.argmax(dim=-1).cpu()
        masks = batch.word_offsets[:, 1:] != -1
        target_labels = batch.bio_tags[:, 1:]

        for predict, target, mask in zip(predict_labels, target_labels, masks):
            epoch_state.det_total += int(mask.sum())
            epoch_state.det_correct += int((predict == target)[mask].sum())

        if args.discriminating:
            error_example_mask = (1 - batch.tf_labels).bool().to(output.detector_loss.device)
            if int(error_example_mask.sum()) != 0:
                output.detector_loss = output.detector_loss[error_example_mask].mean()
                epoch_state.det_loss += output.detector_loss.item()
            else:
                output.detector_loss = None
        else:
            output.detector_loss = output.detector_loss.mean()
            epoch_state.det_loss += output.detector_loss.item()

    if args.correcting and batch.patch_idx is not None:
        # epoch_state.cor_total += int((batch.patch_ids != -100).sum())
        predict_tokens = output.corrector_logits.argmax(dim=-1).cpu()
        masks = batch.patch_ids != -100
        target_tokens = batch.patch_ids
        for predict, target, mask in zip(predict_tokens, target_tokens, masks):
            epoch_state.cor_total += int(mask.sum())
            epoch_state.cor_correct += int((predict == target)[mask].sum())
        epoch_state.cor_loss += output.corrector_loss.item()

    losses = [output.discriminator_loss, output.detector_loss, output.corrector_loss]

    return epoch_state, losses


def multi_task_loss(losses, args):
    if args.loss_weight == "manual":
        loss = 0
        if losses[0] is not None:
            loss += losses[0] * args.dis_weight
        if losses[1] is not None:
            loss += losses[1] * args.det_weight
        if losses[2] is not None:
            loss += losses[2] * args.cor_weight
        return loss
    elif args.loss_weight == "balance":
        losses = [loss for loss in losses if loss is not None]
        total_loss = sum([loss.item() for loss in losses])
        weights = [len(losses) * loss.item() / total_loss for loss in losses]
        loss = sum(weight * loss for weight, loss in zip(weights, losses))
        return loss


def train(model: Patcher, train_data: GECDataset, valid_data: GECDataset, model_save_dir: str, args):
    if args.freeze:
        for name, value in model.encoder.named_parameters():
            value.requires_grad = False
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", 'gamma', 'beta']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay':
        args.decay
    }, {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay':
        0.0
    }]
    total_train_steps = len(train_data) // args.batch_size * args.epoch
    warmup_steps = int(args.warmup * total_train_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    for i in range(args.epoch):
        model.train()
        train_state = EpochState()
        train_gen = DataLoader(train_data, args.batch_size, shuffle=True, collate_fn=train_data.collect_fn)
        process_bar = tqdm(enumerate(train_gen),
                           total=len(train_data) // args.batch_size,
                           desc="Training in epoch %d/%d" % (i + 1, args.epoch))
        for step, batch in process_bar:
            train_state, losses = handle_a_batch(model, batch, train_state)
            optimizer.zero_grad()
            loss = multi_task_loss(losses, args)
            loss.backward()
            process_bar.set_postfix(train_state.get_current_log(step + 1))
            optimizer.step()
            scheduler.step()
        logging.info(f"epoch-{i+1} train result")
        logging.info(str(train_state.get_current_log(step)))
        if valid_data is not None:
            model.eval()
            valid_gen = DataLoader(valid_data, 32, shuffle=False, collate_fn=valid_data.collect_fn)
            valid_state = EpochState()
            process_bar = tqdm(enumerate(valid_gen),
                               total=len(valid_data) // 32,
                               desc="Validing in epoch %d/%d" % (i + 1, args.epoch))
            with torch.no_grad():
                for step, batch in process_bar:
                    valid_state, _ = handle_a_batch(model, batch, valid_state)
                    # discriminator_loss, detector_loss, predict_labels, decoder_loss = outputs
                    process_bar.set_postfix(valid_state.get_current_log(step + 1))
            logging.info(f"epoch-{i+1} valid result")
            logging.info(str(valid_state.get_current_log(step)))
        model.save(tokenizer, model_save_dir, i + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("-lower_case", default=False, action="store_true")
    parser.add_argument("-only_wrong", default=False, action="store_true")

    parser.add_argument("-discriminating", default=True, action="store_false")
    parser.add_argument("--discriminating_threshold", default=0.5, type=float)

    parser.add_argument("-detecting", default=True, action="store_false")
    parser.add_argument("-use_crf", default=True, action="store_false")
    parser.add_argument("-use_lstm", default=False, action="store_true")

    parser.add_argument("-correcting", default=True, action="store_false")
    parser.add_argument("-use_detect_out", default=False, action="store_true")
    parser.add_argument("--max_patch_len", default=4, type=int)
    parser.add_argument("--max_piece", default=4, type=int)

    parser.add_argument("-freeze", default=False, action="store_true")
    parser.add_argument("--truncate", type=int, default=50)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=1e-2)

    parser.add_argument("--loss_weight", type=str, choices=["manual", "balance"], default="manual")
    parser.add_argument("--dis_weight", type=float, default=1)
    parser.add_argument("--det_weight", type=float, default=1)
    parser.add_argument("--cor_weight", type=float, default=1)

    args = parser.parse_args()

    set_seed(123)

    if not (args.correcting or args.detecting or args.discriminating):
        raise ValueError("Cannot set discriminating, detecting and correcting to False at same time.")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_logger(os.path.join(args.output_dir, "train.log"))
    logging.debug(str(args))
    model_save_dir = os.path.join(args.output_dir, "model")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, do_lower_case=args.lower_case, use_fast=True)
    model = Patcher(args.bert_dir,
                    discriminating=args.discriminating,
                    detecting=args.detecting,
                    correcting=args.correcting,
                    use_crf=args.use_crf,
                    use_lstm=args.use_lstm,
                    max_patch_len=args.max_patch_len)
    model = model.cuda()
    train_data = GECDataset(args.train_file, tokenizer, False, args.only_wrong, args.max_piece, args.truncate,
                            args.max_patch_len)
    if args.valid_file:
        valid_data = GECDataset(args.valid_file, tokenizer, False, args.only_wrong, args.max_piece, 512, args.max_patch_len)
    else:
        valid_data = None

    train(model, train_data, valid_data, model_save_dir, args)
