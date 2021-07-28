import os
import logging
from data_loader import Dataset
from model.patcher import Patcher
import argparse
import torch
import math
import random
from transformers import BertModel, AdamW
from transformers.optimization import get_constant_schedule_with_warmup
from utils.tokenizer import Tokenizer
from utils.recoder import Statistic
from utils.patch_handler import Patch_handler
from tqdm import tqdm
import numpy as np
import logging
logging.getLogger("train.py").setLevel(logging.INFO)


def handle_a_batch(step, model, batch, recoder, args):
    discriminator_loss, detector_loss, decoder_loss = None, None, None
    data = {"input_ids": batch.input_ids, "masks": batch.attention_mask, "token_type_ids": batch.token_type_ids}
    encoder_outputs = model("encode", data)
    if args.discriminating:
        data = {}
        data["first_hiddens"] = encoder_outputs[1]
        data["target_tfs"] = batch.target_tfs
        predict_tf_logits, discriminator_loss = model("discriminate", data)

        predict_tfs = predict_tf_logits > args.discriminating_threshold
        recoder.update_discriminator(step + 1,
                                    discriminator_loss.mean().item(),
                                    predict_tfs.cpu().tolist(),
                                    batch.target_tfs.cpu().tolist())
    if args.detecting:
        data = {}
        data["masks"] = batch.attention_mask[:, 1:]
        data["encoder_output"] = encoder_outputs[0][:, 1:, :]
        data["target_labels"] = batch.target_labels[:, 1:]
        labeling_output, detector_loss = model("detect", data)
        predict_labels = torch.softmax(labeling_output, dim=-1).argmax(dim=-1)
        list_predict_labels = [labels[mask].cpu().tolist() for labels, mask in zip(predict_labels, data["masks"])]
        list_target_labels = [labels[mask].cpu().tolist() for labels, mask in zip(data["target_labels"], data["masks"])]
        recoder.update_detector(list_predict_labels, list_target_labels, batch.examples)

        if args.discriminating:
            detector_loss = detector_loss[batch.error_example_mask].mean()
        else:
            detector_loss = detector_loss.mean()
        recoder.update_detector_loss(step + 1, detector_loss.item())

    if args.correcting:
        start_pos = batch.target_starts
        end_pos = batch.target_ends
        patch_ids = batch.target_ids
        if patch_ids is not None:
            patch_ids = patch_ids[:, :args.max_decode_step]
            encoder_output = encoder_outputs[0]
            patch_start_states = encoder_output[start_pos[0], start_pos[1]]
            patch_end_states = encoder_output[end_pos[0], end_pos[1]]
            patch_mid_states = []
            for batch_idx, start_pos, end_pos in zip(start_pos[0], start_pos[1], end_pos[1]):
                if start_pos + 1 == end_pos:
                    if gpu_num > 1:
                        patch_mid_states.append(model.module.corrector.emtpy_state)
                    else:
                        patch_mid_states.append(model.corrector.emtpy_state)
                else:
                    patch_mid_states.append(torch.mean(encoder_output[batch_idx, start_pos + 1:end_pos], dim=0))
            patch_mid_states = torch.stack(patch_mid_states)
            data = {}
            data["patch_start_states"] = patch_start_states
            data["patch_end_states"] = patch_end_states
            data["patch_mid_states"] = patch_mid_states
            data["patch_ids"] = patch_ids
            data["length"] = patch_ids.size(-1)
            _, corrector_loss = model("correct", data)
            recoder.update_corrector(corrector_loss.mean().item())
    losses = [discriminator_loss, detector_loss, corrector_loss]
    return losses


def train(model: Patcher, train_data: Dataset, valid_data: Dataset, model_save_dir: str, gpu_num: int, recoder: Statistic,
          args):
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
    total_train_steps = train_data.get_batch_num() * args.epoch
    warmup_steps = int(args.warmup * total_train_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    current_step = 0
    decay_ratio = None
    for i in range(args.epoch):
        model.train()
        recoder.reset("train", i + 1)
        train_gen = train_data.generator()
        step_num = train_data.get_batch_num()
        process_bar = tqdm(enumerate(train_gen), total=step_num, desc="Training in epoch %d/%d" % (i + 1, args.epoch))
        for step, batch in process_bar:
            losses = handle_a_batch(step, model, batch, recoder, args)
            optimizer.zero_grad()
            loss = sum(filter(lambda x: x is not None, losses))
            if gpu_num > 1:
                loss.mean().backward()
            else:
                loss.backward()
            process_bar.set_postfix(recoder.get_current_log())
            optimizer.step()
            scheduler.step()
            current_step += 1
        recoder.save()
        if valid_data is not None:
            with torch.no_grad():
                model.eval()
                recoder.reset("valid", i + 1)
                valid_gen = valid_data.generator()
                step_num = valid_data.get_batch_num()
                process_bar = tqdm(enumerate(valid_gen), total=step_num, desc="Validing in epoch %d/%d" % (i + 1, args.epoch))
                for step, batch in process_bar:
                    handle_a_batch(step, model, batch, recoder, args)
                    # discriminator_loss, detector_loss, predict_labels, decoder_loss = outputs
                    process_bar.set_postfix(recoder.get_current_log())
                recoder.save()
        if gpu_num > 1:
            model.module.save(model_save_dir, i + 1)
        else:
            model.save(model_save_dir, i + 1)
        tokenizer.save(model_save_dir, i+1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gpus", type=int, nargs='+', default=None)

    parser.add_argument("-lower_case", default=False, action="store_true")
    parser.add_argument("-only_wrong", default=False, action="store_true")

    parser.add_argument("-discriminating", default=True, action="store_false")
    parser.add_argument("--discriminating_threshold", default=0.5, type=float)

    parser.add_argument("-detecting", default=True, action="store_false")
    parser.add_argument("-use_crf", default=True, action="store_false")
    parser.add_argument("-use_lstm", default=False, action="store_true")
    parser.add_argument("-dir_del", default=False, action="store_true")

    parser.add_argument("-correcting", default=True, action="store_false")
    parser.add_argument("-use_detect_out", default=False, action="store_true")
    parser.add_argument("--max_decode_step", default=4, type=int)

    parser.add_argument("-freeze", default=False, action="store_true")
    parser.add_argument("--truncate", type=int, default=512)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=1e-2)

    args = parser.parse_args()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    
    if not (args.correcting or args.detecting or args.discriminating):
        raise ValueError("Cannot set discriminating, detecting and correcting to False at same time.")
    if args.gpus:
        gpu_num = len(args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpus])
    else:
        gpu_num = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "cmd"), "w") as f:
        f.write(str(args))
    log_dir = os.path.join(args.output_dir, "log")
    model_save_dir = os.path.join(args.output_dir, "model")
    tokenizer = Tokenizer(args.bert_dir, args.lower_case)
    patch_handler = Patch_handler(tokenizer.PATCH_EMPTY_ID, args.dir_del)
    recoder = Statistic(log_dir,
                        args.discriminating,
                        args.detecting,
                        args.correcting,
                        max_decode_step=args.max_decode_step,
                        patch_handler=patch_handler)
    model = Patcher(args.bert_dir,
                    discriminating=args.discriminating,
                    detecting=args.detecting,
                    correcting=args.correcting,
                    use_crf=args.use_crf,
                    use_lstm=args.use_lstm)
    if gpu_num == 1:
        model = model.cuda()
    if gpu_num > 1:
        model = torch.nn.DataParallel(model).cuda()
    train_data = Dataset(args.train_file,
                         args.batch_size,
                         inference=False,
                         tokenizer=tokenizer,
                         discriminating=args.discriminating,
                         detecting=args.detecting,
                         correcting=args.correcting,
                         dir_del=args.dir_del,
                         only_wrong=args.only_wrong,
                         truncate=args.truncate)
    if args.valid_file:
        valid_data = Dataset(args.valid_file,
                             args.batch_size,
                             inference=False,
                             tokenizer=tokenizer,
                             discriminating=args.discriminating,
                             detecting=args.detecting,
                             correcting=args.correcting,
                             dir_del=args.dir_del,
                             only_wrong=args.only_wrong,
                             truncate=args.truncate)
    else:
        valid_data = None

    train(model, train_data, valid_data, model_save_dir, gpu_num, recoder, args)
