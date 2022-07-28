import os
import argparse
import torch
from torch.utils.data import DataLoader
from model.patcher import Patcher
from transformers import AutoTokenizer
from dataset import GECDataset
from tqdm import tqdm
from typing import List
import pickle
import time
from patch_handler import Patch_handler

from utils import Batch, Example, Patch

cuda = torch.cuda.is_available()

def gather_patches(incor_num, patch_idx, patch_start_pos, patch_end_pos, predict_ids, tokenizer, EOP: int):
    assert patch_idx.size(0) == len(predict_ids)
    none_empty_patches = [[] for _ in range(int(incor_num))]
    for idx, start_pos, end_pos, ids in zip(patch_idx, patch_start_pos, patch_end_pos, predict_ids):
        if EOP in ids:
            ids = ids[:ids.index(EOP)]
        words = tokenizer.decode(ids).strip().split()
        patch = Patch(start_pos, end_pos, words,ids)
        none_empty_patches[idx].append(patch)
    
    return none_empty_patches

def merge_del_patches(patches, del_patches):
    for idx, d_patch in del_patches:
        patches[idx].append(d_patch)
    return patches

def add_empty_patch(incor_mask, patches):
    incor_mask = incor_mask.cpu()
    for idx, is_incor in enumerate(incor_mask):
        if not is_incor:
            patches[idx:idx] = [[]]
    return patches

def predict_a_batch(model: Patcher, batch: Batch, tokenizer, patch_handler, args):
    enc_time, dis_time, det_time, cor_time = 0, 0, 0, 0
    examples = batch.examples
    start_time = time.time()
    words_offsets = batch.word_offsets.cuda() if cuda else batch.word_offsets
    cls_states, words_states = model.encode(input_ids=batch.input_ids.cuda() if cuda else batch.input_ids,
                                            attention_mask=batch.attention_mask.cuda() if cuda else batch.attention_mask,
                                            word_offsets=words_offsets)
    current_time = time.time()
    enc_time = current_time - start_time
    predict_tf_logits, predict_tfs = None, None
    predict_labels = None
    output_sentences = None
    # incor_ids = list(range(len(examples)))
    incor_num = len(examples)
    incor_mask = torch.ones((incor_num,), dtype=torch.bool)
    if args.discriminating:
        torch.cuda.synchronize()
        start_time = time.time()
        predict_tf_logits, _ = model.discriminate(cls_states, None)
        predict_tfs = predict_tf_logits > args.discriminating_threshold
        # incor_ids = [idx for idx, value in enumerate(predict_tfs) if value == False]
        incor_mask = predict_tfs == False
        incor_num = int(incor_mask.sum())
        if incor_num:
            words_states = words_states[incor_mask]
            words_offsets = words_offsets[incor_mask]
        torch.cuda.synchronize()
        current_time = time.time()
        dis_time = current_time - start_time
    if args.detecting and incor_num:
        torch.cuda.synchronize()
        start_time = time.time()
        mask = (words_offsets != -1)[:,1:]
        labeling_output, _ = model.detect(words_states, mask, None)
        predict_labels = model.detector.inference(labeling_output, mask)
        torch.cuda.synchronize()
        current_time = time.time()
        det_time = current_time - start_time
    if args.correcting and incor_num:
        del_patches = []
        if predict_labels is not None:
            patch_idx, patch_start_pos, patch_mid_pos, patch_end_pos, del_patches = patch_handler.get_patch_pos(predict_labels)
        else:
            patch_idx = batch.patch_idx
            patch_start_pos = batch.patch_start_pos
            patch_mid_pos = patch_mid_pos 
            patch_end_pos = patch_end_pos
        if patch_idx is not None:
            torch.cuda.synchronize()
            start_time = time.time()
            corrector_output, _ = model.correct(words_states,
                                                patch_idx=patch_idx.cuda() if cuda else patch_idx,
                                                patch_start_pos=patch_start_pos.cuda() if cuda else patch_start_pos,
                                                patch_mid_pos=patch_mid_pos.cuda() if cuda else patch_mid_pos,
                                                patch_end_pos=patch_end_pos.cuda() if cuda else patch_end_pos)
            
            corrector_output_ids = corrector_output.argmax(dim=-1) # (patch_num, max_patch_len)
            predict_ids = corrector_output_ids.cpu().tolist()
            # predict_ids = corrector_output_ids.transpose(0,1).cpu().tolist()
            torch.cuda.synchronize()
            current_time = time.time()
            cor_time = current_time - start_time
            patches = gather_patches(incor_num, patch_idx, patch_start_pos, patch_end_pos, predict_ids, tokenizer, tokenizer.sep_token_id)
            if del_patches:
                patches = merge_del_patches(patches, del_patches)
            patches = add_empty_patch(incor_mask, patches)
            examples = [Example(example.tokens, None, None, patch, example.target_sentence) 
                        for example, patch in zip(examples, patches)]
        elif del_patches:
            patches = [[] for _ in range(len(incor_num))]
            patches = merge_del_patches(patches, del_patches)
            patches = add_empty_patch(incor_mask, patches)

            examples = [Example(example.tokens, None, None, patch, example.target_sentence)  
                        for example, patch in zip(examples, patches)]
        else:
            examples = [Example(example.tokens, None, None, None, example.target_sentence)  for example in batch.examples]
        output_sentences = []
        for example in examples:
            output_sentences.append(example.apply_patches())
    results = []
    for i in range(len(examples)):
        results.append({"source": examples[i].source_sentence,
                        "tf_logit":float(predict_tf_logits[i]) if predict_tf_logits is not None else None,
                        "tf":bool(predict_tfs[i]) if predict_tfs is not None else None,
                        # "labels":predict_labels[incor_ids.index(i)] if predict_labels is not None and incor_mask[i] else None,
                        "patches":examples[i].patches if examples[i].patches is not None else None,
                        "output":output_sentences[i] if output_sentences else None})
    times = [enc_time, dis_time, det_time, cor_time]
    return results, times

def test(model: Patcher, test_data: GECDataset, 
        tokenizer: AutoTokenizer, patch_handler, args):
    model.eval()
    test_gen = DataLoader(test_data, args.batch_size, shuffle=False, collate_fn=test_data.collect_fn_inference)
    results = []
    enc_time, dis_time, det_time, cor_time = 0, 0, 0, 0
    overall_start = time.time()
    for batch in tqdm(test_gen, desc="Inferencing", total=len(test_data)//args.batch_size):
        with torch.no_grad():
            batch_results, times = predict_a_batch(model, batch, tokenizer, patch_handler, args)
        enc_time += times[0]
        dis_time += times[1]
        det_time += times[2]
        cor_time += times[3]
        results.extend(batch_results)
    print("tot time:", time.time() - overall_start)
    print("enc time:", enc_time)
    print("dis time:", dis_time)
    print("det time:", det_time)
    print("cor time:", cor_time)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=None)

    parser.add_argument("-lower_case", default=False, action="store_true")
    parser.add_argument("-only_wrong", default=False, action="store_true")

    parser.add_argument("-discriminating", default=True, action="store_false")
    parser.add_argument("--discriminating_threshold", default=0.5, type=float)

    parser.add_argument("-detecting", default=True, action="store_false")
    parser.add_argument("-use_crf", default=True, action="store_false")
    parser.add_argument("-dir_del", default=False, action="store_true")
    parser.add_argument("-use_lstm", default=False, action="store_true")

    parser.add_argument("-correcting", default=True, action="store_false")
    parser.add_argument("--max_patch_len", default=4, type=int)
    parser.add_argument("--max_piece", default=4, type=int)
    parser.add_argument("--truncate", type=int, default=512)

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = Patcher(args.model_dir, 
                    discriminating=args.discriminating,
                    detecting=args.detecting,
                    correcting=args.correcting,
                    use_crf=args.use_crf,
                    use_lstm=args.use_lstm)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, do_lower_case=args.lower_case, use_fast=True)
    patch_handler = Patch_handler(tokenizer.sep_token_id, dir_del=args.dir_del)
    test_data = GECDataset(args.test_file,tokenizer, inference=True, only_wrong=False, 
                            max_piece=args.max_piece, truncate=args.truncate, max_patch_len=args.max_patch_len)
    if cuda:
        model = model.cuda()

    results = test(model, test_data, tokenizer, patch_handler, args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    _, file_name = os.path.split(args.test_file)
    file_prefix, _ = os.path.splitext(file_name)
    output_file = os.path.join(args.output_dir, file_prefix+".output.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    if args.correcting:
        result_file = os.path.join(args.output_dir, file_prefix+".predict")
        with open(result_file, "w") as f:
            for result in results:
                if result["tf"] == 0 or result["tf"] is None or (not args.discriminating and not args.detecting and args.decoding):
                    f.write(result["output"].strip()[6:-6].strip()+"\n")
                else: # 1 or None
                    f.write(result["source"].strip()[6:-6].strip()+"\n")
