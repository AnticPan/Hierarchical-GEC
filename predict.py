import os
import argparse
import torch
from model.patcher import Patcher
from data_loader import Dataset
from utils.structure import Patch, Token, Example
from utils.patch_handler import Patch_handler
from utils.tokenizer import Tokenizer
from tqdm import tqdm
from typing import List
import pickle
import time

def gather_patches(batch_size:int, start_pos:List[List[int]], end_pos:List[List[int]], predict_ids:List[List[int]], EOP:int):
    patches = [[] for _ in range(batch_size)]
    for idx, (batch_idx, start, end) in enumerate(zip(start_pos[0], start_pos[1], end_pos[1])):
        token_ids = []
        for ids in predict_ids:
            if ids[idx] == EOP:
                break
            token_ids.append(ids[idx])
        if token_ids:
            tokens = Token(word='', ids=token_ids,start=0,end=0)
            patch = Patch(start+1, end, [tokens])
            patches[batch_idx].append(patch)
    return patches

def example2sentence(example:Example, tokenizer:Tokenizer):
    if example.patches:
        patches = example.patches
        patch_ptr = 0
        sentence = ""
        words = []
        pre_ids = []
        for token in example.tokens:
            if patch_ptr >= len(patches):
                # sentence += " "+ token.word
                words.append(token.word)
            elif token.end <= patches[patch_ptr].start:
                # sentence += " "+token.word
                words.append(token.word)
            elif token.start == patches[patch_ptr].start:
                if patches[patch_ptr].end == token.start:
                    for patch_token in patches[patch_ptr].tokens:
                        # sentence += " "+tokenizer.decode(patch_token.ids, patches[patch_ptr].start, example.oovs)
                        patch_words = tokenizer.decode(patch_token.ids, patches[patch_ptr].start, example.oovs)
                        words.append(patch_words)
                    # sentence += " "+token.word
                    words.append(token.word)
                    patch_ptr +=1
                elif patches[patch_ptr].end <= token.end:
                    right = token.end - patches[patch_ptr].end
                    ids = []
                    for patch_token in patches[patch_ptr].tokens:
                        ids.extend(patch_token.ids)
                    if right > 0:
                        ids.extend(token.ids[-right:])
                    # sentence += " "+tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    patch_words = tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    words.append(patch_words)
                    patch_ptr +=1 
                else:
                    continue
            elif token.start < patches[patch_ptr].start:
                if token.end == patches[patch_ptr].end:
                    left = patches[patch_ptr].start - token.start
                    ids = pre_ids
                    ids.extend(token.ids[:left])
                    for patch_token in patches[patch_ptr].tokens:
                        ids.extend(patch_token.ids)
                    # sentence += tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    patch_words = tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    words.append(patch_words)
                    patch_ptr += 1
                    pre_ids = []
                elif token.end < patches[patch_ptr].end:
                    left = patches[patch_ptr].start - token.start
                    if left > 0:
                        pre_ids = token.ids[:left]
                else:
                    right = token.end - patches[patch_ptr].end
                    left = patches[patch_ptr].start - token.start
                    ids = pre_ids
                    if left > 0:
                        ids.extend(token.ids[:left])
                    for patch_token in patches[patch_ptr].tokens:
                        ids.extend(patch_token.ids)
                    if right > 0:
                        ids.extend(token.ids[-right:])
                    # sentence += " "+tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    patch_words = tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    words.append(patch_words)
                    patch_ptr += 1
                    pre_ids = []
            elif token.start > patches[patch_ptr].start:
                if token.end < patches[patch_ptr].end:
                    continue
                elif token.end >= patches[patch_ptr].end:
                    right = token.end-patches[patch_ptr].end
                    ids = pre_ids
                    for patch_token in patches[patch_ptr].tokens:
                        ids.extend(patch_token.ids)
                    if right > 0:
                        ids.extend(token.ids[-right:])
                    # sentence += " "+tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    patch_words = tokenizer.decode(ids, patches[patch_ptr].start, example.oovs)
                    words.append(patch_words)
                    patch_ptr += 1
                    pre_ids = []
        return " ".join(filter(lambda x: x!='', words))
    else:
        return " ".join([token.word for token in example.tokens])

def predict_a_batch(model, batch, tokenizer, patch_handler, args):
    enc_time, dis_time, det_time, cor_time = 0, 0, 0, 0
    start_time = time.time()
    examples = batch.examples
    data = {"input_ids":batch.input_ids,
            "masks":batch.attention_mask,
            "token_type_ids":batch.token_type_ids}
    encoder_outputs = model("encode", data)
    current_time = time.time()
    enc_time = current_time - start_time
    predict_tf_logits, predict_tfs = None, None
    predict_labels = None
    output_sentences = None
    incor_ids = list(range(len(examples)))
    if args.discriminating:
        torch.cuda.synchronize()
        start_time = time.time()
        data = {}
        data["first_hiddens"] = encoder_outputs[1]
        data["target_tfs"] = None
        predict_tf_logits, _ = model("discriminate", data)
        predict_tfs = predict_tf_logits > args.discriminating_threshold
        incor_ids = [idx for idx, value in enumerate(predict_tfs) if value == False]
        torch.cuda.synchronize()
        current_time = time.time()
        dis_time = current_time - start_time
    if args.detecting and incor_ids:
        torch.cuda.synchronize()
        start_time = time.time()
        data = {}
        data["masks"] = batch.attention_mask[:,1:][incor_ids]
        data["encoder_output"] = encoder_outputs[0][:, 1:, :][incor_ids] # modify here
        data["target_labels"] = None
        labeling_output, _ = model("detect", data)
        predict_labels = model.detector.inference(labeling_output, data["masks"])
        torch.cuda.synchronize()
        current_time = time.time()
        det_time = current_time - start_time
    if args.correcting and incor_ids:
        del_patches = []
        if predict_labels:
            start_pos, end_pos, del_patches = patch_handler.get_patch_pos(predict_labels, incor_ids)
        else:
            start_pos = batch.target_starts
            end_pos = batch.target_ends
        if start_pos and end_pos:
            torch.cuda.synchronize()
            start_time = time.time()
            encoder_output = encoder_outputs[0]
            patch_start_states = encoder_output[start_pos[0], start_pos[1]]
            patch_end_states = encoder_output[end_pos[0], end_pos[1]]
            
            patch_mid_states = []
            for batch_idx, start, end in zip(start_pos[0], start_pos[1], end_pos[1]):
                if start + 1 == end:
                    patch_mid_states.append(model.corrector.emtpy_state)
                else:
                    patch_mid_states.append(torch.mean(encoder_output[batch_idx, start+1:end], dim=0))
            patch_mid_states = torch.stack(patch_mid_states)

            data = {}
            data["patch_start_states"]=patch_start_states
            data["patch_end_states"]=patch_end_states
            data["patch_mid_states"] = patch_mid_states
            data["patch_ids"] = None
            data["length"] = args.max_decode_step
            corrector_output, _ = model("correct", data)
            corrector_output_ids = torch.softmax(corrector_output, dim=-1).argmax(dim=-1) # (patch_num, max_decode_step)
            predict_ids = corrector_output_ids.transpose(0,1).cpu().tolist()
            torch.cuda.synchronize()
            current_time = time.time()
            cor_time = current_time - start_time
            patches = gather_patches(len(examples), start_pos, end_pos, predict_ids, tokenizer.PATCH_END_ID)
            if del_patches:
                for idx, d_patch in del_patches:
                    if patches[idx]:
                        flag = False
                        for index in range(len(patches[idx])):
                            if patches[idx][index].start > d_patch.start:
                                flag = True
                                break
                        if flag:
                            patches[idx][index:index] = [d_patch]
                        else:
                            patches.append(d_patch)
                    else:
                        patches[idx] = [d_patch]
            examples = [Example(example.tokens, patch, example.oovs, example.targets) 
                        for example, patch in zip(examples, patches)]
        elif del_patches:
            patches = [[] for _ in range(len(examples))]
            for idx, d_patch in del_patches:
                if patches[idx]:
                    patches.append(d_patch)
                else:
                    patches[idx] = [d_patch]
            examples = [Example(example.tokens, patch, example.oovs, example.targets) 
                        for example, patch in zip(examples, patches)]
        else:
            examples = [Example(example.tokens, None, example.oovs, example.targets) for example in batch.examples]
        output_sentences = []
        for example in examples:
            output_sentences.append(example2sentence(example, tokenizer))
    results = []
    for i in range(len(examples)):
        results.append({"source": " ".join([token.word for token in examples[i].tokens]),
                        "tf_logit":float(predict_tf_logits[i]) if predict_tf_logits is not None else None,
                        "tf":bool(predict_tfs[i]) if predict_tfs is not None else None,
                        "labels":predict_labels[incor_ids.index(i)] if predict_labels and i in incor_ids else None,
                        "output":output_sentences[i] if output_sentences else None})
    times = [enc_time, dis_time, det_time, cor_time]
    return results, times

def test(model: Patcher, test_data: Dataset, 
        tokenizer: Tokenizer, patch_handler:Patch_handler, args):
    model.eval()
    test_gen = test_data.generator()
    results = []
    enc_time, dis_time, det_time, cor_time = 0, 0, 0, 0
    overall_start = time.time()
    with torch.no_grad():
        for batch in tqdm(test_gen, desc="Inferencing", total=test_data.get_batch_num()):
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
    parser.add_argument("--max_decode_step", default=4, type=int)

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = Patcher(args.model_dir, 
                    discriminating=args.discriminating,
                    detecting=args.detecting,
                    correcting=args.correcting,
                    use_crf=args.use_crf,
                    use_lstm=args.use_lstm)
    tokenizer = Tokenizer(args.model_dir, args.lower_case)
    patch_handler = Patch_handler(tokenizer.PATCH_EMPTY_ID, dir_del=args.dir_del)
    test_data = Dataset(args.test_file,args.batch_size,True,tokenizer,
                        discriminating=args.discriminating,
                        detecting=args.detecting,
                        correcting=args.correcting,
                        only_wrong=args.only_wrong)
    if torch.cuda.is_available():
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
        result_file = os.path.join(args.output_dir, file_prefix+".txt")
        with open(result_file, "w") as f:
            for result in results:
                if result["tf"] == 0 or result["tf"] is None or (not args.discriminating and not args.detecting and args.decoding):
                    f.write(result["output"].strip()[6:-6].strip()+"\n")
                else: # 1 or None
                    f.write(result["source"].strip()[6:-6].strip()+"\n")
