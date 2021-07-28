import random
import time
import numpy as np
import os
import difflib
import torch
from utils.structure import Example, Batch, Patch, lists2tensor, Token, BIO
from utils.tokenizer import Tokenizer
from typing import List, Union
from tqdm import tqdm
from collections import Counter
import Levenshtein
import math
import copy

class Dataset(object):

    def __init__(self, data_dir:str, batch_size:int, inference:bool, tokenizer:Tokenizer, 
                 discriminating:bool=False, detecting: bool = False, correcting:bool=False, 
                 dir_del:bool=False, only_wrong:bool=False, truncate: int = 512):
        # self._data_paths, self.total_example_num = devide_large_file(data_dir, cache_dir, single_pass)
        self.inference = inference
        self.truncate = truncate
        self.example_num = 0
        self.wrong_example_ids = []
        self.right_example_ids = []
        self.tsv_examples, self.domain_words = self.load_tsv(data_dir)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.discriminating = discriminating
        self.detecting = detecting
        self.correcting = correcting
        self.dir_del = dir_del
        self.only_wrong = only_wrong

    def load_tsv(self, data_dir:str):
        if os.path.isdir(data_dir):
            data_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir)]
        elif os.path.isfile(data_dir):
            data_paths = [data_dir]
        else:
            raise ValueError(f"{data_dir} is neither a file nor a directory.")
        tsv_examples = []
        domain_words = []
        for data_path in sorted(data_paths):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    sentences = line.strip("\n").split("\t")
                    if self.inference:
                        tsv_examples.append(sentences)
                    else:
                        assert len(sentences) >= 2, f"line-{i} error in {data_path}"
                        is_correct = any(sentences[0] == sentence for sentence in sentences[1:])
                        if is_correct:
                            self.right_example_ids.append(self.example_num)
                        else:
                            self.wrong_example_ids.append(self.example_num)
                        # record the words from target sentence
                        #domain_words.extend(sentences[1].split())
                        tsv_examples.append(sentences)
                    self.example_num += 1
        if self.inference:
            domain_words_freq = None
        else:
            domain_words_freq = Counter(domain_words)
        return tsv_examples, domain_words_freq
    
    def get_batch_num(self):
        return math.ceil(len(self.tsv_examples)/self.batch_size)

    def generator(self):
        if self.inference:
            example_ids = list(range(self.example_num))
        else:
            # example_ids = list(range(self.example_num))
            # random.shuffle(example_ids)
            if self.only_wrong:
                example_ids = copy.copy(self.wrong_example_ids)
                random.shuffle(example_ids)
            else:
                error_ratio = len(self.wrong_example_ids) / self.example_num
                batch_wrong_num = int(error_ratio * self.batch_size)
                batch_right_num = self.batch_size - batch_wrong_num
                wrong_example_ids = copy.copy(self.wrong_example_ids)
                right_example_ids = copy.copy(self.right_example_ids)
                random.shuffle(wrong_example_ids)
                random.shuffle(right_example_ids)
                example_ids = []
                for ptr in range(
                        max(math.ceil(len(wrong_example_ids) / batch_wrong_num),
                            math.ceil(len(right_example_ids) / batch_right_num))):
                    example_ids.extend(wrong_example_ids[ptr * batch_wrong_num:(ptr + 1) * batch_wrong_num])
                    example_ids.extend(right_example_ids[ptr * batch_right_num:(ptr + 1) * batch_right_num])
        for ptr in range(math.ceil(len(example_ids) / self.batch_size)):
            ids = example_ids[ptr * self.batch_size:(ptr + 1) * self.batch_size]
            es = []
            for idx in ids:
                tsv_example = self.tsv_examples[idx]
                source_sentence, *target_sentences = tsv_example
                es.append(self.make_example(source_sentence, target_sentences))
            batch = self.make_batch(es)
            yield self.to_device(batch)
    
    @staticmethod
    def to_device(batch):
        if torch.cuda.is_available():
            input_ids = batch.input_ids.cuda()
            attention_mask = batch.attention_mask.cuda()
            token_type_ids = batch.token_type_ids.cuda()
            if batch.target_tfs is not None:
                target_tfs = batch.target_tfs.cuda()
            else:
                target_tfs = None
            if batch.target_labels is not None:
                error_example_mask = batch.error_example_mask.cuda()
                target_labels = batch.target_labels.cuda()
            else:
                target_labels = None
                error_example_mask = None
            if batch.target_ids is not None:
                target_ids = batch.target_ids.cuda()
            else:
                target_ids = None
            return Batch(batch.examples, input_ids, attention_mask, token_type_ids, target_tfs,
                target_labels, error_example_mask, batch.target_starts, batch.target_ends, target_ids)
        else:
            return batch

    def make_batch(self, examples:List[Example]):
        pad_token_id = 0
        input_ids = []
        target_labels = []

        target_starts = [[], []]
        target_ends = [[], []]
        target_ids = []
        error_example_mask = [0] * len(examples)
        for i, example in enumerate(examples):
            input_tokens = example.tokens
            patches = example.patches
            ids = []
            for token in input_tokens:
                ids.extend(token.ids)
            input_ids.append(ids)
            labels = [ BIO["O"]] * len(ids)
            if patches is not None:
                pre_type = None
                for patch in patches:
                    if patch.start >= self.truncate or patch.end >= self.truncate:
                        break
                    error_example_mask[i] = 1
                    if patch.start == patch.end: # insert
                        labels[patch.start] = BIO["B-M"]
                        target_starts[0].append(i)
                        target_starts[1].append(patch.start-1)
                        target_ends[0].append(i)
                        target_ends[1].append(patch.end)
                        target_ids.append([])
                        for token in patch.tokens:
                            target_ids[-1].extend(token.ids)
                        target_ids[-1].append(self.tokenizer.PATCH_END_ID)
                    elif patch.tokens[0].word == '': # delete
                        labels[patch.start] = BIO["B-R"]
                        labels[patch.start+1:patch.end] = [BIO["I-R"]]*(patch.end-patch.start-1)
                        if self.dir_del:
                            continue
                        target_starts[0].append(i)
                        target_starts[1].append(patch.start-1)
                        target_ends[0].append(i)
                        target_ends[1].append(patch.end)
                        target_ids.append([])
                        for token in patch.tokens:
                            target_ids[-1].extend(token.ids)
                        target_ids[-1].append(self.tokenizer.PATCH_END_ID)
                    else: # replace
                        labels[patch.start] = BIO["B-WS"]
                        labels[patch.start+1:patch.end] = [BIO["I-WS"]]*(patch.end-patch.start-1)
                        target_starts[0].append(i)
                        target_starts[1].append(patch.start-1)
                        target_ends[0].append(i)
                        target_ends[1].append(patch.end)
                        target_ids.append([])
                        for token in patch.tokens:
                            target_ids[-1].extend(token.ids)
                        target_ids[-1].append(self.tokenizer.PATCH_END_ID)
            target_labels.append(labels)

        input_max_len = max([len(id_list) for id_list in input_ids])
        input_ids = lists2tensor(input_ids, input_max_len, self.truncate, 0)
        attention_mask = torch.full(
            input_ids.size(), pad_token_id, dtype=torch.bool)
        attention_mask[torch.where(input_ids != pad_token_id)] = 1
        token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        if self.discriminating:
            # 0 wrong, 1 correct
            target_tfs = torch.tensor([0 if value==1 else 1 for value in error_example_mask], dtype=torch.float)
        else:
            target_tfs = None
        if self.detecting:
            error_example_mask = torch.tensor(error_example_mask).bool()
            target_labels = lists2tensor(
                target_labels, input_max_len, self.truncate, -100)
        else:
            error_example_mask = None
            target_labels = None
        if self.correcting:
            if target_ids:
                target_max_len = max(len(id_list)
                                    for id_list in target_ids)
                target_ids = lists2tensor(
                    target_ids, target_max_len, 20, -100)
            else:
                target_starts = None
                target_ends = None
                target_ids = None
        else:
            target_starts = None
            target_ends = None
            target_ids = None
        return Batch(examples, input_ids, attention_mask, token_type_ids, target_tfs,
                        target_labels, error_example_mask, target_starts, target_ends, target_ids)

    def make_example(self, source_sentence:str, target_sentences:List[str]):
        source_words = list(filter(lambda x:x!='',source_sentence.strip().split(" ")))
        source_tokens, oovs = self.tokenizer.encode(source_words, is_patch=False)
        if len(target_sentences) == 0 or any(source_sentence == sentence for sentence in target_sentences):
            example = Example(source_tokens, None, oovs, target_sentences)
        else:
            if len(target_sentences) > 1:
                levenshtein_distances = []
                for target_sentence in target_sentences:
                    target_words = list(filter(lambda x:x!='',target_sentence.strip().split(" ")))
                    distance = Levenshtein_distance_list(source_words, target_words)
                    levenshtein_distances.append(distance)
                min_index = levenshtein_distances.index(min(levenshtein_distances))
                target_sentence = target_sentences[min_index]
            else:
                target_sentence = target_sentences[0] 
            patch_list = []
            # target_words = [token.text for token in nlp(target_sentence)]
            target_words = list(filter(lambda x:x!='',target_sentence.strip().split(" ")))
            source_words = ['[CLS]'] + source_words + ['[SEP]']
            target_words = ['[CLS]'] + target_words + ['[SEP]']
            matcher = difflib.SequenceMatcher(None, source_words, target_words)
            ops = matcher.get_opcodes()
            # https://docs.python.org/3.8/library/difflib.html#difflib.SequenceMatcher.get_opcodes
            for tag, s1, s2, t1, t2 in ops:
                if tag == 'equal':
                    continue
                start = source_tokens[s1].start
                end = source_tokens[s2].start
                if tag == 'replace':
                    target_tokens, _ = self.tokenizer.encode(target_words[t1:t2], is_patch=True)
                elif tag == 'delete':
                    target_tokens, _ = self.tokenizer.encode([], is_patch=True)
                elif tag == 'insert':
                    target_tokens, _ = self.tokenizer.encode(target_words[t1:t2], is_patch=True)
                patch = Patch(start, end, target_tokens)
                # if not equal(source_tokens, patch):
                #     patch_list.append(patch)
                patch_list.append(patch)
            example = Example(source_tokens, patch_list, oovs, target_sentences)
        return example

def Levenshtein_distance_list(source, target):
    unique_elements = sorted(set(source + target)) 
    char_list = [chr(i) for i in range(len(unique_elements))]
    if len(unique_elements) > len(char_list):
        raise Exception("too many elements")
    else:
        unique_element_map = {ele:char_list[i]  for i, ele in enumerate(unique_elements)}
    source_str = ''.join([unique_element_map[ele] for ele in source])
    target_str = ''.join([unique_element_map[ele] for ele in target])
    distance = Levenshtein.distance(source_str, target_str)
    return distance

def devide_large_file(data_dir, output_dir, no_split=False, max_line_num=100000):
    if os.path.isdir(data_dir):
        data_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir)]
    elif os.path.isfile(data_dir):
        data_paths = [data_dir]
    else:
        raise ValueError(f"{data_dir} is neither a file nor a directory.")
    line_nums = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_data_paths = []
    for data_path in data_paths:
        line_num = 0
        with open(data_path,"r",encoding="utf-8") as f:
            for line in f:
                line_num += 1
        line_nums.append(line_num)
        if line_num == 0:
            raise ValueError("empty file: %s"%data_path)
        elif line_num <= max_line_num or no_split:
            new_data_paths.append(data_path)
        else:
            with open(data_path, 'r', encoding='utf-8') as fin:
                devide_num = line_num // max_line_num+1
                for part in range(devide_num):
                    _, file_name = os.path.split(data_path)
                    new_path = os.path.join(output_dir, f"{file_name}.part.{part}")
                    new_data_paths.append(new_path)
                    with open(new_path, "w") as fout:
                        for idx, line in enumerate(fin):
                            fout.write(line)
                            if idx == max_line_num-1:
                                break

    return new_data_paths, sum(line_nums)

def equal(source_tokens: List[Token], patch: Patch):
    # FIXME: what if the token is oov or something like?
    start = patch.start
    end = patch.end
    source_ids = []
    for token in source_tokens:
        if token.start>=start and token.end <= end:
            source_ids.extend(token.ids)
    target_ids = [i for token in patch.tokens for i in token.ids]
    if source_ids == target_ids:
        return True
    return False
