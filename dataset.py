from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import os
import re
import difflib
from utils import Patch, Batch, Example, BIO, batch_tokenization, lists2tensor
from typing import List, Tuple

class GECDataset(Dataset):
    def __init__(self, data_dir:str, tokenizer: AutoTokenizer, inference:bool, only_wrong:bool, 
                 max_piece:int, truncate:int, max_patch_len:int):
        super().__init__()
        self.inference = inference
        self.tokenizer = tokenizer
        self.only_wrong = only_wrong
        self.pairs = self.load_tsv(data_dir)
        self.max_piece = max_piece
        self.truncate = truncate
        self.max_patch_len = max_patch_len


    def load_tsv(self, data_dir:str):
        if os.path.isdir(data_dir):
            data_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".tsv")]
        elif os.path.isfile(data_dir):
            data_paths = [data_dir]
        else:
            raise ValueError(f"{data_dir} is neither a file nor a directory.")
        pairs = []
        for data_path in sorted(data_paths):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    sentences = line.strip("\n").split("\t")
                    if self.inference:
                        pairs.append((self.add_special_token(sentences[0]), None))
                    else:
                        assert len(sentences) == 2, f"line-{i} error in {data_path}"
                        is_correct = sentences[0] == sentences[1]
                        if is_correct and self.only_wrong:
                            continue
                        pairs.append((self.add_special_token(sentences[0]), 
                                      self.add_special_token(sentences[1])))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]
    
    def make_patch(self, start, end, words):
        words.append(self.tokenizer.sep_token)
        phrase = " ".join(words)
        words_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
        return Patch(start, end, words, words_ids)
    
    def compare_with_matcher(self, source_sentence, target_sentence):
        tf_label = int(source_sentence == target_sentence)
        source_words = source_sentence.split()
        target_words = target_sentence.split()
        bio_tags = [BIO["O"]]*len(source_words)
        patches = []
        if not tf_label:
            matcher = difflib.SequenceMatcher(None, source_words, target_words)
            ops = matcher.get_opcodes()
            # https://docs.python.org/3.8/library/difflib.html#difflib.SequenceMatcher.get_opcodes
            for tag, s1, s2, t1, t2 in ops:
                if tag == 'equal':
                    continue
                if tag == 'replace':
                    bio_tags[s1] = BIO["B-WS"]
                    bio_tags[s1+1:s2] = [BIO["I-WS"]]*(s2-s1-1)
                elif tag == 'delete':
                    bio_tags[s1] = BIO["B-R"]
                    bio_tags[s1+1:s2] = [BIO["I-R"]]*(s2-s1-1)
                elif tag == 'insert':
                    bio_tags[s1] = BIO["B-M"]
                patch = self.make_patch(s1-1, s2, target_words[t1:t2])
                patches.append(patch)
        return tf_label, bio_tags, patches

    def add_special_token(self, sentence):
        if not sentence.isprintable():
            sentence = ''.join(x for x in sentence if x.isprintable())
        sentence = re.sub(' +', ' ', sentence)
        return f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"

    
    def make_examples(self, pairs: List[Tuple[str]])->List[Example]:
        source_sentences = [pair[0] for pair in pairs]
        batch_tokens = batch_tokenization(self.tokenizer, source_sentences, self.max_piece)

        examples = []
        for idx, (source_sentence, target_sentence) in enumerate(pairs):
            source_sentences.append(source_sentence)
            if target_sentence:
                tf_label, bio_tags, patches = self.compare_with_matcher(source_sentence, target_sentence)
            else:
                tf_label, bio_tags, patches = None, None, None
            example = Example(batch_tokens[idx], tf_label, bio_tags, patches, target_sentence)
            examples.append(example)
        
        return examples
    
    def collect_fn(self, selected_pairs: List[Tuple[str]]) -> Batch:

        examples = self.make_examples(selected_pairs)

        input_ids = []
        tf_labels = []
        bio_tags = []
        word_offsets = []

        patch_idx = []
        patch_start_pos = []
        patch_mid_pos = []
        patch_end_pos = []
        patch_ids = []
        for idx, example in enumerate(examples):
            example = example.truncate(self.truncate)
            input_ids.append(example.ids)
            offsets = example.offsets
            word_offsets.append(offsets)
            tf_labels.append(example.tf_label)
            bio_tags.append(example.tags)
            for patch in example.patches:
                patch_idx.append(idx)
                patch_start_pos.append(patch.start)
                if patch.start+1 == patch.end:
                    patch_mid_pos.append([-1])
                else:
                    patch_mid_pos.append(list(range(patch.start+1, patch.end)))
                patch_end_pos.append(patch.end)
                patch_ids.append(patch.ids)

        max_len = max(len(ids) for ids in input_ids)
        max_offset_len = max(len(offsets) for offsets in word_offsets)
        assert max_len <= self.truncate
        input_ids = lists2tensor(input_ids, max_len, self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        word_offsets = lists2tensor(word_offsets, max_offset_len, -1)
        tf_labels = torch.tensor(tf_labels, dtype=torch.float)
        bio_tags = lists2tensor(bio_tags, max_offset_len, -100)
        if len(patch_idx):
            max_patch_len = max(len(pos) for pos in patch_mid_pos)
            patch_idx = torch.tensor(patch_idx, dtype=torch.long).unsqueeze(1)
            patch_start_pos = torch.tensor(patch_start_pos, dtype=torch.long).unsqueeze(1)
            patch_mid_pos = lists2tensor(patch_mid_pos, max_patch_len, -2)
            patch_end_pos = torch.tensor(patch_end_pos, dtype=torch.long).unsqueeze(1)
            patch_ids = lists2tensor(patch_ids, self.max_patch_len, -100)
        else:
            patch_idx = None
            patch_start_pos = None
            patch_end_pos = None
            patch_mid_pos = None
            patch_ids = None

        return Batch(examples, input_ids, attention_mask, word_offsets, tf_labels, bio_tags,
                     patch_idx, patch_start_pos, patch_mid_pos, patch_end_pos, patch_ids)


    def collect_fn_inference(self, selected_pairs):
        examples = self.make_examples(selected_pairs)
        input_ids = []
        word_offsets = []
        for idx, example in enumerate(examples):
            example = example.truncate(self.truncate)
            input_ids.append(example.ids)
            offsets = example.offsets
            word_offsets.append(offsets)
        max_len = max(len(ids) for ids in input_ids)
        max_offset_len = max(len(offsets) for offsets in word_offsets)
        assert max_len <= self.truncate
        input_ids = lists2tensor(input_ids, max_len, self.tokenizer.pad_token_id)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        word_offsets = lists2tensor(word_offsets, max_offset_len, -1)
        return Batch(examples, input_ids, attention_mask, word_offsets, None, None,
                     None, None, None, None, None)



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../lib/bert-base-cased")
    dataset = GECDataset("./debug.tsv", tokenizer, False, only_wrong=False, 
                 max_piece=4, truncate=50, max_patch_len=4)
    exps = dataset.make_examples([dataset.pairs[0]])
    batch = dataset.collect_fn([dataset.pairs[0]])
    print(exps)
    print(exps[0].source_sentence)
    print(batch.input_ids, batch.input_ids.size())
    print(batch.word_offsets, batch.word_offsets.size())
