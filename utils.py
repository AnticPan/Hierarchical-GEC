from __future__ import annotations
import logging
from typing import List, NamedTuple, Dict, Union, Tuple
from transformers import AutoTokenizer
import torch
import numpy as np

BIO = {"B-M":0,"B-R":1,"I-R":2,"B-WS":3,"I-WS":4,"O":5}

class Token(NamedTuple):
    word: str
    ids: List[int]

class Patch(NamedTuple):
    start: int
    end: int
    words: List[str]
    ids: List[int]

class Example(NamedTuple):
    tokens: List[Token]
    tf_label: int
    tags: List[int]
    patches: List[Patch]
    target_sentence: str

    @property
    def words(self) -> List[str]:
        return [token.word for token in self.tokens]

    @property
    def source_sentence(self) -> str:
        return " ".join(self.words)

    @property
    def ids(self) -> List[int]:
        return [id for token in self.tokens for id in token.ids]

    @property
    def offsets(self) -> List[int]:
        offsets = []
        offset = 0
        for token in self.tokens:
            offsets.append(offset)
            offset += len(token.ids)
        return offsets
    
    def apply_patches(self, EOP_word: str="[SEP]"):
        if self.patches is None:
            return self.source_sentence
        patches = sorted(self.patches, key=lambda x:x.start)
        words = self.words
        for patch in patches[::-1]:
            words[patch.start+1:patch.end]=patch.words[:patch.words.index(EOP_word)] if EOP_word in patch.words else patch.words
        return " ".join(words)
    
    def truncate(self, limit_length: int):
        if self.tf_label is not None:
            tokens = []
            tags = []
            patches = []
            length = 0
            for token, tag in zip(self.tokens, self.tags):
                if len(token.ids) + length > limit_length:
                    break
                else:
                    tokens.append(token)
                    tags.append(tag)
                    length += len(token.ids)
            for patch in self.patches:
                if patch.end < len(tokens):
                    patches.append(patch)
            return Example(tokens, self.tf_label, tags, patches, self.target_sentence)
            
        else:
            tokens = []
            length = 0
            for token in self.tokens:
                if len(token.ids) + length > limit_length:
                    break
                else:
                    tokens.append(token)
                    length += len(token.ids)
            return Example(tokens, self.tf_label, self.tags, self.patches, self.target_sentence)

class Batch(NamedTuple):
    examples: List[Example]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    word_offsets: torch.Tensor
    tf_labels: torch.Tensor
    bio_tags: torch.Tensor

    patch_idx: torch.Tensor
    patch_start_pos: torch.Tensor
    patch_mid_pos: torch.Tensor
    patch_end_pos: torch.Tensor
    patch_ids: torch.Tensor


def lists2tensor(lists: List[List[int]], length: int, fill_value: int) -> torch.Tensor:
    list_num = len(lists)
    np_array = np.full((list_num, length), fill_value)
    for i in range(list_num):
        l = min(len(lists[i]),length)
        np_array[i, :l] = lists[i][:l]

    return torch.tensor(np_array)

def get_words_end_offsets(words: List[str]) -> List[int]:
    words_offsets = []
    offset = 0
    for word in words:
        offset += len(word)+1 # add 1 for the space between words
        words_offsets.append(offset)
    return words_offsets

def seq2words_ids(seq_ids: List[int], seq_offsets: List[Tuple], words_offsets: List[int]) -> List[Tuple]:
    words_ids = []
    word_ids = []
    seq_offset_ptr = 0
    seq_ids_ptr = 0
    for word_offset in words_offsets:
        while seq_offset_ptr < len(seq_offsets) and seq_offsets[seq_offset_ptr][1] <= word_offset:
            word_ids.append(seq_ids[seq_ids_ptr])
            seq_ids_ptr += 1
            seq_offset_ptr += 1
        words_ids.append(word_ids)
        word_ids = []
    
    return words_ids


def batch_tokenization(tokenizer: AutoTokenizer, batch_sentences: List[str],
                       max_piece: int):
    batch_words = [text.split() for text in batch_sentences]

    batch_words_ids = []

    output = tokenizer.batch_encode_plus(batch_sentences,
                                        add_special_tokens = False,
                                        padding = False,
                                        truncation = False,
                                        return_offsets_mapping = True,
                                        return_attention_mask = False,
                                        return_length = False)
    batch_seq_ids = output["input_ids"]
    batch_seq_offsets = output["offset_mapping"]

    batch_words_offsets = [get_words_end_offsets(words) for words in batch_words]
    for idx, (seq_ids, seq_offsets, words_offsets) in enumerate(zip(batch_seq_ids, 
                                                                   batch_seq_offsets, 
                                                                   batch_words_offsets)):
        words_ids = seq2words_ids(seq_ids, seq_offsets, words_offsets)
        assert len(words_ids) == len(batch_words[idx])
        batch_words_ids.append(words_ids)
    # TODO: limit max subword id num per word
    batch_tokens = []
    for words, words_ids in zip(batch_words, batch_words_ids):
        tokens = [Token(word, word_ids[:max_piece]) for word, word_ids in zip(words, words_ids)]
        batch_tokens.append(tokens)
    return batch_tokens


class EpochState(object):
    def __init__(self) -> None:
        self.dis_total = 0
        self.dis_correct = 0
        self.dis_loss = 0
        self.det_total = 0
        self.det_correct = 0
        self.det_loss = 0
        self.cor_total = 0
        self.cor_correct = 0
        self.cor_loss = 0

    def get_current_log(self, step: int):
        return {"disA": self.dis_correct/self.dis_total if self.dis_total else 0,
                "disL": self.dis_loss/step,
                "detA": self.det_correct/self.det_total if self.det_total else 0,
                "detL": self.det_loss/step,
                "corA": self.cor_correct/self.cor_total if self.cor_total else 0,
                "corL": self.cor_loss/step}

def set_logger(log_file):
    logging.basicConfig(level=logging.DEBUG ,filename=log_file ,
                        filemode="w" ,format="%(levelname)-9s- %(message)s")
    logger = logging.getLogger()
    SH = logging.StreamHandler()
    SH.setLevel(logging.INFO)
    logger.addHandler(SH)
