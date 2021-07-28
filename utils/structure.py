from typing import List, NamedTuple, Set, Dict, Union
import torch
import os
import re
import numpy as np

BIO = {"B-M":0,"B-R":1,"I-R":2,"B-WS":3,"I-WS":4,"O":5}

class Token(NamedTuple):
    word: str
    ids: List[int]
    start: int
    end: int

class Patch(NamedTuple):
    start: int
    end: int
    tokens: List[Token]
    # type: str # U,M,R


class Example(NamedTuple):
    tokens: List[Token]
    patches: List[Patch]
    oovs : Dict[int, List[Union[int, str]]]
    targets: List[str]


class Batch(NamedTuple):
    examples: List[Example]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    target_tfs: torch.Tensor
    target_labels: torch.Tensor
    error_example_mask: torch.Tensor

    target_starts: List[List[int]]  # [[index_dim_0], [index_dim_1]]
    target_ends: List[List[int]]  # [[index_dim_0], [index_dim_1]]
    target_ids: List[torch.Tensor]


def lists2tensor(lists: List[List[int]], length: int, max_length:int, fill_value: int):
    list_num = len(lists)
    length = min(length, max_length)
    np_array = np.full((list_num, length), fill_value)
    for i in range(list_num):
        l = min(len(lists[i]),length)
        np_array[i, :l] = lists[i][:l]

    return torch.tensor(np_array)

