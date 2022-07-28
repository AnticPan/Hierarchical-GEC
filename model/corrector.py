# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from https://github.com/facebookresearch/SpanBERT/blob/master/code/pytorch_pretrained_bert/modeling.py
import torch
import torch.nn as nn
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class RNN_Decoder(nn.Module):
    def __init__(self, hidden_size: int, embedding_weight: torch.Tensor):
        super(RNN_Decoder, self).__init__()
        self.vocab_size = embedding_weight.size()[0]
        self.embedding_dim = embedding_weight.size()[1]
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight, padding_idx=0)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.merge_state_layer = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                               nn.ReLU())
        self.context_state_layer = nn.Sequential(nn.Linear(hidden_size*2+self.embedding_dim, hidden_size),
                                                 nn.ReLU())

        self.project_layer = nn.Linear(hidden_size, self.vocab_size)
        self.project_layer.weight = embedding_weight
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, start_states: torch.Tensor, end_states: torch.Tensor,
                input_ids: torch.Tensor, hidden_state: torch.Tensor):
        embedded = self.embedding(input_ids)
        if hidden_state is None:
            states = torch.cat([start_states, end_states], dim=-1)
            hidden_state = self.merge_state_layer(states)
        context = self.context_state_layer(torch.cat([start_states, end_states, embedded], dim=-1))
        gru_output = self.gru(context, hidden_state)
        # output_state = self.output_state_layer(torch.cat([start_states, end_states, gru_output], dim=-1))
        output = self.log_softmax(self.project_layer(gru_output))

        return output, gru_output

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class MLPWithLayerNorm(nn.Module):
    def __init__(self, hidden_size:int, input_size:int, hidden_act:str="gelu"):
        super(MLPWithLayerNorm, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.non_lin1 = ACT2FN[hidden_act]
        self.layer_norm1 = BertLayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.non_lin2 = ACT2FN[hidden_act]
        self.layer_norm2 = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden):
        return self.layer_norm2(self.non_lin2(self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))))

class Corrector(nn.Module):
    def __init__(self, hidden_size:int ,word_embedding_weight: torch.Tensor, enable_mid_state:bool = True,
                 max_decode_length:int=20, pos_embedding_dim:int = 200):
        super(Corrector, self).__init__()
        self.enable_mid_state = enable_mid_state
        self.position_embeddings = nn.Embedding(max_decode_length, pos_embedding_dim)
        if enable_mid_state:
            self.mlp_layer_norm = MLPWithLayerNorm(word_embedding_weight.size(1), hidden_size * 3 + pos_embedding_dim)
        else:
            self.mlp_layer_norm = MLPWithLayerNorm(word_embedding_weight.size(1), hidden_size * 2 + pos_embedding_dim)
        self.project_layer = nn.Linear(word_embedding_weight.size(1),
                                        word_embedding_weight.size(0),
                                        bias=False)
        self.project_layer.weight = word_embedding_weight
        self.bias = nn.Parameter(torch.zeros(word_embedding_weight.size(0)))
        self.max_decode_length = max_decode_length
        self.emtpy_state = nn.Parameter(torch.rand((hidden_size,), dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, words_states:torch.Tensor, patch_idx: torch.Tensor, patch_start_pos: torch.Tensor,
                patch_end_pos, patch_mid_pos=None, patch_ids:torch.Tensor=None):

        
        start_states = words_states[patch_idx, patch_start_pos]
        end_states = words_states[patch_idx, patch_end_pos]
        if patch_mid_pos is not None:
            mid_states = words_states[patch_idx, patch_mid_pos]
            empty_tag = torch.tensor(-1, device=mid_states.device)
            empty_idx, empty_pos = torch.nonzero(patch_mid_pos==empty_tag, as_tuple=True)
            if empty_idx.size(0) > 0:
                mid_states[empty_idx, empty_pos] = self.emtpy_state
            mask = (patch_mid_pos != -2).unsqueeze(-1)
            mid_len = mask.sum(1)
            mid_states = (mid_states*mask).sum(1)/mid_len

        patch_num = start_states.size()[0]
        # pair states: patch num, decode_length, dim
        left_hidden = start_states.repeat(1, self.max_decode_length, 1)
        right_hidden = end_states.repeat(1, self.max_decode_length, 1)
        
        # (decode_length, dim)
        position_embeddings = self.position_embeddings.weight
        if self.enable_mid_state:
            mid_hidden = mid_states.unsqueeze(1).repeat(1, self.max_decode_length, 1)
            states = torch.cat((left_hidden, mid_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(patch_num, 1, 1)), -1)
        else:
            states = torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(patch_num, 1, 1)), -1)
        hidden_states = self.mlp_layer_norm(states)
        # target scores : patch_num, decode_length, vocab_size

        target_logits = self.project_layer(hidden_states) + self.bias

        loss = None
        if patch_ids is not None:
            loss = self.criterion(target_logits.view(-1, target_logits.size(-1)), patch_ids.view(-1))
        return target_logits, loss
