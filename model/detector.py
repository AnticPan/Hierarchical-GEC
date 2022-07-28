import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

class Detector(nn.Module):
    def __init__(self, hidden_size:int, output_dim: int = 2, drop_out: float = 0.3,
                 use_punish: bool = False, use_crf: bool = False, use_lstm: bool = False):
        super(Detector, self).__init__()
        self.hidden_size = hidden_size
        self.detect_layer = nn.Sequential(nn.Dropout(drop_out),
                                          nn.Linear(self.hidden_size, output_dim))
        self.use_lstm = use_lstm
        self.use_crf = use_crf
        if use_lstm:
            self.lstm_layer = nn.LSTM(self.hidden_size, self.hidden_size,
                                      num_layers=1, batch_first=True, bidirectional=True)
            self.merge_layer = nn.Linear(self.hidden_size*2, self.hidden_size)
        if use_crf:
            self.crf_layer = CRF(output_dim, batch_first=True)
        elif use_punish:
            self.criterion = None
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            # self.criterion = focal_loss(num_classes=output_dim)
    
    def inference(self, labeling_output:torch.Tensor, masks:torch.Tensor):
        if self.use_crf:
            list_predict_labels = self.crf_layer.decode(
                labeling_output, masks)
        else:
            predict_labels = torch.softmax(
                labeling_output, dim=-1).argmax(dim=-1)
            list_predict_labels = [labels[mask].cpu().tolist() for labels, mask in
                                   zip(predict_labels, masks)]
        return list_predict_labels

    def forward(self, words_states_no_cls: torch.Tensor, mask: torch.Tensor, target_labels: torch.Tensor = None):
        if self.use_lstm:
            self.lstm_layer.flatten_parameters()
            lengths = torch.sum(mask, dim=-1)
            words_states_no_cls = pack_padded_sequence(words_states_no_cls, lengths,
                                                  batch_first=True, enforce_sorted=False)
            words_states_no_cls, _ = self.lstm_layer(words_states_no_cls)
            words_states_no_cls, _ = pad_packed_sequence(words_states_no_cls, batch_first=True, total_length=mask.size()[-1])
            words_states_no_cls = self.merge_layer(words_states_no_cls)
        labeling_output = self.detect_layer(words_states_no_cls)
        loss = None
        if target_labels is not None:
            #target_labels = target_labels
            if self.use_crf:
                
                loss = -1 * \
                    self.crf_layer(labeling_output, torch.where(target_labels==-100, torch.zeros(1, dtype=torch.long, device=target_labels.device), target_labels),
                                   mask, reduction='none')
            else:
                loss = self.criterion(labeling_output.view(-1, labeling_output.size(-1)), target_labels.reshape(-1)).view(target_labels.size())
                loss = loss.sum(-1)/mask.sum(-1)
        return labeling_output, loss
