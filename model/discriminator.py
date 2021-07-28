import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden_size:int, sent_emb:bool = False):
        super(Discriminator, self).__init__()
        self.sent_emb = sent_emb
        self.discriminate_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                                nn.Sigmoid())
        self.criterion = nn.BCELoss()

    def forward(self, first_hiddens:torch.Tensor, target_tfs:torch.Tensor=None):
        predict_logits = self.discriminate_layer(first_hiddens).squeeze(dim=-1)
        loss = None
        if isinstance(target_tfs, torch.Tensor):
            # weight = target_tfs*0.9+0.1
            # criterion = nn.BCELoss(weight=weight)
            # self.criterion.weight = weight
            loss = self.criterion(predict_logits, target_tfs)
        
        return predict_logits, loss