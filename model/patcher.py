import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .corrector import Corrector
from .discriminator import Discriminator
from .detector import Detector

import logging
logging.getLogger().setLevel(logging.INFO)

class Patcher(nn.Module):
    def __init__(self, bert_dir: str,  
                 discriminating: bool = True, detecting: bool = True, correcting:bool=True,
                 use_crf: bool = False, use_lstm: bool = False):
        super(Patcher, self).__init__()
        self.label_num = 6
        config = AutoConfig.from_pretrained(bert_dir)
        # config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(bert_dir, config = config)
        self.hidden_size = self.encoder.config.hidden_size
        if discriminating:
            self.discriminator = Discriminator(self.hidden_size)
        else:
            self.discriminator = None 
        if detecting:
            self.detector = Detector(hidden_size=self.hidden_size, 
                                    use_crf=use_crf,
                                    use_lstm=use_lstm, 
                                    output_dim=self.label_num)
        else:
            self.detector = None
        if correcting:
            self.corrector = Corrector(self.hidden_size, self.encoder.embeddings.word_embeddings.weight, enable_mid_state=True)
        else:
            self.corrector = None
        self.restore(bert_dir)

    def restore(self, pretrained_dir: str):
        pretrained_discriminator = os.path.join(pretrained_dir, "discriminator.pt")
        if os.path.exists(pretrained_discriminator) and self.discriminator:
            self.discriminator.load_state_dict(
                torch.load(pretrained_discriminator, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        pretrained_detector = os.path.join(pretrained_dir, "detector.pt")
        if os.path.exists(pretrained_detector) and self.detector:
            self.detector.load_state_dict(
                torch.load(pretrained_detector, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        pretrained_corrector = os.path.join(pretrained_dir, "corrector.pt")
        if os.path.exists(pretrained_corrector) and self.corrector:
            self.corrector.load_state_dict(
                torch.load(pretrained_corrector, map_location="cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, task:str, data):
        if task == "encode":
            encoder_outputs = self.encoder(input_ids=data["input_ids"],
                                           attention_mask=data["masks"],
                                           token_type_ids=data["token_type_ids"])
            return encoder_outputs
        elif task == "discriminate":
            predict_tfs_logits, discriminator_loss = self.discriminator(data["first_hiddens"], data["target_tfs"])
            return predict_tfs_logits, discriminator_loss
        elif task == "detect":
            labeling_output, detector_loss = self.detector(encoder_output=data["encoder_output"],
                                                           masks=data["masks"],
                                                           target_labels=data["target_labels"])
            return labeling_output, detector_loss
        elif task == "correct":
            decoder_output, decoder_loss = self.corrector(data["patch_start_states"],
                                                        data["patch_end_states"],
                                                        data["length"],
                                                        mid_states=data["patch_mid_states"],
                                                        target_ids=data["patch_ids"])
            return decoder_output, decoder_loss

    def save(self, save_dir: str, epoch: int = None):
        if epoch:
            save_dir = os.path.join(save_dir,"epoch-%d"%epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.encoder.save_pretrained(save_dir)
        if self.discriminator is not None:
            torch.save(self.discriminator.state_dict(),
                    os.path.join(save_dir, "discriminator.pt"))
        if self.detector is not None:
            torch.save(self.detector.state_dict(),
                    os.path.join(save_dir, "detector.pt"))
        if self.corrector is not None:
            torch.save(self.corrector.state_dict(),
                       os.path.join(save_dir, 'decoder.pt'))
