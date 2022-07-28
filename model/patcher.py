import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .corrector import Corrector
from .discriminator import Discriminator
from .detector import Detector

import logging
logging.getLogger().setLevel(logging.INFO)


class PatcherOutput(object):
    def __init__(self) -> None:
        self.discriminator_logits = None
        self.discriminator_loss = None
        self.detector_logits = None
        self.detector_loss = None
        self.corrector_logits = None
        self.corrector_loss = None

class Patcher(nn.Module):
    def __init__(self, bert_dir: str,  
                 discriminating: bool = True, detecting: bool = True, correcting:bool=True,
                 use_crf: bool = False, use_lstm: bool = False, max_patch_len:int= 4):
        super(Patcher, self).__init__()
        self.label_num = 6
        config = AutoConfig.from_pretrained(bert_dir)
        # config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(bert_dir, config = config)
        self.hidden_size = self.encoder.config.hidden_size
        self.discriminating = discriminating
        self.detecting = detecting
        self.correcting = correcting
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
            self.corrector = Corrector(self.hidden_size, self.encoder.embeddings.word_embeddings.weight, enable_mid_state=True, max_decode_length=max_patch_len)
        else:
            self.corrector = None
        self.restore(bert_dir)

    def restore(self, pretrained_dir: str):
        pretrained_discriminator = os.path.join(pretrained_dir, "discriminator.pt")
        if os.path.exists(pretrained_discriminator) and self.discriminating:
            self.discriminator.load_state_dict(
                torch.load(pretrained_discriminator, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        pretrained_detector = os.path.join(pretrained_dir, "detector.pt")
        if os.path.exists(pretrained_detector) and self.detecting:
            self.detector.load_state_dict(
                torch.load(pretrained_detector, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        pretrained_corrector = os.path.join(pretrained_dir, "corrector.pt")
        if os.path.exists(pretrained_corrector) and self.correcting:
            self.corrector.load_state_dict(
                torch.load(pretrained_corrector, map_location="cuda" if torch.cuda.is_available() else "cpu"))


    def forward(self, input_ids, attention_mask, word_offsets, tf_labels=None, 
                bio_tags=None, patch_idx=None, patch_start_pos=None, patch_mid_pos=None,
                patch_end_pos=None, patch_ids=None) -> PatcherOutput:
        output = PatcherOutput()
        cls_states, words_states = self.encode(input_ids, attention_mask, word_offsets)

        if self.discriminating:
            output.discriminator_logits, output.discriminator_loss = self.discriminate(cls_states, tf_labels)
        
        if self.detecting or self.correcting:
            if self.detecting:
                mask = (word_offsets!=-1)[:,1:]
                output.detector_logits, output.detector_loss = self.detect(words_states, mask, bio_tags)
            if self.correcting and patch_idx is not None:
                output.corrector_logits, output.corrector_loss = self.correct(words_states,
                                                                                patch_idx,
                                                                                patch_start_pos,
                                                                                patch_end_pos,
                                                                                patch_mid_pos=patch_mid_pos,
                                                                                patch_ids=patch_ids)
        return output

    def encode(self, input_ids, attention_mask, word_offsets):
        encoder_outputs = self.encoder(input_ids, attention_mask)
        batch_size = input_ids.size(0)
        batch_idx = torch.arange(0, batch_size, dtype=torch.long, device=input_ids.device).unsqueeze(1)
        cls_states = encoder_outputs[1]
        tokens_states = encoder_outputs[0]
        words_states = tokens_states[batch_idx, word_offsets]

        return cls_states, words_states
    
    def discriminate(self, cls_states, tf_labels=None):
        logits, loss = self.discriminator(cls_states, tf_labels)
        return logits, loss

    def detect(self, words_states, mask, bio_tags=None):
        words_states_no_cls = words_states[:,1:,:]
        if bio_tags is not None:
            bio_tags = bio_tags[:,1:]
        logits, loss = self.detector(words_states_no_cls, mask, bio_tags)
        return logits, loss

    def correct(self, words_states, patch_idx, patch_start_pos, patch_end_pos, patch_mid_pos, patch_ids=None):
        logits, loss = self.corrector(words_states,
                                        patch_idx,
                                        patch_start_pos,
                                        patch_end_pos,
                                        patch_mid_pos=patch_mid_pos,
                                        patch_ids=patch_ids)
        return logits, loss

    def save(self, tokenizer, save_dir: str, epoch: int = None):
        if epoch:
            save_dir = os.path.join(save_dir,"epoch-%d"%epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.encoder.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        if self.discriminator is not None:
            torch.save(self.discriminator.state_dict(),
                    os.path.join(save_dir, "discriminator.pt"))
        if self.detector is not None:
            torch.save(self.detector.state_dict(),
                    os.path.join(save_dir, "detector.pt"))
        if self.corrector is not None:
            torch.save(self.corrector.state_dict(),
                       os.path.join(save_dir, 'corrector.pt'))
