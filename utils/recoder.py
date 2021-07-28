from typing import List, NamedTuple, Set, Dict, Union, Tuple
import os
import copy
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .structure import Example, lists2tensor
from .patch_handler import Patch_handler

class Statistic(object):
    def __init__(self, log_dir: str, 
                 discriminating:bool,
                 detecting:bool,
                 correcting:bool,
                 max_decode_step:int = 5,
                 patch_handler:Patch_handler=None):
        self.discriminating = discriminating
        self.detecting = detecting
        self.correcting = correcting
        self.label_num = 6
        self.patch_handler=patch_handler
        self.log_dir = log_dir
        self.max_decode_step = max_decode_step
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        board_dir = os.path.join(log_dir, "board")
        self.writer = SummaryWriter(
            board_dir, comment="Logging start at %s." % (str(datetime.now())))
        self.reset("train", 1)

    def reset(self, mode: str, epoch: int):
        self.mode = mode
        self.epoch = epoch
        self.step = 0
        self.corrector_step = 0
        self.discriminating_losses = 0
        self.detecting_losses = 0
        self.correcting_losses = 0
        self.count_by_tf = {0:{"total": 0, "predict": 0, "right": 0},
                            1:{"total": 0, "predict": 0, "right": 0}}
        self.count_by_label = {}
        for idx in range(self.label_num):
            self.count_by_label[idx] = {
                "total": 0, "predict": 0, "right": 0}  # total, predict, right
        # total, right, conditional_right
        self.count_by_example = {"total": 0, "right": 0, "all_t_in_p": 0,
                                 "total_no_patch": 0, "predict_no_patch": 0, "right_no_patch": 0}
        # total, predict, right, conditional_right, incomplete
        self.count_by_patch = {"total": 0, "predict": 0,
                               "right": 0, "incomplete": 0, "t_in_p": 0,
                               "p_in_t": 0, "cross": 0, "out": 0}

    def get_current_log(self):
        log = {}
        if self.discriminating:
            log.update({"Dis_L":self.Discriminating_Loss,
                        "Dis_A":self.Discriminating_Acc})
        if self.detecting:
            P1, R1 = self.get_patch_PR(count_t_in_p=False)
            P2, R2 = self.get_patch_PR(count_t_in_p=True)
            # "Acc":self.statistic.Detecting_Acc})
            log.update({"Det-L": self.Detecting_Loss,
                        "P-P1": P1,"P-R1": R1,
                        "P-P2": P2,"P-R2": R2})
        if self.correcting:
            log["Dec-L"] = self.Decoding_Loss

        return log
    
    def update_discriminator(self, step:int, loss:float, predicts:List[int], targets:List[int]):
        assert len(predicts) == len(targets)
        self.step = step
        self.discriminating_losses += loss
        self.writer.add_scalar(
            "Loss-Discriminator/%s/epoch-%d" % (self.mode, self.epoch), self.discriminating_losses/step, step)
        one_batch_tf = {0:{"total": 0, "predict": 0, "right": 0},
                        1:{"total": 0, "predict": 0, "right": 0}}
        for p, t in zip(predicts, targets):
            self.count_by_tf[t]["total"] += 1
            self.count_by_tf[p]["predict"] += 1
            one_batch_tf[t]["total"] += 1
            one_batch_tf[p]["predict"] += 1
            if p == t:
                self.count_by_tf[t]["right"] += 1
                one_batch_tf[t]["right"] += 1
        
        # T_P, T_R = self._get_PR(one_batch_tf[1])
        # F_P, F_R = self._get_PR(one_batch_tf[0])
        # F1_T_P_and_F_R = 2*T_P*F_R/(T_P + F_R)

    def update_detector_loss(self, step:int, loss:float):
        self.step = step
        self.detecting_losses += loss
        self.writer.add_scalar(
            "Loss-Detecter/%s/epoch-%d" % (self.mode, self.epoch), self.detecting_losses/step, step)

    def update_detector(self, predicts: List[List[int]], targets: List[List[int]], es):
        self.count_by_example["total"] += len(predicts)
        for example in predicts:
            for label in example:
                self.count_by_label[label]["predict"] += 1

        assert len(targets) == len(predicts)
        # patch_start_pos = [[], []]
        # patch_end_pos = [[], []]
        # rewards = []
        # predict_patch_ids = []
        for i, (predict, target, example) in enumerate(zip(predicts, targets, es)):
            assert len(predict) == len(target)
            for p, t in zip(predict, target):
                self.count_by_label[t]["total"] += 1
                if p == t:
                    self.count_by_label[t]["right"] += 1
            predict_incomplete, predict_patches = self.patch_handler.get_patch_sets(predict)
            target_incomplete, target_patches = self.patch_handler.get_patch_sets(target)
            if predict_patches:
                predict_patches = self.patch_handler.align_patch(len(predict), predict_patches, example)

            if predict == target:
                self.count_by_example["right"] += 1
            elif self._ps_cover_ts(predict_patches, target_patches):
                self.count_by_example["all_t_in_p"] += 1

            if target_patches == []:
                self.count_by_example["total_no_patch"] += 1
            if predict_patches == []:
                self.count_by_example["predict_no_patch"] += 1
                if target_patches == []:
                    self.count_by_example["right_no_patch"] += 1

            self.count_by_patch["total"] += len(target_patches)
            self.count_by_patch["predict"] += len(predict_patches)
            self.count_by_patch["incomplete"] += predict_incomplete

            # predict_right = 0
            for predict_patch in predict_patches:
                state = self._relation_p_and_ts(predict_patch, target_patches)
                self.count_by_patch[state] += 1
                # if state in self.rl_goal:
                #     predict_right += 1
            # filtered_patches, patch_ids = self.patch_handler.get_target_ids(predict_patches, example)
            # for predict_start, predict_end in filtered_patches:
            #     patch_start_pos[0].append(i)
            #     patch_start_pos[1].append(predict_start)
            #     patch_end_pos[0].append(i)
            #     patch_end_pos[1].append(predict_end)
            # rewards.append(self._get_reward(len(predict_patches), len(target_patches), predict_right, self.rl_beta))
            # if patch_ids:
            #     predict_patch_ids.extend(patch_ids)
        # if predict_patch_ids:
        #     max_len = max([len(ids) for ids in predict_patch_ids])
        #     predict_patch_ids = lists2tensor(predict_patch_ids, max_len, self.max_decode_step, -100).cuda()
        # else:
        #     predict_patch_ids = None
        # return patch_start_pos, patch_end_pos, rewards, predict_patch_ids

    def update_corrector(self, loss: float):
        self.corrector_step += 1
        self.correcting_losses += loss
        self.writer.add_scalar(
            "Loss-Decoder/%s/epoch-%d" %
            (self.mode, self.epoch), self.correcting_losses/self.corrector_step, self.step)

    def _relation_p_and_ts(self, p: Tuple[int], ts: List[Tuple[int]]):
        assert len(p) == 2
        predict_start, predict_end = p
        cross = False
        for (target_start, target_end) in ts:
            if target_start == predict_start and target_end == predict_end:
                return "right"
            elif target_start>= predict_start and target_end <= predict_end:
                return "t_in_p"
            elif target_start <= predict_start and target_end >= predict_end:
                return "p_in_t"
            elif (target_end <= predict_end and target_end >= predict_start) or \
                (target_start >= predict_start and target_start<= predict_end):
                cross = True
        if cross:
            return "cross"
        return "out"

    def _ps_cover_ts(self, ps: List[Tuple[int]], ts: List[Tuple[int]]):
        if len(ps) == 0 or len(ts) == 0:
            return False
        for p in ps:
            if self._relation_p_and_ts(p, ts) in ["out", "cross", "p_in_t"]:
                return False
        return True

    def get_patch_PR(self, count_t_in_p:bool = False):
        if count_t_in_p:
            patch_dict = copy.copy(self.count_by_patch)
            patch_dict["right"]+= patch_dict["t_in_p"]
            return self._get_PR(patch_dict)
        else:
            return self._get_PR(self.count_by_patch)

    def _get_PR(self, count_dict: Dict[str, int]):
        total = count_dict["total"]
        predict = count_dict["predict"]
        right = count_dict["right"]
        P = 0 if predict == 0 else 1.0*right/predict
        R = 0 if total == 0 else 1.0*right/total
        return P, R

    # def _get_reward(self, predict:int, target:int, right:int, beta:float=1.0):
    #     if target == 0:
    #         return 1
    #     if right == 0:
    #         return 10
    #     P = 1.0*right/predict
    #     R = 1.0*right/target
    #     F = (1+beta**2)*(P*R)/((beta**2)*P+R)
    #     if F<0.1:
    #         return 10
    #     return 1.0/F
        
    @property
    def Discriminating_Acc(self):
        predict_num = sum([self.count_by_tf[idx]["predict"]
                           for idx in self.count_by_tf])
        right_num = sum([self.count_by_tf[idx]["right"]
                         for idx in self.count_by_tf])
        try:
            return right_num/predict_num
        except ZeroDivisionError as e:
            return 0

    @property
    def Discriminating_Loss(self):
        try:
            return self.discriminating_losses/self.step
        except ZeroDivisionError:
            return -1

    @property
    def Detecting_Acc(self):
        predict_num = sum([self.count_by_label[idx]["predict"]
                           for idx in self.count_by_label])
        right_num = sum([self.count_by_label[idx]["right"]
                         for idx in self.count_by_label])
        try:
            return right_num/predict_num
        except ZeroDivisionError as e:
            return 0

    @property
    def Detecting_Loss(self):
        try:
            return self.detecting_losses/self.step
        except ZeroDivisionError:
            return -1

    @property
    def Decoding_Loss(self):
        try:
            return self.correcting_losses/self.corrector_step
        except ZeroDivisionError:
            return -1

    def _get_discriminating_state(self):
        state = ""
        state += "TF statistic\n\n"
        for idx in [0, 1]:
            state += "LABEL-%s: " % bool(idx)
            for key in self.count_by_tf[idx]:
                state += "%s:%d\t" % (key, self.count_by_tf[idx][key])
            state += "P:%.4f\tR:%.4f\n\n" % (
                self._get_PR(self.count_by_tf[idx]))
        state += "\nDiscriminating Acc: %.4f\n\n" % self.Discriminating_Acc
        state += "Discriminating Loss: %.4f\n" % self.Discriminating_Loss
        return state

    def _get_detecting_state(self):
        state = ""
        state += "Label statistic\n\n"
        for idx in self.count_by_label:
            state += "LABEL-%d: " % (idx)
            for key in self.count_by_label[idx]:
                state += "%s:%d\t" % (key, self.count_by_label[idx][key])
            state += "P:%.4f\tR:%.4f\n\n" % (
                self._get_PR(self.count_by_label[idx]))

        state += "\nPatch statistic\n\n"
        for key in self.count_by_patch:
            state += "%s:%d\t" % (key, self.count_by_patch[key])
        state += "P1:%.4f\tR1:%.4f\t" % (self.get_patch_PR(False))
        state += "P2:%.4f\tR2:%.4f\n\n" % (self.get_patch_PR(True))

        state += "\nExample statistic\n\n"
        for key in self.count_by_example:
            state += "%s:%d\t" % (key, self.count_by_example[key])
        if self.count_by_example["total"]:
            state += "P:%.4f\n\n" % (
                self.count_by_example["right"]/self.count_by_example["total"])
        else:
            state += "P:0\n\n"
        state += "\nLabeling Acc: %.4f\n\n" % self.Detecting_Acc
        state += "Labeling Loss: %.4f\n" % self.Detecting_Loss
        return state
    
    def _get_decoding_state(self):
        state = "Decoding Loss: %.4f\n"%self.Decoding_Loss
        return state

    def record_one_example(self, step: int, example: Example, predict_label: List[int], target_label: List[int]):
        text = "tokens:%s\n\npatches:%s\n\npredict_label:%s\n\ntarget_label:%s" %\
            (example.tokens, example.patches, str(
                predict_label), str(target_label))
        self.writer.add_text("record/%s/epoch-%d" %
                             (self.mode, self.epoch), text, step)
        self.writer.flush()

    def save(self):
        discriminating_state = self._get_discriminating_state()
        detecting_state = self._get_detecting_state()
        decoding_state = self._get_decoding_state()
        self.writer.add_text("statistic/%s/epoch-%d" %
                             (self.mode, self.epoch), discriminating_state+detecting_state+decoding_state)
        self.writer.flush()
        save_dir = os.path.join(self.log_dir, "statistic")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(
            save_dir, "%s_epoch_%d.log" % (self.mode, self.epoch))
        with open(save_file, 'w') as f:
            f.write(discriminating_state+detecting_state+decoding_state)
