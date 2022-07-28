from utils import Example, Token, lists2tensor, BIO, Patch
from typing import List, Tuple
import copy
import torch

class Patch_handler:
    def __init__(self, empty_patch_id:int, dir_del:bool):
        self.empty_patch_id = empty_patch_id
        self.dir_del = dir_del

    def get_target_ids(self, predict_patches:List[Tuple[int]], example:Example, filter_cross:bool=True):
        filtered_patches = []
        patch_ids = []
        tokens = example.tokens
        source_ptr = 0
        target_patches = example.patches
        target_patch_ptr = 0
        for (predict_start, predict_end) in predict_patches:
            source_ids = []
            while source_ptr< len(tokens) and tokens[source_ptr].start != predict_start+1:
                source_ptr += 1
            while source_ptr< len(tokens) and tokens[source_ptr].start != predict_end:
                source_ids.extend(tokens[source_ptr].ids)
                source_ptr += 1
            target_ids = None
            is_cross = False
            if target_patches is not None:
                target_ids = copy.copy(source_ids)
                pre_offset = 0
                while target_patch_ptr<len(target_patches) and target_patches[target_patch_ptr].start < predict_start+1:
                    cur_patch_end = target_patches[target_patch_ptr].end
                    if cur_patch_end > predict_start+1 and cur_patch_end <= predict_end:
                        is_cross = True
                    target_patch_ptr += 1
                while target_patch_ptr<len(target_patches) and target_patches[target_patch_ptr].start <= predict_end:
                    target_patch = target_patches[target_patch_ptr]
                    offset = target_patch.start - (predict_start+1) + pre_offset
                    if target_patch.end <= predict_end:
                        ids = []
                        for token in target_patch.tokens:
                            ids.extend(token.ids)
                        patch_length = target_patch.end - target_patch.start
                        target_ids[offset:offset+patch_length] = ids
                        pre_offset += len(ids) - patch_length
                    else:
                        is_cross = True
                    target_patch_ptr += 1
            
            if target_ids is None:
                target_ids = source_ids
            elif len(target_ids) == 0:
                if self.dir_del:
                    target_ids = source_ids # for filter
                else:
                    target_ids = [self.empty_patch_id]
            if not filter_cross or target_ids!=source_ids or not is_cross:
                filtered_patches.append((predict_start, predict_end))
                patch_ids.append(target_ids)

        return filtered_patches, patch_ids


    def get_patch_pos(self, labels_list: List[List[int]]):
        patch_idx = []
        patch_start_pos = []
        patch_mid_pos = []
        patch_end_pos = []
        del_patches = []
        for idx, labels in enumerate(labels_list):
            _, patch_sets = self.get_patch_sets(labels)
            for start, end in patch_sets:
                if self.dir_del and BIO["B-R"] in labels[start:end-1]:
                    del_patches.append([idx, Patch(start, end, "",[self.empty_patch_id])])
                else:
                    patch_idx.append(idx)
                    patch_start_pos.append(start)
                    patch_end_pos.append(end)
                    if start+1 == end:
                        patch_mid_pos.append([-1])
                    else:
                        patch_mid_pos.append(list(range(start+1, end)))
        if len(patch_idx) == 0:
            patch_idx = None
            patch_start_pos = None
            patch_mid_pos = None
            patch_end_pos = None
        else:
            max_mid_pos_len = max(len(mid_pos) for mid_pos in patch_mid_pos)
            patch_idx = torch.tensor(patch_idx, dtype=torch.long).unsqueeze(1)
            patch_start_pos = torch.tensor(patch_start_pos, dtype=torch.long).unsqueeze(1)
            patch_end_pos = torch.tensor(patch_end_pos, dtype=torch.long).unsqueeze(1)
            patch_mid_pos = lists2tensor(patch_mid_pos, max_mid_pos_len, -2)
        return patch_idx, patch_start_pos, patch_mid_pos, patch_end_pos, del_patches

    def get_patch_sets(self, labels: List[int]):
        incomplete, patch_pos = self._get_BIO_patch_sets(labels)
        return incomplete, patch_pos
    
    def _get_BIO_patch_sets(self, labels: List[int]):
        pre_label = BIO["O"]
        incomplete = 0
        patch_pos = []
        merge = False
        patch_start = -1
        # the patch position here is added 1 for the hidden [CLS] token
        for i, label in enumerate(labels):
            if label == BIO["B-M"]:
                if pre_label in [BIO["O"], BIO["B-M"]]:
                    patch_pos.append([i,i+1])
            elif label in [BIO["B-R"], BIO["B-WS"]]:
                if pre_label in [BIO["O"], BIO["B-M"]]:
                    patch_start = i
                else:
                    incomplete += 1
            elif label in [BIO["I-R"], BIO["I-WS"]]:
                if pre_label == BIO["O"]:
                    incomplete += 1
            else:
                if patch_start != -1:
                    patch_pos.append([patch_start, i+1])
                    patch_start = -1
            pre_label = label
        if patch_start != -1:
            patch_pos.append([patch_start, i+1])
        return incomplete, patch_pos
