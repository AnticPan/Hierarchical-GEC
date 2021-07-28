from .structure import Example, Token, lists2tensor, BIO, Patch
from typing import List, Tuple
import copy
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

    def align_patch(self, label_length: int, predict_patch_pos: List[Tuple[int]], example: Example):
        wordpiece_map = [0] * (label_length + 1) # plus 1 for [CLS]
        for token in example.tokens:
            if token.start >= label_length or token.end > label_length:
                break
            wordpiece_map[token.start:token.end] = list(range(token.end - token.start))
        aligned_patch_pos = []
        pre_end = -1
        for start, end in predict_patch_pos:
            if wordpiece_map[start + 1] == 0:
                aligned_start = start
            else:
                aligned_start = start - wordpiece_map[start] - 1
                assert aligned_start >= 0
            if wordpiece_map[end] != 0:
                aligned_end = end + 1
                while aligned_end < label_length and wordpiece_map[aligned_end] != 0:
                    aligned_end += 1
                if aligned_end == label_length:
                    aligned_end -= 1
            else:
                aligned_end = end
            if aligned_start < pre_end:
                pre_start = aligned_patch_pos[-1][0]
                aligned_patch_pos.pop()
                aligned_patch_pos.append((pre_start, aligned_end))
            else:
                aligned_patch_pos.append((aligned_start, aligned_end))
            pre_end = aligned_end
        return aligned_patch_pos

    def get_patch_pos(self, labels_list: List[List[int]], incor_ids: List[int]):
        start_pos = [[],[]]
        end_pos = [[],[]]
        del_patches = []
        for idx, labels in enumerate(labels_list):
            _, patch_sets = self.get_patch_sets(labels)
            for start, end in patch_sets:
                if self.dir_del:
                    if BIO["B-R"] in labels[start:end-1]:
                        del_patches.append([incor_ids[idx], Patch(start, end, [Token("",[self.empty_patch_id],0,1)])])
                start_pos[0].append(incor_ids[idx])
                start_pos[1].append(start)
                end_pos[0].append(incor_ids[idx])
                end_pos[1].append(end)
        if start_pos == [[],[]]:
            start_pos = None
            end_pos = None
        return start_pos, end_pos, del_patches

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
