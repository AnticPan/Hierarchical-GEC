from transformers import AutoTokenizer
from typing import List, Dict, Union
from utils.structure import Token
import re
import os

class Tokenizer:
    def __init__(self, bert_dir: str, lower_case: bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_dir, do_lower_case=lower_case)
        self.tokenizer.bos_token = "[CLS]"
        self.tokenizer.eos_token = "[SEP]"
        self.vocab = self.tokenizer.vocab
        self.global_oov = {}
        self._enable_unused()
    
    @property
    def PATCH_START_ID(self):
        return self.vocab[self.special_tokens_map["<SOP>"]]
    
    @property
    def PATCH_END_ID(self):
        return self.vocab[self.special_tokens_map["<EOP>"]]
    
    @property
    def PATCH_EMPTY_ID(self):
        return self.vocab[self.special_tokens_map['<NONE>']]

    def _enable_unused(self):
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[unused%d]" % i for i in range(1, 100)]})
        self.special_tokens = ["<SOP>", "<EOP>", "<NONE>", "<NUM>", "<URL>",
                          "'ll", "'s", "'ve", "n't", "'m", "'re", "``", "wo", "'d", "nbsp"]
        self.special_tokens_map = {token: "[unused%d]" % (
            i+1) for i, token in enumerate(self.special_tokens)}
        self.special_token_ids = [self.vocab["[unused%d]"% (i+1)] for i in range(len(self.special_tokens))]

    def _compress_word(self, word: str):
        ptr = None
        new_word = ""
        for char in word:
            if char != ptr:
                new_word += char
                ptr = char
        return new_word

    def _handle_oov(self, word: str, word_piece_num_limit: int = 5):
        # 1. handle special tokens
        if word in self.special_tokens_map:
            return [self.vocab[self.special_tokens_map[word]]]
        # 2. handle number
        if re.match("[0-9\.:%,]*$", word):
            return [self.vocab[self.special_tokens_map["<NUM>"]]]
        # 3. handle url
        # word == "URL" ?
        if re.match("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", word):
            return [self.vocab[self.special_tokens_map["<URL>"]]]
        # 4. handle case
        word_piece_ids = self.tokenizer.encode(word, add_special_tokens=False)
        word_piece_num = len(word_piece_ids)
        if word.isalpha():
            format_words = [
                word.lower(), word.lower().capitalize(), word.upper()]
            format_ids = tuple(map(lambda x: self.tokenizer.encode(
                x, add_special_tokens=False), format_words))
            format_ids_lens = [len(ids) for ids in format_ids]
            min_ids_len = min(format_ids_lens)
            min_ids_len_index = format_ids_lens.index(min_ids_len)
            if min_ids_len <= word_piece_num_limit and min_ids_len < word_piece_num:
                return format_ids[min_ids_len_index]
            # 5. handle repeat one char
            compressed_word = self._compress_word(word.lower())
            if compressed_word in self.vocab:
                return [self.vocab[compressed_word]]
        # 6. handle remain long words
        if word_piece_num > word_piece_num_limit:
            idx = min(99, len(self.special_tokens_map)+word_piece_num)
            return [self.vocab["[unused%d]" % idx]]
        else:
            return word_piece_ids

    @property
    def START_TOKEN(self):
        return Token("[CLS]", [self.tokenizer.bos_token_id], 0, 1)

    def encode(self, words: List[str], is_patch: bool = False):
        if is_patch and words == []:
            return [Token('', [self.PATCH_EMPTY_ID], 0, 1)], None
        tokens = []
        if not is_patch:
            tokens.append(self.START_TOKEN)
        oovs = {}
        ptr = 1
        for word in words:
            if word == "":
                continue
            elif word in self.vocab:
                tokens.append(
                    Token(word, [self.vocab[word]], ptr, ptr+1))
                ptr += 1
            else:
                if word in self.global_oov:
                    ids = self.global_oov[word]
                else:
                    ids = self._handle_oov(word)
                    self.global_oov[word] = ids
                # ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 1 and not is_patch:
                    oov_id = ids[0]
                    if oov_id in oovs:
                        oovs[oov_id].append([ptr, word])
                    else:
                        oovs[oov_id] = [[ptr, word]]
                tokens.append(Token(word, ids, ptr, ptr+len(ids)))
                ptr += len(ids)
        if not is_patch:
            tokens.append(
                Token("[SEP]", [self.tokenizer.eos_token_id], ptr, ptr+1))
        if oovs == {}:
            oovs = None
        return tokens, oovs
    
    def _find_closest_oov(self, pos:int, items:List[Union[int, str]]):
        if len(items) == 1:
            return items[0][1]
        else:
            closest = 0
            min_distance = 1000
            for i, item in enumerate(items):
                distance = abs(item[0]-pos)
                if distance < min_distance:
                    closest = i
                    min_distance = distance
            return items[closest][1]

    def decode(self, ids:List[int], patch_start:int, oovs:Dict[int, List[Union[int, str]]]):
        in_vocab_ids = []
        output_words = []
        for i, token_id in enumerate(ids):
            if oovs and token_id in oovs:
                if in_vocab_ids:
                    output_words.append(self.tokenizer.decode(in_vocab_ids, clean_up_tokenization_spaces=False).strip())
                    in_vocab_ids = []
                word = self._find_closest_oov(patch_start, oovs[token_id]).strip()
                if word not in ["<EOP>", "<NONE>"]:
                    output_words.append(word)
                else:
                    break
            elif token_id in self.special_token_ids:
                if in_vocab_ids:
                    output_words.append(self.tokenizer.decode(in_vocab_ids, clean_up_tokenization_spaces=False).strip())
                    in_vocab_ids = []
                word = self.special_tokens[token_id-1]
                if word not in ["<EOP>", "<NONE>"]:
                    output_words.append(word)
                else:
                    break
            else:
                in_vocab_ids.append(token_id)
        if in_vocab_ids:
            output_words.append(self.tokenizer.decode(in_vocab_ids, clean_up_tokenization_spaces=False).strip())

        return " ".join(output_words)

    def save(self, save_dir: str, epoch: int = None):
        if epoch:
            save_dir = os.path.join(save_dir,"epoch-%d"%epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.tokenizer.save_vocabulary(save_dir)

  
