import grpc
from grpc_proto import GEC_pb2, GEC_pb2_grpc
from concurrent import futures

import argparse
import torch
import os
import time
from model.patcher import Patcher
from utils.tokenizer import Tokenizer
from utils.patch_handler import Patch_handler
from data_loader import Dataset
from utils.structure import Example, Batch, lists2tensor, BIO
from predict import predict_a_batch


class ComputeServicer(GEC_pb2_grpc.ComputeServicer):
    def __init__(self, model, tokenizer, patch_handler, args) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.patch_handler = patch_handler
        self.args = args

    def make_example(self, sentence):
        source_words = list(filter(lambda x:x!='',sentence.strip().split(" ")))
        source_tokens, oovs = self.tokenizer.encode(source_words, is_patch=False)
        example = Example(source_tokens, None, oovs, None)
        return example

    def make_batch(self, examples):
        pad_token_id = 0
        input_ids = []
        for i, example in enumerate(examples):
            input_tokens = example.tokens
            ids = []
            for token in input_tokens:
                ids.extend(token.ids)
            input_ids.append(ids)

        input_max_len = max([len(id_list) for id_list in input_ids])
        input_ids = lists2tensor(input_ids, input_max_len, 512, 0)
        attention_mask = torch.full(
            input_ids.size(), pad_token_id, dtype=torch.bool)
        attention_mask[torch.where(input_ids != pad_token_id)] = 1
        token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        target_tfs = None
        error_example_mask = None
        target_labels = None
        target_starts = None
        target_ends = None
        target_ids = None
        return Batch(examples, input_ids, attention_mask, token_type_ids, target_tfs,
                        target_labels, error_example_mask, target_starts, target_ends, target_ids)

    def predict(self, sentences):
        examples = [self.make_example(sent) for sent in sentences]
        batch_num = int((len(sentences)-0.1)//args.batch_size+1)
        results = []
        for i in range(batch_num):
            batch = self.make_batch(examples[i*args.batch_size:(i+1)*args.batch_size])
            batch = Dataset.to_device(batch)
            batch_results, times = predict_a_batch(self.model, batch, self.tokenizer, self.patch_handler, self.args)
            results.extend(batch_results)
        return results    

    def GEC(self,request,ctx):
        print(len(request.text))
        num = len(request.text)
        labels = []
        gec_results = self.predict(request.text)
        result = [r['output'][6:-6].strip() for r in gec_results]
        return GEC_pb2.GECReply(result=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=None)

    parser.add_argument("-lower_case", default=False, action="store_true")

    parser.add_argument("-discriminating", default=True, action="store_false")
    parser.add_argument("--discriminating_threshold", default=0.5, type=float)

    parser.add_argument("-detecting", default=True, action="store_false")
    parser.add_argument("-use_lstm", default=False, action="store_true")
    parser.add_argument("-use_crf", default=True, action="store_false")

    parser.add_argument("-correcting", default=True, action="store_false")
    parser.add_argument("--max_decode_step", default=4, type=int)

    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in args.gpus])

    model = Patcher(args.model_dir, 
                    discriminating=args.discriminating,
                    detecting=args.detecting,
                    correcting=args.correcting,
                    use_crf=args.use_crf,
                    use_lstm=args.use_lstm)
    tokenizer = Tokenizer(args.model_dir, args.lower_case)
    patch_handler = Patch_handler(tokenizer.PATCH_EMPTY_ID, dir_del=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ComputeServicer(model, tokenizer, patch_handler, args)
    GEC_pb2_grpc.add_ComputeServicer_to_server(servicer, server)
    server.add_insecure_port('127.0.0.1:19999')
    server.start()
    try:
        print("running...")
        time.sleep(1000)
    except KeyboardInterrupt:
        print("stopping...")
        server.stop(0)