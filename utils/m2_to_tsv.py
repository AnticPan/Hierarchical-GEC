import argparse
import os
from tqdm import tqdm

def convert(file_path):
    tsv_examples = []
    with open(file_path,'r', encoding='utf-8') as f:
        examples = f.read().strip().split("\n\n")
        for example in tqdm(examples):
            source, *targets = example.split("\n")
            source = source[2:].strip()
            if targets:
                words = list(filter(lambda x:x!='',source.split(" ")))
                tsv_examples.append([source])
                pre_tagger = 0
                offset = 0
                for line in targets:
                    edits = line.strip().split("|||")
                    start, end = edits[0][2:].split(" ")
                    start = int(start)
                    end = int(end)
                    tagger = int(edits[-1])
                    fix = list(filter(lambda x:x!='',edits[2].split(" ")))
                    if tagger != pre_tagger:
                        tsv_examples[-1].append(" ".join(words))
                        offset = 0
                        words = list(filter(lambda x:x!='',source.split(" ")))
                    if start == -1 and end == -1:
                        tsv_examples[-1].append(source)
                    else:
                        words[start+offset:end+offset] = fix
                        offset = offset-(end-start)+len(fix)
                    pre_tagger = tagger
                target = " ".join(words)
                if target != source or pre_tagger==0:
                    tsv_examples[-1].append(target)
            else:
                tsv_examples.append([source, source])
    return tsv_examples

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2_file", type=str, required=True)
    parser.add_argument("--output_dir",type=str, required=True)
    args = parser.parse_args()
    tsv_examples = convert(args.m2_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    _, file_name = os.path.split(args.m2_file)
    if file_name.endswith(".m2"):
        file_name = file_name.replace(".m2",".tsv")
    else:
        file_name+=".tsv"
    with open(os.path.join(args.output_dir, file_name),'w') as f:
        for example in tsv_examples:
            f.write("\t".join(example)+"\n")