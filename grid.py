import numpy as np
import pickle
import argparse

def search(tsv,pkl, threshold=None):
    with open(tsv) as f:
        tfs = []
        for line in f:
            source, *targets = line.strip().split("\t")
            if any(source == t for t in targets):
                tfs.append(1)
            else:
                tfs.append(0)
        tfs = np.array(tfs)
    with open(pkl,"rb") as f:
        es = pickle.load(f)
        ps = [e["tf_logit"] for e in es]
        ps = np.array(ps)

    assert len(tfs)==len(ps)
    
    acc=0
    best=0
    for i in range(101):
        thres = i*0.01
        p = ps>thres
        r = int(sum(p==tfs))
        a = r/len(tfs)
        if a > acc:
            acc = a
            best= thres
    return best, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--valid_pkl", type=str, required=True)
    args = parser.parse_args()
    thres, acc = search(args.valid_file, args.valid_pkl)
    print(args.valid_file)
    print("Threshold: %.2f"%thres)
    print("Accuarcy: %.2f"%acc)
