#!/bin/bash
model_dir=$1
valid_basename=dev
valid_file=data/$valid_basename.tsv
valid_src=data/$valid_basename.src
valid_ref_m2=data/$valid_basename.m2
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu  python predict.py --model_dir $model_dir --test_file $valid_file --output_dir /tmp/patcher/$model_dir --gpu 0 -detecting -correcting --batch_size 64
res=$(python grid.py --valid_file $valid_file --valid_pkl /tmp/patcher/$model_dir/$valid_basename.output.pkl)
thres=$(echo $res|cut -d" " -f3)
echo Threshold $thres
CUDA_VISIBLE_DEVICES=$gpu python predict.py --model_dir $model_dir --test_file $valid_src --output_dir $model_dir --gpu 0 --discriminating_threshold $thres --max_patch_len 4 --batch_size 64
../miniconda3/envs/py36/bin/errant_parallel -orig $valid_src -cor $model_dir/$valid_basename.predict -out $model_dir/$valid_basename.predict.m2
../miniconda3/envs/py36/bin/errant_compare -hyp $model_dir/$valid_basename.predict.m2 -ref $valid_ref_m2 | tee $model_dir/$valid_basename.m2.score
echo Threshold $thres >> $model_dir/$valid_basename.m2.score