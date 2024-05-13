# run_all_datasets.sh
# 터미널 실행시 sh run_all_datasets.sh 명령어 실행

#!/bin/bash

dataset_names=("algebra2005" "algebra2005_pid" "algebra2006" "algebra2006_pid" "assist2009" "assist2009_pid" "assist2009_pid_diff" "dkt_assist2009_pid" "dkt_assist2009_pid_diff" "assist2012" "assist2012_pid" "assist2017" "assist2017_pid" "ednet")  
# 데이터셋 이름 리스트
model_names=("dkt" "dkvmn" "akt" "dkt_plus")  
# 모델 이름 리스트

for dataset_name in "${dataset_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        python \
        train.py \
        --model_fn ${model_name}.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset_name} \
        --optimizer adam \
        --crit binary_cross_entropy \
        --max_seq_len 100
    done
done

