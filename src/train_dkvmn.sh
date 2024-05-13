# train_dkvmn.sh
# 터미널 실행시 sh train_dkvmn.sh 명령어 실행

#!/bin/bash

dataset_names=("algebra2005" "algebra2006" "assist2009" "assist2012" "assist2017")
# "ednet" 제외하고 돌리기  
# "assist2009_pid_diff"  데이터는 dkvmn으로만 돌려보기: collate 상으로 그렇게 돌아가도록 만들어진 것 같음.
# "dkt_assist2009_pid" "dkt_assist2009_pid_diff" 
# 데이터셋 이름 리스트
model_names=("dkvmn")  
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

