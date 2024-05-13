# pid_collate_fn과 akt_trainer return 값 개수가 일치
# train_akt.sh
# 터미널 실행시 sh train_akt.sh 명령어 실행

#!/bin/bash

dataset_names=("algebra2005_pid" "algebra2006_pid" "assist2009_pid" "assist2012_pid" "assist2017_pid")  
# "dkt_assist2009_pid" 는 빼고 돌리기
# 데이터셋 이름 리스트
model_names=("akt")  
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