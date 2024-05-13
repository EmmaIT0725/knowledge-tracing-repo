dataset_name = "algebra2005"

python \
train.py \
--model_fn dkt.pth \
--model_name dkt \
--dataset_name algebra2005 \
--batch_size 512 \
--fivefold True \
--n_epochs 100

model_names="dkt"

for model_name in ${model_names}
do
    python \
    train.py \
    --model_fn ${model_name}.pth \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --batch_size 512 \
    --fivefold True \
    --n_epochs 100
done
