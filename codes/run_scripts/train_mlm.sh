export CUDA_VISIBLE_DEVICES=0
cd /mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/mlm_score

model_name=roberta_base
model_path=/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/mlm_score/pretrain_models/roberta_base
dataset_name=wizard_of_wikipedia
data_path=/mnt/ai2lab/weishao4/programs/InterpretableKGC/data/${dataset_name}/hf_data
save_dir=/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/mlm_score/save_models/${model_name}_${dataset_name}
mkdir -p $save_dir
# --num_train_epochs $num_epochs \
python train.py \
    --model_name $model_name \
    --model_path $model_path \
    --data_path $data_path \
    --dataset_name $dataset_name \
    --output_dir_path $save_dir \
    --train_s 60000 \
    --train_e 83247