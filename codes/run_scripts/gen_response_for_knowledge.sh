export CUDA_VISIBLE_DEVICES=3
cd /mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/gen_response

model_name=flan_t5_base
model_path=/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/kgc_model/saved_checkpoints/wizard_of_wikipedia_flan_t5_base/checkpoint-3900
dataset_name=wizard_of_wikipedia
data_path=/mnt/ai2lab/weishao4/programs/InterpretableKGC/data/${dataset_name}/hf_data
output_dir=/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/gen_response/generation_results/${model_name}_${dataset_name}
mkdir -p $output_dir
# --num_train_epochs $num_epochs \
python gen_response_for_knowledge.py \
    --model_name $model_name \
    --model_path $model_path \
    --data_path $data_path \
    --dataset_name $dataset_name \
    --output_dir_pat $output_dir \
    --train_s 60000 \
    --train_e 83247