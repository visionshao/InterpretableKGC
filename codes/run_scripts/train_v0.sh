export CUDA_VISIBLE_DEVICES=0,1
cd /mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/kgc_model

model_name=flan_t5_base
model_path=/mnt/ai2lab/weishao4/programs/InterpretableKGC/pretrained_weights/FLAN-T5-base
dataset_name=wizard_of_wikipedia
data_path=/mnt/ai2lab/weishao4/programs/InterpretableKGC/data/${dataset_name}/hf_data
output_dir=/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/kgc_model/saved_checkpoints/${dataset_name}_${model_name}
lr=1e-4
optim="adamw_torch"
train_bs=64
eval_bs=32
weight_decay=1e-2
save_total_limit=3
# num_epochs=1
max_steps=10000
gen_len=128
pred_w_gen=True
logging_steps=20
eval_steps=100
save_steps=100
evaluation_strategy=steps
gradient_accumulation_steps=2
# metric_for_best_model=bleu
# greater_is_better=True
gradient_checkpointing=True
group_by_length=True
bf16=True
report_to=None

gpu_num=2

# --num_train_epochs $num_epochs \
torchrun --nproc_per_node $gpu_num train.py \
    --model_name $model_name \
    --model_path $model_path \
    --data_path $data_path \
    --dataset_name $dataset_name \
    --output_dir $output_dir \
    --evaluation_strategy $evaluation_strategy \
    --optim $optim \
    --learning_rate $lr \
    --log_level warning \
    --log_level_replica error \
    --per_device_train_batch_size $train_bs \
    --per_device_eval_batch_size $eval_bs \
    --weight_decay $weight_decay \
    --save_total_limit $save_total_limit \
    --max_steps $max_steps \
    --generation_max_length $gen_len \
    --predict_with_generate $pred_w_gen \
    --logging_steps $logging_steps \
    --eval_steps $eval_steps \
    --save_steps $save_steps \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --bf16 $bf16 \
    --group_by_length $group_by_length \
    --gradient_checkpointing $gradient_checkpointing \
    --run_name ${model_name}_${dataset_name}_${lr}_${train_bs}_${gradient_accumulation_steps}_${num_epochs}

# wandb sync /mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/kgc_model/wandb/latest-run

    # --report_to $report_to \