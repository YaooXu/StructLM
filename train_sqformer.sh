# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1

export PYTHONPATH=/home/yaoxu/StructLM
export WANDB_PROJECT=WTQ

deepspeed_config_file=ds_zero2_no_offload.json
# deepspeed_config_file=ds_zero3.json
max_desc_length=1792
max_seq_length=2048
num_train_epochs=5
lr=1e-5
wd=0.05
eval_steps=500
master_port=29500
strategy=v2.5
num_query_tokens=16
cross_attention_freq=1


dataset_dir=data/WTQ

wandb online

qformer_ckpt_path=StructLM_cwq_webqsp/v2.5_1792_2048_10_1_0.05_1e-5/checkpoint-10000/Qformer.bin

            # --qformer_ckpt_path=${qformer_ckpt_path} \

            # --model_name_or_path=/home/yaoxu/StructLM/models/ckpts/StructLM-7B \
for strategy in v2.5; do
    for lr in 1e-5; do
        deepspeed --include localhost:0,1,2,3 --master_port=${master_port} StructQformer/train_sqformer.py \
            --do_train \
            --gradient_checkpointing \
            --overwrite_output_dir \
            --deepspeed=${deepspeed_config_file} \
            --do_eval \
            --do_predict \
            --bf16 \
            --strategy=${strategy} \
            --num_query_tokens=${num_query_tokens} \
            --max_desc_length=${max_desc_length} \
            --max_seq_length=${max_seq_length} \
            --cross_attention_freq=${cross_attention_freq} \
            --dataset_dir=${dataset_dir} \
            --output_dir=sqformer_outputs/${dataset_dir}/${strategy}_nf_${max_desc_length}_${max_seq_length}_${num_query_tokens}_${cross_attention_freq}_${wd}_${lr} \
            --seed=0 \
            --num_train_epochs=${num_train_epochs} \
            --per_device_train_batch_size=2 \
            --per_device_eval_batch_size=4 \
            --gradient_accumulation_steps=1 \
            --save_strategy=steps \
            --evaluation_strategy=steps \
            --eval_steps=${eval_steps} \
            --save_steps=${eval_steps} \
            --save_total_limit=5 \
            --learning_rate=${lr} \
            --weight_decay=${wd} \
            --warmup_ratio=0.03 \
            --lr_scheduler_type=cosine \
            --logging_steps=10 \
            --report_to wandb
    done
done