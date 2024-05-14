# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1

export HF_HOME=/mnt/publiccache/huggingface
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=WTQ

# deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=5
lr=1e-5
wd=0.05
eval_steps=500
master_port=29500
num_query_tokens=0
cross_attention_freq=1

dataset_dir=data/WTQ_Mistral

wandb offline

# qformer_ckpt_path=StructLM_cwq_webqsp/v2.5_1792_2048_10_1_0.05_1e-5/checkpoint-10000/Qformer.bin
            # --qformer_ckpt_path=${qformer_ckpt_path} \

            # --skip_graph_encoder \

model_name_or_path=TIGER-Lab/StructLM-7B-Mistral
model_name=$(basename "$model_name_or_path")

for strategy in pt; do
    for lr in 1e-5; do
        deepspeed --include localhost:0,1,2,3 --master_port=${master_port} StructQformer/train_sqformer.py \
            --model_name_or_path=${model_name_or_path} \
            --do_predict \
            --bf16 \
            --strategy=${strategy} \
            --num_query_tokens=${num_query_tokens} \
            --max_desc_length=${max_desc_length} \
            --max_seq_length=${max_seq_length} \
            --cross_attention_freq=${cross_attention_freq} \
            --dataset_dir=${dataset_dir} \
            --output_dir=sqformer_outputs/${dataset_dir}/${model_name}_${strategy}_${max_desc_length}_${max_seq_length}_${num_query_tokens}_${cross_attention_freq}_${wd}_${lr} \
            --seed=0 \
            --num_train_epochs=${num_train_epochs} \
            --per_device_train_batch_size=1 \
            --per_device_eval_batch_size=2 \
            --gradient_accumulation_steps=2 \
            --save_strategy=steps \
            --evaluation_strategy=steps \
            --eval_steps=${eval_steps} \
            --save_steps=${eval_steps} \
            --save_total_limit=5 \
            --learning_rate=${lr} \
            --weight_decay=${wd} \
            --warmup_ratio=0.03 \
            --lr_scheduler_type=cosine \
            --logging_steps=50 \
            --report_to wandb
    done
done