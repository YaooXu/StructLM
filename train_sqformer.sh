# chmod 777 /mnt/publiccache/yaoxu
# chmod 777 /mnt/userdata
# export HF_HOME=/mnt/publiccache/huggingface
# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# cd /mnt/publiccache/yaoxu/StructLM/

export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=WTQ

# deepspeed_config_file=ds_zero2_offload.json
deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=3
lr=2e-5
wd=0.05
eval_steps=2000
master_port=29504
strategy=v2.6
num_query_tokens=10
cross_attention_freq=1
finetuning_type=freeze_backbone

dataset_dir=data/8Tab_Tasks

wandb online

# qformer_ckpt_path=StructLM_cwq_webqsp/v2.5_1792_2048_10_1_0.05_1e-5/checkpoint-10000/Qformer.bin
        # --qformer_ckpt_path=${qformer_ckpt_path} \

        # --skip_graph_encoder \

model_name_or_path=TIGER-Lab/StructLM-7B-Mistral
# model_name_or_path=codellama/CodeLlama-7b-Instruct-hf
    # --gradient_checkpointing \

for model_name_or_path in meta-llama/Llama-2-7b-hf ; do 

    for finetuning_type in lora ; do
    
        for num_query_tokens in 10 ; do

            strategy=v2.6

            if [[ "$finetuning_type" == "freeze_backbone" && "$num_query_tokens" -eq 0 ]]; then
                strategy="pt"
                num_query_tokens=10
            fi

            model_name=$(basename "$model_name_or_path")

                # --gradient_checkpointing \
            deepspeed --master_port=${master_port} StructQformer/train_sqformer.py \
                --model_name_or_path=${model_name_or_path} \
                --do_train \
                --load_best_model_at_end=False \
                --finetuning_type=${finetuning_type} \
                --overwrite_output_dir \
                --deepspeed=${deepspeed_config_file} \
                --do_eval \
                --bf16 \
                --strategy=${strategy} \
                --num_query_tokens=${num_query_tokens} \
                --max_desc_length=${max_desc_length} \
                --max_seq_length=${max_seq_length} \
                --cross_attention_freq=${cross_attention_freq} \
                --dataset_dir=${dataset_dir} \
                --output_dir=/mnt/userdata/StructLM/outputs/${dataset_dir}/inter_${model_name}_${finetuning_type}_${strategy}_${max_desc_length}_${max_seq_length}_${num_query_tokens}_${cross_attention_freq}_${wd}_${lr} \
                --seed=0 \
                --num_train_epochs=${num_train_epochs} \
                --per_device_train_batch_size=2 \
                --per_device_eval_batch_size=4 \
                --gradient_accumulation_steps=2 \
                --save_strategy=epoch \
                --evaluation_strategy=steps \
                --eval_steps=0.2 \
                --save_total_limit=1 \
                --learning_rate=${lr} \
                --weight_decay=${wd} \
                --warmup_ratio=0.03 \
                --lr_scheduler_type=cosine \
                --logging_steps=50 \
                --report_to wandb
        done
    done
done