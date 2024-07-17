# chmod 777 /mnt/publiccache/yaoxu
# chmod 777 /mnt/userdata
# export HF_HOME=/mnt/publiccache/huggingface
# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# cd /mnt/publiccache/yaoxu/StructLM/

export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# deepspeed_config_file=ds_zero2_offload.json
deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=10
lr=2e-5
wd=0.05
eval_steps=2000
master_port=29508
strategy=v2.6
num_query_tokens=10
cross_attention_freq=1
finetuning_type=freeze_backbone

wandb online

# qformer_ckpt_path=StructLM_cwq_webqsp/v2.5_1792_2048_10_1_0.05_1e-5/checkpoint-10000/Qformer.bin
        # --qformer_ckpt_path=${qformer_ckpt_path} \

        # --skip_encoder \

# meta-llama/Meta-Llama-3-8B
# model_name_or_path=codellama/CodeLlama-7b-Instruct-hf
    # --gradient_checkpointing \

dataset_dir=data/uniskg/wikitq
roberta_size=large
ft_type=lora

t5_ft_type=fre_enc-full_dec
t5_model_type=scrach
t5_size=large

llm=llama

gas=1

for t5_size in base ; do

    for t5_ft_type in freeze_enc_full_dec ; do

        for t5_model_type in scrach ; do

        # cfg=v3.2-roberta_${roberta_size}-${t5_ft_type}_T5_${t5_size}-${ft_type}_${llm}-10.cfg
        # cfg=t5_qformer/v3.3-${t5_ft_type}_trained_T5_${t5_size}-lora_${llm}-10.cfg
        cfg=uniskg/v3-lora_all_trained_T5_large-lora_llama-10.cfg

        echo ${cfg}

        export WANDB_PROJECT=$(basename "$dataset_dir")

            # --gradient_checkpointing \
        deepspeed --master_port=${master_port} StructQformer/train_sqformer.py \
            --do_train \
            --bf16 \
            --deepspeed=${deepspeed_config_file} \
            --cfg=${cfg} \
            --load_best_model_at_end=True \
            --do_eval \
            --max_desc_length=${max_desc_length} \
            --max_seq_length=${max_seq_length} \
            --dataset_dir=${dataset_dir} \
            --overwrite_output_dir \
            --output_dir=/mnt/userdata/StructLM/outputs/${dataset_dir}/${cfg} \
            --seed=0 \
            --num_train_epochs=${num_train_epochs} \
            --per_device_train_batch_size=2 \
            --gradient_accumulation_steps=${gas} \
            --per_device_eval_batch_size=8 \
            --save_strategy=epoch \
            --evaluation_strategy=epoch \
            --save_total_limit=1 \
            --learning_rate=${lr} \
            --weight_decay=${wd} \
            --warmup_ratio=0.05 \
            --lr_scheduler_type=cosine \
            --logging_steps=50 \
            --report_to wandb
        done
    done
done