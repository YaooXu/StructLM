# chmod 777 /mnt/publiccache/yaoxu
# chmod 777 /mnt/userdata
# export HF_HOME=/mnt/publiccache/huggingface
# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# cd /mnt/publiccache/yaoxu/StructLM/

export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

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

dataset_dir=data/hytrel/wikitq

llm=llama

gas=1

for cfg in hytrel-llama3/v2-lora_llama-0.cfg hytrel-llama3/v2-roberta_base-lora_llama-10.cfg ; do

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
        --output_dir=./outputs/${dataset_dir}/${cfg} \
        --seed=0 \
        --num_train_epochs=${num_train_epochs} \
        --per_device_train_batch_size=4 \
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