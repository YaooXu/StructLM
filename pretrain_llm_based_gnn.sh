# chmod 777 /mnt/publiccache/yaoxu
# chmod 777 /mnt/userdata
# cd /mnt/publiccache/yaoxu/StructLM/

# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

cd /cpfs/29f69eb5e2e60f26/user/GPT/pretrain/zengxiangrong2/intern/xuyao/StructLM

export HF_HOME=/cpfs/29f69eb5e2e60f26/user/GPT/pretrain/zengxiangrong2/intern/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

deepspeed_config_file=ds_zero2.json
# deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
lr=2e-5
wd=0.05
strategy=v2.6
num_train_epochs=10
cross_attention_freq=1
finetuning_type=freeze_backbone

wandb online


dataset_dir=data/hytrel/llm_based_gnn_pretraining

llm=llama

gas=1


cfg=hytrel-llama/v2-10M_only_query-not_freeze_gnn-trained_roberta_base-freeze_llama-10.cfg
include=localhost:0,1,2,3,4,5,6,7
master_port=29501

echo ${cfg}

export WANDB_PROJECT=$(basename "$dataset_dir")

deepspeed --master_port=${master_port} --include=${include} train_sqformer.py \
    --do_train \
    --do_predict \
    --bf16 \
    --deepspeed=${deepspeed_config_file} \
    --cfg=${cfg} \
    --max_desc_length=${max_desc_length} \
    --max_seq_length=${max_seq_length} \
    --dataset_dir=${dataset_dir} \
    --overwrite_output_dir \
    --output_dir=./outputs/${dataset_dir}/${cfg} \
    --seed=0 \
    --num_train_epochs=${num_train_epochs} \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=${gas} \
    --per_device_eval_batch_size=16 \
    --save_strategy=steps \
    --save_steps=10000 \
    --save_total_limit=1 \
    --learning_rate=${lr} \
    --weight_decay=${wd} \
    --warmup_ratio=0.05 \
    --lr_scheduler_type=cosine \
    --logging_steps=10 \
    --report_to wandb