# chmod 777 /mnt/publiccache/yaoxu
# chmod 777 /mnt/userdata
# cd /mnt/publiccache/yaoxu/StructLM/

# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export HF_HOME=/cpfs/29f69eb5e2e60f26/user/GPT/pretrain/zengxiangrong2/intern/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0

# deepspeed_config_file=ds_zero2_offload.json
deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=3
lr=2e-5
wd=0.05
master_port=29503
strategy=v2.6
num_query_tokens=10
cross_attention_freq=1
finetuning_type=freeze_backbone

wandb online

dataset_dir=data/hytrel/wikitq

llm=llama

gas=8

for cfg in hytrel/v2-predict.cfg ; do

    echo ${cfg}

    export WANDB_PROJECT=$(basename "$dataset_dir")

        # --gradient_checkpointing \
    deepspeed --master_port=${master_port} StructQformer/train_sqformer.py \
        --do_predict \
        --bf16 \
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
        --per_device_eval_batch_size=16 \
        --save_strategy=epoch \
        --evaluation_strategy=epoch \
        --save_total_limit=3 \
        --learning_rate=${lr} \
        --weight_decay=${wd} \
        --warmup_ratio=0.05 \
        --lr_scheduler_type=cosine \
        --logging_steps=2 \
        --report_to wandb

done