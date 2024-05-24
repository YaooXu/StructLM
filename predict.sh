# chmod 777 /mnt/publiccache/yaoxu
# chmod 777 /mnt/userdata
# cd /mnt/publiccache/yaoxu/StructLM/
# export HF_HOME=/mnt/publiccache/huggingface

export HF_HOME=/mnt/publiccache/huggingface
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=WTQ

# deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=3
lr=2e-5
wd=0.05
eval_steps=1000
master_port=29500
strategy=v2.6
num_query_tokens=10
cross_attention_freq=1
finetuning_type=freeze_backbone

dataset_dir=data/WTQ_Inter_new
# dataset_dir=data/5Task_Mistral

wandb offline

ckpt_path=/mnt/userdata/StructLM/outputs/data/WTQ_Inter_new/3_ln_StructLM-7B-Mistral_freeze_backbone_v2.6_2048_2560_10_1_0.05_2e-5/checkpoint-2831

model_name_or_path=TIGER-Lab/StructLM-7B-Mistral
# model_name_or_path=codellama/CodeLlama-7b-Instruct-hf
        # --gradient_checkpointing \
batch_size=2

model_name=$(basename "$model_name_or_path")

        # --deepspeed=${deepspeed_config_file} \
        # --ckpt_path=${ckpt_path} \
deepspeed --include localhost:1,2,3,4 --master_port=${master_port} StructQformer/train_sqformer.py \
        --model_name_or_path=${model_name_or_path} \
        --do_predict \
        --do_eval \
        --bf16 \
        --strategy=${strategy} \
        --num_query_tokens=${num_query_tokens} \
        --max_desc_length=${max_desc_length} \
        --max_seq_length=${max_seq_length} \
        --cross_attention_freq=${cross_attention_freq} \
        --dataset_dir=${dataset_dir} \
        --output_dir=./tmp_pred/ \
        --seed=0 \
        --num_train_epochs=${num_train_epochs} \
        --per_device_train_batch_size=${batch_size} \
        --per_device_eval_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --save_strategy=steps \
        --evaluation_strategy=steps \
        --eval_steps=${eval_steps} \
        --save_steps=${eval_steps} \
        --save_total_limit=1 \
        --learning_rate=${lr} \
        --weight_decay=${wd} \
        --warmup_ratio=0.03 \
        --lr_scheduler_type=cosine \
        --logging_steps=50 \
        --report_to wandb