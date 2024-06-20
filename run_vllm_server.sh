pip install -U openai

export HF_HOME=/mnt/publiccache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# model_name=meta-llama/Meta-Llama-3-70B-Instruct
model_name=Qwen/Qwen1.5-72B-Chat

HOST=0.0.0.0
PORT=12240

python -m vllm.entrypoints.openai.api_server \
    --model  ${model_name} \
    --dtype bfloat16 \
    --host ${HOST} \
    --tensor-parallel-size 4 \
    --port ${PORT}