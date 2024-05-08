export RAY_TMPDIR=/home/yaoxu/tmp/

python -m vllm.entrypoints.api_server \
    --model ./models/ckpts/StructLM-7B \
    --tensor-parallel-size 4