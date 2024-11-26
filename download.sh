# # download the model
# huggingface-cli download --repo-type=model --local-dir=models/ckpts/StructLM-7B TIGER-Lab/StructLM-7B

# # download the data required for executing evaluation
# huggingface-cli download --repo-type=dataset --local-dir=data/downloads/extracted/ TIGER-Lab/SKGInstruct ./skg_raw_data.zip
# # unzip it in that folder
# unzip -o data/downloads/extracted/skg_raw_data.zip -d data/downloads/extracted/

# # download the test data
# # NOTE: the 7b and 13/34b has a slightly different eval format due to a slight training bug that does not affect performance
# huggingface-cli download --repo-type=dataset --local-dir=data/processed/ TIGER-Lab/SKGInstruct ./skginstruct.json 

# export HF_HOME=/mnt/publiccache/huggingface

# huggingface-cli download --repo-type=dataset --local-dir=data/processed/ TIGER-Lab/SKGInstruct ./skginstruct_test_file_7b.json 

huggingface-cli download --repo-type=model meta-llama/Llama-2-7b
