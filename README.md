# LLaSA

This is the repository for the paper "LLaSA: Large Language and Structured Data Assistant". 

You can use this repository to evaluate the models. To reproduce the models, use [SKGInstruct](https://huggingface.co/datasets/TIGER-Lab/SKGInstruct) in your preferred finetuning framework.

The processed test data is already provided, but the prompts used for training and testing can be found in `/prompts`

More details about configuration and training will come soon.

## Install Requirements

Requirements:
- Python 3.10
- Linux
- support for CUDA 12.1

```
pip install -r requirements.txt
```

## Prepare pretraining datasets

```bash
# download pretraining data
git clone https://github.com/YaooXu/TaBERT.git
cd TaBERT
bash get_pretrain_data.sh

python preprocess/construct_pretrain_data.py 
```

## Pretraining

```
bash pretrain_gformer.sh
```


## Trianing

```
bash ./train_llama.sh
```


## Evaluation
```
bash ./predict.sh
```