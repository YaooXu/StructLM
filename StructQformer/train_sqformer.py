import sys

sys.path.append("./")

import random
from utils.configure import Configure
import os
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PredictionOutput
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from StructQformer.models import StructQformerLLM
from StructQformer.SQformer_dataset_tabert import (
    DEFAULT_GRAPH_PAD_TOKEN,
    DataCollatorForGenerating,
    DataCollatorForGraphSupervisedDataset,
    build_instruction_dataset,
)
import transformers
import torch
from typing import TYPE_CHECKING, Optional, Union
import pathlib
import logging
import json
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer as Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules import Module
from SQformerTrainer import (
    StructQASeq2SeqTrainer,
    PredictionProgressCallback,
    post_process_function,
)
import wandb
import numpy as np
from collections import OrderedDict
from utils.utils import load_jsonl


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    # model_name_or_path: Optional[str] = field(default="/home/yaoxu/StructLM/models/ckpts/StructLM-7B")
    encoder_model_path: Optional[str] = field(default="google-bert/bert-base-uncased")
    num_query_tokens: int = field(default=10)
    cross_attention_freq: int = field(default=1)

    encoder_finetuning_type: str = field(default='freeze_plm')

    finetuning_type: str = field(default='freeze_backbone')
    target_modules: str = field(default='q_proj,v_proj')

    strategy: str = field(default="pt")

    skip_encoder: bool = field(default=False)

    ckpt_path: str = field(default=None)

    attn_implementation: str = field(default='flash_attention_2')


@dataclass
class WarppedTrainingArguments(TrainingArguments):

    # data args
    dataset_dir: str = field(default="dataset/webqsp/processed_files")
    max_desc_length: int = field(default=2048)
    max_seq_length: int = field(default=2560)
    preprocessing_num_workers: int = field(default=8)
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The datasets processed stored"}
    )

    cfg: str = field(default="qformer/v3.cfg")

    output_dir: str = field(default="trainer_outputs")

    # to avoid Warning
    optim: str = field(default="adamw_torch")

    flash_attn: Optional[bool] = field(default=False)

    remove_unused_columns: int = field(default=False)

    generation_config: str = field(default="generation_config.json")

    predict_with_generate: bool = field(default=True)

    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="loss")
    greater_is_better: bool = field(default=False)

    disable_tqdm: bool = False


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((WarppedTrainingArguments))
    training_args, = parser.parse_args_into_dataclasses()

    model_args = Configure.Get(training_args.cfg)

    training_args.run_name = training_args.output_dir.split("/")[-1]

    set_seed(training_args.seed)
    torch_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    llm_tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm.model_name_or_path, use_fast=False)

    encoder_tokenizer = AutoTokenizer.from_pretrained(
        model_args.encoder.model_name_or_path, use_fast=False)

    model = StructQformerLLM(model_args,
                             llm_tokenizer,
                             encoder_tokenizer,
                             use_cache=False if training_args.gradient_checkpointing else True,
                             torch_dtype=torch_dtype)

    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if training_args.bf16:
    #             module = module.to(torch.bfloat16)
    #         if training_args.fp16:
    #             module = module.to(torch.float16)
    #     if "norm" in name:
    #         module = module.to(torch.float16)
    #     if "lm_head" in name or "embed_tokens" in name:
    #         if hasattr(module, "weight"):
    #             if training_args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    #             if training_args.fp16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.float16)

    if training_args.should_log:
        model.print_trainable_params()

    dataset_dir = pathlib.Path(training_args.dataset_dir)

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = build_instruction_dataset(
            dataset_dir / f"train.pq",
            llm_tokenizer,
            encoder_tokenizer,
            max_seq_length=training_args.max_seq_length,
            max_desc_length=training_args.max_desc_length,
            num_query_tokens=model_args.qformer.num_query_tokens,
        )
        eval_dataset = build_instruction_dataset(
            dataset_dir / f"val.pq",
            llm_tokenizer,
            encoder_tokenizer,
            max_seq_length=training_args.max_seq_length,
            max_desc_length=training_args.max_desc_length,
            num_query_tokens=model_args.qformer.num_query_tokens,
        )
        eval_dataset = eval_dataset.select(random.sample(
            range(len(eval_dataset)), k=min(10000, len(eval_dataset))))

    if training_args.do_train or training_args.do_predict:
        test_dataset = build_instruction_dataset(
            dataset_dir / f"test.pq",
            llm_tokenizer,
            encoder_tokenizer,
            max_seq_length=training_args.max_seq_length,
            max_desc_length=training_args.max_desc_length,
            num_query_tokens=model_args.qformer.num_query_tokens,
            training=False,
        )
        test_examples = load_jsonl(dataset_dir / f"ori_test.jsonl")

        # test_dataset = test_dataset.select(range(10))

    data_collator = DataCollatorForGraphSupervisedDataset(llm_tokenizer, encoder_tokenizer)

    trainer = StructQASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        data_collator=data_collator,
        post_process_function=post_process_function,
        # compute_metrics=compute_metrics,
    )

    callback = PredictionProgressCallback(
        trainer, llm_tokenizer, encoder_tokenizer, test_dataset, test_examples)
    trainer.add_callback(callback)

    if training_args.do_train:
        if "debug" not in training_args.output_dir and list(
            pathlib.Path(training_args.output_dir).glob("checkpoint-*")
        ):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

    elif training_args.do_predict:
        trainer.data_collator = DataCollatorForGenerating(llm_tokenizer, encoder_tokenizer)
        logger.info("*** Predict ***")
        metrics = trainer.predict(test_dataset, test_examples)
