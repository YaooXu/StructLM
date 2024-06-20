from collections import OrderedDict
import logging
from math import sqrt
import os
from typing import Optional
import torch
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from StructQformer.models.base_models.modeling_roberta import RobertaEncoder, RobertaModel
from StructQformer.models.graphormer.graphormer import Graphormer

from models.hytrel import HyperGraphEncoder

# from model.graph_encoder_geo import GraphEncoder

from .Qformer import BertConfig, BertLMHeadModel, BertModel

import torch.nn as nn

from SQformer_dataset import DEFAULT_CVT_TOKEN, DEFAULT_GRAPH_PAD_TOKEN
from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)


class QueryEmbedsGenerator(nn.Module):
    def __init__(self, num_query_tokens, llm_hidden_size=4096) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.llm_hidden_size = llm_hidden_size
        self.query_token_embeds = nn.Parameter(
            torch.zeros(self.num_query_tokens, llm_hidden_size)
        )
        self.query_token_embeds.data.normal_(mean=0.0, std=0.02)
        self.ln_norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, graph, llm, llm_graph_pad_token_id):
        inputs_embeds = llm.get_input_embeddings()(graph['question_input_ids'])
        batch_size = inputs_embeds.shape[0]

        graph_pad_st_idx = torch.argmax(
            (graph['question_input_ids'] == llm_graph_pad_token_id).int(), dim=1)
        graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

        query_token_embeds = self.query_token_embeds
        # query_token_embeds = self.ln_norm(query_token_embeds)

        new_inputs_embeds = torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        for i in range(batch_size):
            cur_inputs_embeds = inputs_embeds[i]
            cur_graph_pad_st_idx, cur_graph_pad_ed_idx = graph_pad_st_idx[i], graph_pad_ed_idx[i]

            new_inputs_embeds[i][:cur_graph_pad_st_idx] += cur_inputs_embeds[:cur_graph_pad_st_idx]
            new_inputs_embeds[i][cur_graph_pad_st_idx: cur_graph_pad_ed_idx] += query_token_embeds

        # LLM generates query tokens
        output = llm(inputs_embeds=new_inputs_embeds,
                     attention_mask=graph['question_attention_mask'], output_hidden_states=True)
        last_hidden_state = output.hidden_states[-1]

        query_embeds = torch.zeros(
            (batch_size, self.num_query_tokens, self.llm_hidden_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        for i in range(batch_size):
            query_embeds[i] += last_hidden_state[i][graph_pad_st_idx[i]: graph_pad_ed_idx[i]]

        return query_embeds


class StructQformer(nn.Module):
    def __init__(self, args, hypergraph_enc_config) -> None:
        super().__init__()

        self.args = args

        self.cross_attention_freq = args.cross_attention_freq
        self.num_query_tokens = args.num_query_tokens
        self.encoder_model_path = args.encoder_model_path
        self.strategy = args.strategy

        self.encoder_config = AutoConfig.from_pretrained(self.encoder_model_path)
        self.encoder_config.encoder_width = self.encoder_config.hidden_size
        # insert cross-attention layer every other block
        self.encoder_config.add_cross_attention = self.cross_attention_freq > 0
        self.encoder_config.cross_attention_freq = self.cross_attention_freq
        self.encoder_config.query_length = self.num_query_tokens
        
        if self.strategy[:2] == "v2":
            # self.hypergraph_encoder = HyperGraphEncoder(hypergraph_enc_config)
            self.graph_encoder = Graphormer(args.encoder_model_path)
                
            # for roberta
            self.encoder_config.add_cross_attention = True
            self.encoder_config.is_decoder = True
            self.encoder_config.use_dist_bias = False
            self.model = RobertaModel.from_pretrained(
                self.encoder_model_path, config=self.encoder_config,
            )
            # self.model = self.graph_encoder.model
            
            self.query_token_embeds = nn.Parameter(
                torch.zeros(self.num_query_tokens, self.encoder_config.hidden_size)
            )
            self.query_embeds_generator = QueryEmbedsGenerator(self.num_query_tokens)
        else:
            self.graph_encoder = None
            self.model = None
            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, 4096))

        self.projector1 = nn.Linear(4096, self.encoder_config.hidden_size)
        self.projector2 = nn.Linear(self.encoder_config.hidden_size, 4096)

        self.ln_norm1 = nn.LayerNorm(self.encoder_config.hidden_size, self.encoder_config.layer_norm_eps)
        self.ln_norm2 = nn.LayerNorm(4096, self.encoder_config.layer_norm_eps)

        self.init_weight()

    def init_weight(self):
        self.query_token_embeds.data.normal_(mean=0.0, std=self.encoder_config.initializer_range)
        for proj in [self.projector1, self.projector2]:
            proj.weight.data.normal_(
                mean=0.0, std=self.encoder_config.initializer_range)
            if proj.bias is not None:
                proj.bias.data.zero_()
                
        for ln in [self.ln_norm1, self.ln_norm2]:
            ln.bias.data.zero_()
            ln.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens):
        if self.graph_encoder:
            self.graph_encoder.model.resize_token_embeddings(new_num_tokens)
        # if self.model is not None:
        #     self.model.resize_token_embeddings(new_num_tokens)

    @property
    def base_model(self):
        return self.model

    # as this value is added after initializing the model
    @property
    def bert_graph_pad_token(self):
        return self.args.bert_graph_pad_token

    @property
    def device(self):
        return self.model.device

    def gen_query_embeds_pt(self, graph, llm, llm_graph_pad_token_id):
        # vanilla prompt tuning
        batch_size = graph["question_input_ids"].shape[0]
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return query_embeds

    def gen_query_embeds_v2(self, graph, llm, llm_graph_pad_token_id):
        question_ids = graph["question_input_ids"]
        batch_size = question_ids.shape[0]

        if self.args.skip_graph_encoder:
            res_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            graph_output = self.graph_encoder(graph['graph'])
            graph_attention_mask = graph['graph']["graph_attention_mask"]

            query_embeds = self.query_embeds_generator(graph, llm, llm_graph_pad_token_id)

            query_embeds = self.projector1(query_embeds)
            # query_embeds = self.ln_norm1(query_embeds)
            
            query_atts = torch.ones(query_embeds.shape[:-1]).to(self.device)
            
            question_output = self.model(
                inputs_embeds=query_embeds,
                attention_mask=query_atts,
                encoder_hidden_states=graph_output.last_hidden_state,
                encoder_attention_mask=graph_attention_mask,
                # query_length=self.num_query_tokens,
                # use_cache=False,
            )
            res_embeds = question_output.last_hidden_state[:, -self.num_query_tokens:, :]

        # print(res_embeds.norm(dim=-1))
        res_embeds = self.projector2(res_embeds)
        # res_embeds = self.ln_norm2(res_embeds)
        # print(res_embeds.norm(dim=-1))
        
        return res_embeds

    def forward(self, graph, llm, llm_graph_pad_token_id):
        if self.strategy == "pt":
            query_embeds = self.gen_query_embeds_pt(graph, llm, llm_graph_pad_token_id)
        elif self.strategy[:2] == "v2":
            query_embeds = self.gen_query_embeds_v2(graph, llm, llm_graph_pad_token_id)
        else:
            raise NotImplementedError

        return query_embeds


class StructQformerLLM(nn.Module):
    def __init__(self, args, hypergraph_enc_config, llm_tokenizer, bert_tokenizer, **kwargs) -> None:
        super().__init__()

        self.num_query_tokens = args.num_query_tokens

        # set in init_tokenizer_and_embeds
        self.bert_graph_pad_token = None
        self.llm_graph_pad_token_id = None
        self.llm_pad_token_id = None
        self.finetuning_type = args.finetuning_type
        self.args = args

        if self.num_query_tokens > 0:
            self.qformer = StructQformer(args, hypergraph_enc_config)

            bert_tokenizer.add_tokens(["[TAB]", "[HEAD]", "[CELL]", "[ROW]",
                                       "scinotexp"], special_tokens=True)
            self.bert_graph_pad_token = bert_tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PAD_TOKEN])[
                0
            ]
            self.qformer.resize_token_embeddings(len(bert_tokenizer))

            # self.qformer.load_state_dict(torch.load(os.path.join('/mnt/userdata/StructLM/outputs/data/WTQ_ori_input_no_inter/3e_bi_Llama-2-7b-hf_freeze_backbone_v2.6_2048_2560_10_1_0.05_2e-5/checkpoint-8493', "Qformer.bin")))
                        
            if args.ckpt_path is not None and not args.skip_graph_encoder:
                logger.info(f"loading qformer ckpt from {args.ckpt_path}")
                self.qformer.load_state_dict(torch.load(os.path.join(args.ckpt_path, "Qformer.bin")))
            
            self.qformer = self.qformer.to(kwargs['torch_dtype'])
        else:
            self.qformer = None

        self.llm: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            **kwargs
        )
        self.init_tokenizer_and_embeds(llm_tokenizer, bert_tokenizer, DEFAULT_GRAPH_PAD_TOKEN)

        if args.finetuning_type == 'full':
            for name, param in self.llm.named_parameters():
                param.requires_grad = True
        elif args.finetuning_type == 'lora':
            if args.ckpt_path is not None:
                logger.info(f'loading lora ckpt from {args.ckpt_path}')
                self.llm.load_adapter(args.ckpt_path)
            else:
                logger.info('adding lora model')
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=args.target_modules.split(','),
                    r=32,
                    lora_alpha=64,
                    lora_dropout=0.1,
                )
                self.llm = get_peft_model(self.llm, peft_config)
                self.llm.print_trainable_parameters()
        elif args.finetuning_type == 'freeze_backbone':
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError

    @property
    def config(self):
        return self.llm.config

    @property
    def generation_config(self):
        return self.llm.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.llm.generation_config = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    def init_tokenizer_and_embeds(
        self,
        llm_tokenizer: AutoTokenizer,
        bert_tokenizer,
        graph_pad_token=DEFAULT_GRAPH_PAD_TOKEN,
    ):
        llm = self.llm

        llm_tokenizer.add_tokens([graph_pad_token], special_tokens=True)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        self.llm_graph_pad_token_id = llm_tokenizer.convert_tokens_to_ids(
            [DEFAULT_GRAPH_PAD_TOKEN]
        )[0]
        self.llm_pad_token_id = llm_tokenizer.pad_token_id
        llm.resize_token_embeddings(len(llm_tokenizer))

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for name, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                # print(name, param.shape, num_params)
                trainable_params += num_params

        print(f"{trainable_params} / {all_param}, {trainable_params*100/all_param}%")
        return trainable_params, all_param

    def construct_inputs_embeds(self, input_ids, graph):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.shape[0]

        if self.num_query_tokens > 0:
            res_embeds = self.qformer(
                graph, self.llm, self.llm_graph_pad_token_id).to(inputs_embeds.dtype)

            graph_pad_st_idx = torch.argmax((input_ids == self.llm_graph_pad_token_id).int(), dim=1)
            graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

            new_inputs_embeds = torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            for i in range(batch_size):
                cur_inputs_embeds = inputs_embeds[i]
                cur_graph_pad_st_idx, cur_graph_pad_ed_idx = graph_pad_st_idx[i], graph_pad_ed_idx[i]

                new_inputs_embeds[i][:cur_graph_pad_st_idx] += cur_inputs_embeds[:cur_graph_pad_st_idx]
                new_inputs_embeds[i][cur_graph_pad_st_idx: cur_graph_pad_ed_idx] += res_embeds[i]
                new_inputs_embeds[i][cur_graph_pad_ed_idx:] += cur_inputs_embeds[cur_graph_pad_ed_idx:]
        else:
            new_inputs_embeds = inputs_embeds

        return new_inputs_embeds

    def forward(
        self,
        graph,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: torch.List[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        inputs_embeds = self.construct_inputs_embeds(input_ids, graph)

        outputs = self.llm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def generate(
        self,
        graph,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: torch.List[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **gen_kwargs,
    ):
        inputs_embeds = self.construct_inputs_embeds(input_ids, graph)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **gen_kwargs,
        )

        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        Args:
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        re-loaded using the `LoraModel.from_pretrained` class method, and also used by the `LoraModel.push_to_hub`
        method.
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            **kwargs:
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        state_dict = kwargs["state_dict"]

        output_state_dict = {k: state_dict[k] for k in state_dict if "qformer" in k}

        torch.save(output_state_dict, os.path.join(save_directory, "qformer.bin"))
