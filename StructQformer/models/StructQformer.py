from collections import OrderedDict
import logging
from math import sqrt
import os
from pathlib import Path
from typing import Optional
from safetensors import safe_open
import torch
from transformers import (
    LlamaConfig,
    LlamaModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# from StructQformer.models.base_models.modeling_llama import LlamaForCausalLM
from StructQformer.models.base_models.modeling_t5_qformer import T5ForConditionalGeneration
from StructQformer.models.graphormer.graphormer import Graphormer

from models.hytrel import HyperGraphEncoder
from utils.configure import Configure

# from model.encoder_geo import GraphEncoder

from .Qformer_bert import BertConfig, BertLMHeadModel, BertModel
from .Qformer_roberta import  RobertaModel

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
        self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, llm_hidden_size))
        self.query_token_embeds.data.normal_(mean=0.0, std=0.02)
        self.ln_norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, graph, llm, llm_graph_pad_token_id):
        inputs_embeds = llm.get_input_embeddings()(graph["question_input_ids"])
        batch_size = inputs_embeds.shape[0]

        graph_pad_st_idx = torch.argmax((graph["question_input_ids"] == llm_graph_pad_token_id).int(), dim=1)
        graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

        query_token_embeds = self.query_token_embeds
        # query_token_embeds = self.ln_norm(query_token_embeds)

        new_inputs_embeds = torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        for i in range(batch_size):
            cur_inputs_embeds = inputs_embeds[i]
            cur_graph_pad_st_idx, cur_graph_pad_ed_idx = graph_pad_st_idx[i], graph_pad_ed_idx[i]

            new_inputs_embeds[i][:cur_graph_pad_st_idx] += cur_inputs_embeds[:cur_graph_pad_st_idx]
            new_inputs_embeds[i][cur_graph_pad_st_idx:cur_graph_pad_ed_idx] += query_token_embeds
            new_inputs_embeds[i][cur_graph_pad_ed_idx:] += cur_inputs_embeds[cur_graph_pad_ed_idx:]

        # LLM generates query tokens
        output = llm(
            inputs_embeds=new_inputs_embeds, attention_mask=graph["question_attention_mask"], output_hidden_states=True
        )
        last_hidden_state = output.hidden_states[-1]

        query_embeds = torch.zeros(
            (batch_size, self.num_query_tokens, self.llm_hidden_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        for i in range(batch_size):
            query_embeds[i] += last_hidden_state[i][graph_pad_st_idx[i] : graph_pad_ed_idx[i]]

        return query_embeds


class StructQformer(nn.Module):
    def __init__(self, args, hypergraph_enc_config) -> None:
        super().__init__()

        self.args = args
        self.num_query_tokens = args.qformer.num_query_tokens
        self.strategy = args.qformer.strategy

        if "inter" in self.strategy:
            self.query_embeds_generator = QueryEmbedsGenerator(self.num_query_tokens)

        if self.strategy[:2] == "v2":
            self.hypergraph_encoder = HyperGraphEncoder(hypergraph_enc_config)
            self.encoder = self.hypergraph_encoder
            if self.args.qformer.freeze_encoder:
                self.hypergraph_encoder.requires_grad_(False)
            
            self.encoder_config = AutoConfig.from_pretrained(args.qformer.model_name_or_path)
            self.encoder_config.encoder_width = self.encoder_config.hidden_size
            self.encoder_config.add_cross_attention = True
            self.encoder_config.cross_attention_freq = 1
            self.encoder_config.query_length = self.num_query_tokens

            # self.model = BertLMHeadModel.from_pretrained(args.qformer.model_name_or_path, config=self.encoder_config)
            # for roberta
            self.encoder_config.add_cross_attention = True
            self.encoder_config.is_decoder = True
            self.model = RobertaModel.from_pretrained(
                args.qformer.model_name_or_path, config=self.encoder_config,
            )
            if self.args.qformer.model_finetuning_type == 'full':
                self.model.requires_grad_(True)
            elif self.args.qformer.model_finetuning_type == 'lora':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=args.qformer.target_modules.split(","),
                    r=args.qformer.r,
                    lora_alpha=args.qformer.lora_alpha,
                    lora_dropout=args.qformer.lora_dropout,
                )
                self.model = get_peft_model(self.model, peft_config)                
            
            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, self.encoder_config.hidden_size))
            self.projector1 = nn.Linear(self.hypergraph_encoder.config.hidden_size, self.encoder_config.hidden_size)
            self.projector2 = nn.Linear(self.encoder_config.hidden_size, 4096)

            # self.LayerNorm = nn.LayerNorm(
            #     self.encoder_config.hidden_size, eps=self.encoder_config.layer_norm_eps
            # )
        elif self.strategy[:2] == "v3":
            # if 'prefix' in args.encoder.cfg:··
            #     from UnifiedSKG.models.unified.prefixtuning import Model
            # elif 'finetune' in args.encoder.cfg:
            #     from UnifiedSKG.models.unified.finetune import Model

            t5_config = AutoConfig.from_pretrained(args.encoder.model_name_or_path)
            t5_config.num_query_tokens = 10
            self.t5 = T5ForConditionalGeneration.from_pretrained(args.encoder.model_name_or_path, config=t5_config)
            self.encoder = self.t5.encoder
            self.decoder = self.t5.decoder

            if args.encoder.finetuning_type == "full":
                self.t5.requires_grad_(True)
            elif args.encoder.finetuning_type == "lora":
                logger.info("adding lora model")
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    target_modules=args.encoder.target_modules.split(","),
                    r=args.encoder.r,
                    lora_alpha=args.encoder.lora_alpha,
                    lora_dropout=args.encoder.lora_dropout,
                )
                self.t5 = get_peft_model(self.t5, peft_config)
            elif args.encoder.finetuning_type == "freeze_enc_full_dec":
                self.encoder.requires_grad_(False)
                self.decoder.requires_grad_(True)
            else:
                raise NotImplementedError

            self.t5.query_tokens_embeds.requires_grad_(True)

            # # for qformer
            # self.qformer_config = AutoConfig.from_pretrained(args.qformer.model_name_or_path)
            # self.qformer_config.add_cross_attention = True
            # self.qformer_config.is_decoder = True
            # self.qformer_config.use_dist_bias = False

            # self.decoder = RobertaModel.from_pretrained(
            #     args.qformer.model_name_or_path, config=self.qformer_config,
            # )

            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, self.encoder.config.hidden_size))
            self.projector1 = nn.Linear(4096, self.encoder.config.hidden_size)

            self.projector2 = nn.Linear(self.encoder.config.hidden_size, 4096)
        else:
            self.encoder = None
            self.decoder = None
            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, 4096))
            self.projector1 = self.projector2 = None

        self.ln_norm1 = nn.LayerNorm(self.encoder.config.hidden_size)
        self.ln_norm2 = nn.LayerNorm(4096)

        self.init_weight()

    def init_weight(self):
        self.query_token_embeds.data.normal_(mean=0.0, std=0.02)
        for proj in [self.projector1, self.projector2]:
            if proj:
                proj.weight.data.normal_(mean=0.0, std=0.02)
                if proj.bias is not None:
                    proj.bias.data.zero_()

        # for ln in [self.ln_norm1, self.ln_norm2]:
        #     ln.bias.data.zero_()
        #     ln.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens):
        if self.encoder:
            self.encoder.model.resize_token_embeddings(new_num_tokens)
        # if self.model is not None:
        #     self.model.resize_token_embeddings(new_num_tokens)

    @property
    def base_model(self):
        return self.decoder

    # as this value is added after initializing the model
    @property
    def bert_graph_pad_token(self):
        return self.args.bert_graph_pad_token

    @property
    def device(self):
        return self.decoder.device

    def gen_query_embeds_pt(self, graph, llm, llm_graph_pad_token_id):
        # vanilla prompt tuning
        batch_size = graph["question_input_ids"].shape[0]
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return query_embeds

    def gen_query_embeds_v2(self, graph, llm, llm_graph_pad_token_id):
        question_ids = graph["question_input_ids"]
        question_attention_mask = graph["question_attention_mask"]
        batch_size = question_ids.shape[0]

        if self.args.skip_graph_encoder:
            query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            all_graph_embeds = self.hypergraph_encoder(graph["graphs"], llm)

            x_s_idxes = graph["graphs"]["x_s_ptr"].tolist()
            # x_t_idxes = graph["graphs"]["x_t_ptr"].tolist()
            list_graph_embeds = []
            list_graph_attn = []
            for i in range(len(x_s_idxes) - 1):
                graph_embeds = torch.cat(
                    [
                        all_graph_embeds[0][x_s_idxes[i] : x_s_idxes[i + 1], :],  # s_nodes
                        # all_graph_embeds[1][x_t_idxes[i] : x_t_idxes[i + 1], :],  # t_nodes
                    ],
                    dim=0
                )
                list_graph_embeds.append(graph_embeds)
                list_graph_attn.append(torch.LongTensor([1] * graph_embeds.shape[0]))
            graph_embeds = torch.nn.utils.rnn.pad_sequence(list_graph_embeds, batch_first=True)
            graph_attention_mask = torch.nn.utils.rnn.pad_sequence(list_graph_attn, batch_first=True).to(graph_embeds.device)

            graph_embeds = self.projector1(graph_embeds)

            question_embeds = self.model.embeddings(question_ids)
            query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            
            inputs_embeds = torch.cat([query_embeds, question_embeds], dim=1)

            query_atts = torch.ones(query_embeds.shape[:-1]).to(inputs_embeds.device)
            attention_mask = torch.cat([query_atts, question_attention_mask], dim=1)

            question_output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_attention_mask,
                use_cache=False,
                query_length=self.num_query_tokens,
            )
            query_embeds = question_output.last_hidden_state[:, :self.num_query_tokens, :]

        res_embeds = self.projector2(query_embeds)
        res_embeds = self.ln_norm2(res_embeds)

        return res_embeds

    def gen_query_embeds_v3(self, qformer_inputs, llm, llm_graph_pad_token_id):
        question_ids = qformer_inputs["question_input_ids"]
        batch_size = question_ids.shape[0]

        if self.args.skip_encoder:
            res_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            qformer_inputs["qformer_inputs"].pop("labels")

            encoder_output = self.t5(only_return_query_embeds=True, **qformer_inputs["qformer_inputs"])
            res_embeds = encoder_output

            # res_embeds = self.ln_norm1(encoder_output)
            # print(res_embeds.norm(dim=-1))
            # encoder_hidden_states = encoder_output.last_hidden_state
            # encoder_attention_mask = qformer_inputs['qformer_inputs']['attention_mask']

            # if 'inter' in self.strategy:
            #     # inter
            #     query_embeds = self.query_embeds_generator(qformer_inputs, llm, llm_graph_pad_token_id)
            #     query_embeds = self.projector1(query_embeds)
            # else:
            #     # not inter
            #     query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)

            # query_atts = torch.ones(query_embeds.shape[:-1]).to(self.device)

            # question_output = self.decoder(
            #     inputs_embeds=query_embeds,
            #     attention_mask=query_atts,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_attention_mask,
            #     # query_length=self.num_query_tokens,
            #     # use_cache=False,
            # )
            # res_embeds = question_output.last_hidden_state

        # print(res_embeds.norm(dim=-1))
        res_embeds = self.projector2(res_embeds)
        res_embeds = self.ln_norm2(res_embeds)

        # print('q', query_embeds.norm(dim=-1))
        # print('r', res_embeds.norm(dim=-1))

        return res_embeds

    def forward(self, qformer_inputs, llm, llm_graph_pad_token_id):
        if self.strategy == "pt":
            query_embeds = self.gen_query_embeds_pt(qformer_inputs, llm, llm_graph_pad_token_id)
        elif self.strategy[:2] == "v2":
            query_embeds = self.gen_query_embeds_v2(qformer_inputs, llm, llm_graph_pad_token_id)
        elif self.strategy[:2] == "v3":
            query_embeds = self.gen_query_embeds_v3(qformer_inputs, llm, llm_graph_pad_token_id)
        else:
            raise NotImplementedError

        return query_embeds


class StructQformerLLM(nn.Module):
    def __init__(self, args, hypergraph_enc_config, llm_tokenizer, encoder_tokenizer, **kwargs) -> None:
        super().__init__()

        # set in init_tokenizer_and_embeds
        self.bert_graph_pad_token = None
        self.llm_graph_pad_token_id = None
        self.llm_pad_token_id = None

        self.args = args
        self.num_query_tokens = args.qformer.num_query_tokens

        if self.num_query_tokens > 0:
            self.qformer = StructQformer(args, hypergraph_enc_config)

            # TODO: not need in UnifiedSKG
            # encoder_tokenizer.add_tokens(["[TAB]", "[HEAD]", "[CELL]", "[ROW]",
            #                               "scinotexp"], special_tokens=True)
            # self.bert_graph_pad_token = encoder_tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PAD_TOKEN])[
            #     0
            # ]
            # self.qformer.resize_token_embeddings(len(encoder_tokenizer))

            if args.qformer.ckpt_path is not None and not args.qformer.skip_encoder:
                logger.info(f"loading qformer ckpt from {args.qformer.ckpt_path}")

                only_load_grpah_encoder = False
                if only_load_grpah_encoder:
                    state_dict = torch.load(os.path.join(args.qformer.ckpt_path, "Qformer.bin"))
                    prefix = 'hypergraph_encoder.'
                    state_dict = {k[len(prefix):]:v for k,v in state_dict.items() if k.startswith(prefix)}

                    self.qformer.hypergraph_encoder.load_state_dict(state_dict)
                else:
                    self.qformer.load_state_dict(torch.load(os.path.join(args.qformer.ckpt_path, "Qformer.bin")))


            self.qformer = self.qformer.to(kwargs["torch_dtype"])
        else:
            self.qformer = None

        self.llm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.llm.model_name_or_path, attn_implementation=args.llm.attn_implementation, **kwargs
        )
        self.init_tokenizer_and_embeds(llm_tokenizer, encoder_tokenizer, DEFAULT_GRAPH_PAD_TOKEN)

        if args.llm.finetuning_type == "full":
            for name, param in self.llm.named_parameters():
                param.requires_grad = True
        elif args.llm.finetuning_type == "lora":
            if args.ckpt_path is not None:
                logger.info(f"loading lora ckpt from {args.ckpt_path}")
                self.llm.load_adapter(args.ckpt_path)
            else:
                logger.info("adding lora model")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=args.llm.target_modules.split(","),
                    r=args.llm.r,
                    lora_alpha=args.llm.lora_alpha,
                    lora_dropout=args.llm.lora_dropout,
                )
                self.llm = get_peft_model(self.llm, peft_config)
                self.llm.print_trainable_parameters()
        elif args.llm.finetuning_type == "freeze":
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
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def init_tokenizer_and_embeds(
        self,
        llm_tokenizer: AutoTokenizer,
        bert_tokenizer,
        graph_pad_token=DEFAULT_GRAPH_PAD_TOKEN,
    ):
        llm = self.llm

        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        llm_tokenizer.add_tokens([graph_pad_token], special_tokens=True)
        self.llm_graph_pad_token_id = llm_tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PAD_TOKEN])[0]
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

    def construct_inputs_embeds(self, input_ids, qformer_inputs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.shape[0]

        if self.num_query_tokens > 0:
            res_embeds = self.qformer(qformer_inputs, self.llm, self.llm_graph_pad_token_id).to(inputs_embeds.dtype)

            graph_pad_st_idx = torch.argmax((input_ids == self.llm_graph_pad_token_id).int(), dim=1)
            graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

            new_inputs_embeds = torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            for i in range(batch_size):
                cur_inputs_embeds = inputs_embeds[i]
                cur_graph_pad_st_idx, cur_graph_pad_ed_idx = graph_pad_st_idx[i], graph_pad_ed_idx[i]

                new_inputs_embeds[i][:cur_graph_pad_st_idx] += cur_inputs_embeds[:cur_graph_pad_st_idx]
                new_inputs_embeds[i][cur_graph_pad_st_idx:cur_graph_pad_ed_idx] += res_embeds[i]
                new_inputs_embeds[i][cur_graph_pad_ed_idx:] += cur_inputs_embeds[cur_graph_pad_ed_idx:]
        else:
            new_inputs_embeds = inputs_embeds

        return new_inputs_embeds

    def forward(
        self,
        qformer_inputs,
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
        inputs_embeds = self.construct_inputs_embeds(input_ids, qformer_inputs)

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
        qformer_inputs,
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
        inputs_embeds = self.construct_inputs_embeds(input_ids, qformer_inputs)

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
