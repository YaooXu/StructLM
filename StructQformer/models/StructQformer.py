from collections import OrderedDict
import logging
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

from models.hytrel import HyperGraphEncoder

# from model.graph_encoder_geo import GraphEncoder

from .Qformer import BertConfig, BertLMHeadModel

import torch.nn as nn

from SQformer_dataset import DEFAULT_CVT_TOKEN, DEFAULT_GRAPH_PAD_TOKEN
from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)


class StructQformer(nn.Module):
    def __init__(self, args, hypergraph_enc_config) -> None:
        super().__init__()

        self.args = args

        self.cross_attention_freq = args.cross_attention_freq
        self.num_query_tokens = args.num_query_tokens
        self.encoder_model_path = args.encoder_model_path
        self.strategy = args.strategy

        self.encoder_config = BertConfig.from_pretrained(self.encoder_model_path)
        self.encoder_config.encoder_width = self.encoder_config.hidden_size
        # insert cross-attention layer every other block
        self.encoder_config.add_cross_attention = self.cross_attention_freq > 0
        self.encoder_config.cross_attention_freq = self.cross_attention_freq
        self.encoder_config.query_length = self.num_query_tokens
        if self.strategy[:2] == "v2":
            self.hypergraph_encoder = HyperGraphEncoder(hypergraph_enc_config)

            self.model = BertLMHeadModel.from_pretrained(
                self.encoder_model_path, config=self.encoder_config
            )
            self.query_token_embeds = nn.Parameter(
                torch.zeros(self.num_query_tokens, self.encoder_config.hidden_size)
            )
            self.projector = nn.Sequential(
                nn.Linear(self.encoder_config.hidden_size, 2048),
                nn.Tanh(),
                nn.Linear(2048, 4096),
            )
            # self.LayerNorm = nn.LayerNorm(
            #     self.encoder_config.hidden_size, eps=self.encoder_config.layer_norm_eps
            # )
        else:
            self.hypergraph_encoder = None
            self.model = None
            self.projector = None
            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, 4096))

        self.init_weight()

    def init_weight(self):
        self.query_token_embeds.data.normal_(mean=0.0, std=self.encoder_config.initializer_range)
        if self.projector is not None:
            for module in self.projector:
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.encoder_config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens):
        if self.model is not None:
            self.model.resize_token_embeddings(new_num_tokens)

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

    def gen_query_embeds_pt(self, graph):
        # vanilla prompt tuning
        batch_size = graph["question_input_ids"].shape[0]
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return query_embeds

    def gen_query_embeds_v2(self, graph):
        question_ids = graph["question_input_ids"]
        question_attention_mask = graph["question_attention_mask"]
        batch_size = question_ids.shape[0]

        if self.args.skip_graph_encoder:
            query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            graph_embeds = self.hypergraph_encoder(graph["graph"])

            idxes = graph["graph"]["ptr"].tolist()
            list_graph_embeds = []
            list_graph_attn = []
            for i in range(len(idxes) - 1):
                list_graph_embeds.append(graph_embeds[0][idxes[i]: idxes[i + 1], :])
                list_graph_attn.append(torch.LongTensor([1] * (idxes[i + 1] - idxes[i])))
            graph_embeds = torch.nn.utils.rnn.pad_sequence(list_graph_embeds, batch_first=True)
            graph_attention_mask = torch.nn.utils.rnn.pad_sequence(
                list_graph_attn, batch_first=True
            ).to(graph_embeds.device)

            question_embeds = self.model.bert.embeddings(question_ids)
            # add ln and dp
            query_embeds = self.model.bert.embeddings(input_embeds=self.query_token_embeds)
            query_embeds = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)

            input_embeds = torch.cat([question_embeds, query_embeds], dim=1)

            query_atts = torch.ones(query_embeds.shape[:-1]).to(self.device)
            attention_mask = torch.cat([question_attention_mask, query_atts], dim=1)

            question_output = self.model.bert(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_attention_mask,
                query_length=self.num_query_tokens,
                use_cache=False,
            )
            query_embeds = question_output.last_hidden_state[:, -self.num_query_tokens:, :]

        query_embeds = self.projector(query_embeds)

        return query_embeds

    def forward(self, graph):
        if self.strategy == "pt":
            query_embeds = self.gen_query_embeds_pt(graph)
        elif self.strategy[:2] == "v2":
            query_embeds = self.gen_query_embeds_v2(graph)
        else:
            raise NotImplementedError

        return query_embeds


class StructQformerLLM(nn.Module):
    def __init__(self, args, hypergraph_enc_config, llm_tokenizer, **kwargs) -> None:
        super().__init__()

        self.num_query_tokens = args.num_query_tokens

        # set in init_tokenizer_and_embeds
        self.bert_graph_pad_token = None
        self.llm_graph_pad_token_id = None
        self.llm_pad_token_id = None

        self.llm: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **kwargs
        )
        self.init_tokenizer_and_embeds(llm_tokenizer, DEFAULT_GRAPH_PAD_TOKEN)

        if not args.freeze_backbone:
            logger.info('loading lora model')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False,
                target_modules=args.target_modules.split(','),
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
        else:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

        if self.num_query_tokens > 0:
            self.qformer = StructQformer(args, hypergraph_enc_config)

            if self.qformer.hypergraph_encoder:
                # load graph encoder
                logger.info(f"loading hypergraph_encoder ckpt")
                state_dict = torch.load(
                    open(
                        'models/ckpts/hytrel/mp_rank_00_model_states.pt',
                        "rb",
                    )
                )

                new_state_dict = OrderedDict()
                logger.info(f"loading graph encoder")
                for k, v in state_dict["module"].items():
                    if "model" in k:
                        name = k[13:]  # remove `module.model.`
                        new_state_dict[name] = v
                self.qformer.hypergraph_encoder.load_state_dict(new_state_dict, strict=True)
        else:
            self.qformer = None

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
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def init_tokenizer_and_embeds(
        self,
        llm_tokenizer: AutoTokenizer,
        graph_pad_token=DEFAULT_GRAPH_PAD_TOKEN,
    ):
        llm = self.llm
        # qformer = self.qformer

        # bert_tokenizer.add_tokens(["[TAB]", "[HEAD]", "[CELL]", "[ROW]", "scinotexp"], special_tokens=True)
        # self.bert_graph_pad_token = bert_tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PAD_TOKEN])[
        #     0
        # ]
        # qformer.resize_token_embeddings(len(bert_tokenizer))

        llm_tokenizer.add_tokens([graph_pad_token], special_tokens=True)
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        self.llm_graph_pad_token_id = llm_tokenizer.convert_tokens_to_ids(
            [DEFAULT_GRAPH_PAD_TOKEN]
        )[0]
        self.llm_pad_token_id = llm_tokenizer.pad_token_id
        llm.resize_token_embeddings(len(llm_tokenizer), pad_to_multiple_of=8)

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for name, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                print(name, param.shape, num_params)
                trainable_params += num_params

        print(f"{trainable_params} / {all_param}, {trainable_params*100/all_param}%")
        return trainable_params, all_param

    def construct_inputs_embeds(self, input_ids, graph):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if self.num_query_tokens > 0:
            query_embeds = self.qformer(graph).to(inputs_embeds.dtype)

            graph_pad_st_idx = torch.argmax((input_ids == self.llm_graph_pad_token_id).int(), dim=1)
            graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

            batch_size = inputs_embeds.shape[0]
            for i in range(batch_size):
                inputs_embeds[i][graph_pad_st_idx[i]: graph_pad_ed_idx[i]] = query_embeds[i]

        return inputs_embeds

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
