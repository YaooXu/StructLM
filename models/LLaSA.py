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

# from models.base_models.modeling_llama import LlamaForCausalLM
# from models.base_models.modeling_t5_qformer import T5ForConditionalGeneration
from models.graphormer.graphormer import Graphormer

from models.hytrel import HyperGraphEncoder
from utils.configure import Configure
from utils.dist import all_gather_with_grad, concat_all_gather

# from model.encoder_geo import GraphEncoder

from .Qformer_bert import BertConfig, BertLMHeadModel, BertModel

import torch.nn as nn

from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PromptTuningConfig, PromptTuningInit

from torch.nn import functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PAD_TOKEN = "[g_patch]"
DEFAULT_CVT_TOKEN = "[CVT]"

# class QueryEmbedsGenerator(nn.Module):
#     def __init__(self, num_query_tokens, llm_hidden_size=4096) -> None:
#         super().__init__()
#         self.num_query_tokens = num_query_tokens
#         self.llm_hidden_size = llm_hidden_size
#         self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, llm_hidden_size))
#         self.query_token_embeds.data.normal_(mean=0.0, std=0.02)
#         self.ln_norm = nn.LayerNorm(llm_hidden_size)

#     def forward(self, graph, llm, llm_graph_pad_token_id):
#         inputs_embeds = llm.get_input_embeddings()(graph["question_input_ids"])
#         batch_size = inputs_embeds.shape[0]

#         graph_pad_st_idx = torch.argmax((graph["question_input_ids"] == llm_graph_pad_token_id).int(), dim=1)
#         graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

#         query_token_embeds = self.query_token_embeds
#         # query_token_embeds = self.ln_norm(query_token_embeds)

#         new_inputs_embeds = torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
#         for i in range(batch_size):
#             cur_inputs_embeds = inputs_embeds[i]
#             cur_graph_pad_st_idx, cur_graph_pad_ed_idx = graph_pad_st_idx[i], graph_pad_ed_idx[i]

#             new_inputs_embeds[i][:cur_graph_pad_st_idx] += cur_inputs_embeds[:cur_graph_pad_st_idx]
#             new_inputs_embeds[i][cur_graph_pad_st_idx:cur_graph_pad_ed_idx] += query_token_embeds
#             new_inputs_embeds[i][cur_graph_pad_ed_idx:] += cur_inputs_embeds[cur_graph_pad_ed_idx:]

#         # LLM generates query tokens
#         output = llm(
#             inputs_embeds=new_inputs_embeds, attention_mask=graph["question_attention_mask"], output_hidden_states=True
#         )
#         last_hidden_state = output.hidden_states[-1]

#         query_embeds = torch.zeros(
#             (batch_size, self.num_query_tokens, self.llm_hidden_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device
#         )
#         for i in range(batch_size):
#             query_embeds[i] += last_hidden_state[i][graph_pad_st_idx[i] : graph_pad_ed_idx[i]]

#         return query_embeds

class GFormer(nn.Module):
    def __init__(self, args, encoder_tokenizer, hypergraph_enc_config, **kwargs) -> None:
        super().__init__()

        self.args = args
        self.num_query_tokens = args.qformer.num_query_tokens
        self.strategy = args.qformer.strategy

        self.encoder_tokenizer = encoder_tokenizer
        
        self.roberta_config = AutoConfig.from_pretrained(args.qformer.model_name_or_path)
        self.config = self.roberta_config

        # if "inter" in self.strategy:
        #     self.query_embeds_generator = QueryEmbedsGenerator(self.num_query_tokens)

        if self.strategy[:2] == "v3":
            self.graph_encoder = HyperGraphEncoder(hypergraph_enc_config)
            if self.args.qformer.freeze_encoder:
                self.graph_encoder.requires_grad_(False)

            self.roberta_config.add_cross_attention = True
            self.roberta_config.is_decoder = True
            self.roberta_config.query_length = self.num_query_tokens
            self.roberta_config.encoder_width = self.args.hytrel.hidden_size
            self.roberta_config.cross_attn_start_layer = hypergraph_enc_config.num_hidden_layers

            from .Qformer_roberta import RobertaForCausalLM

            self.Qformer = RobertaForCausalLM.from_pretrained(
                args.qformer.model_name_or_path, config=self.roberta_config,
            )

            if self.args.qformer.model_finetuning_type == 'full':
                self.Qformer.requires_grad_(True)
            elif self.args.qformer.model_finetuning_type == 'lora':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=args.qformer.target_modules.split(","),
                    r=args.qformer.r,
                    lora_alpha=args.qformer.lora_alpha,
                    lora_dropout=args.qformer.lora_dropout,
                )
                self.Qformer = get_peft_model(self.Qformer, peft_config)
            
            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, self.roberta_config.hidden_size))
            self.query_token_embeds.data.normal_(mean=0.0, std=self.roberta_config.initializer_range)

            self.projector1 = nn.Linear(self.graph_encoder.config.hidden_size, self.roberta_config.hidden_size)
            self.ln_norm1 = nn.LayerNorm(self.graph_encoder.config.hidden_size)

            self.itm_head = nn.Linear(self.roberta_config.hidden_size, 2)

        elif self.strategy[:2] == "pt":             
            self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, kwargs['llm_hidden_size']))
            self.query_token_embeds.data.normal_(mean=0.0, std=kwargs['llm_initializer_range'])           
        else:
            raise NotImplementedError

        self.init_tokenizer_and_embeds()
    
    def init_tokenizer_and_embeds(self):
        self.encoder_tokenizer.add_tokens(["<dec>", "<gen>"], special_tokens=True)
        self.encoder_tokenizer.dec_token_id = self.encoder_tokenizer.convert_tokens_to_ids("<dec>")
        self.encoder_tokenizer.gen_token_id = self.encoder_tokenizer.convert_tokens_to_ids("<gen>")
        
        self.Qformer.resize_token_embeddings(len(self.encoder_tokenizer))
    
    @property
    def device(self):
        return self.decoder.device

    def gen_query_embeds_pt(self, qformer_inputs, llm, llm_graph_pad_token_id):
        input_ids = qformer_inputs["input_ids"]
        batch_size = input_ids.shape[0]
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embeds

    def gen_query_embeds_v2(self, qformer_inputs, llm, llm_graph_pad_token_id):
        input_ids = qformer_inputs["input_ids"]
        question_attention_mask = qformer_inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        if self.args.qformer.skip_graph_encoder:
            query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if self.args.qformer.without_gnn:
                graph_embeds, graph_attention_mask = None, None
            else:
                graph_embeds, graph_attention_mask = self.graph_encoder(qformer_inputs["graphs"])

            query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            query_atts = torch.ones(query_embeds.shape[:-1]).to(query_embeds.device)
            
            # inputs_embeds = query_embeds
            # attention_mask = query_atts
            
            if self.args.qformer.only_query_embeds:
                attention_mask = query_atts
            else:
                attention_mask = torch.cat([query_atts, question_attention_mask], dim=1)

            query_output = self.model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_attention_mask,
                use_cache=False,
                return_dict=True,
                query_length=self.num_query_tokens,
                query_embeds=query_embeds, 
                only_query_embeds=self.args.qformer.only_query_embeds
            )


            query_embeds = query_output.last_hidden_state[:, :self.num_query_tokens, :]

        return query_embeds

    def gen_query_embeds(self, qformer_inputs, llm, llm_graph_pad_token_id):
        if self.strategy[:2] == "v2":
            query_embeds = self.gen_query_embeds_v2(qformer_inputs, llm, llm_graph_pad_token_id)
        elif self.strategy[:2] == "pt":
            query_embeds = self.gen_query_embeds_pt(qformer_inputs, llm, llm_graph_pad_token_id)
        else:
            raise NotImplementedError

        return query_embeds
    
    def forward(self, qformer_inputs, **kwargs):
        text_ids, text_atts = qformer_inputs["input_ids"], qformer_inputs["attention_mask"]
        batch_size = text_ids.shape[0]

        graph_embeds, graph_attention_mask = self.graph_encoder(qformer_inputs["graphs"])

        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        query_atts = torch.ones(query_embeds.shape[:-1]).to(query_embeds.device)

        ##================= Question Answering ========================##
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        query_output = self.Qformer.roberta(
            text_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        
        output_ids, labels = qformer_inputs["output_ids"], qformer_inputs["labels"]
        label_atts = torch.ones(labels.shape).to(query_embeds.device)

        attention_mask = torch.cat([query_atts, text_atts, label_atts], dim=1)
        lm_output = self.Qformer(
            output_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        lm_loss = lm_output.loss
        
        ###============== Graph-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_ids)
        text_attention_mask_world = concat_all_gather(text_atts)
        graph_embeds_world = all_gather_with_grad(graph_embeds)

        rank = dist.get_rank()
        sim_t2i = sim_i2t = torch.ones((batch_size, batch_size * dist.get_world_size()))
        
        with torch.no_grad():
            sim_t2i[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(0)
            sim_i2t[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(0)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative graph for each text
        graph_embeds_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            graph_embeds_neg.append(graph_embeds_world[neg_idx])
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)

        # select a negative text for each graph
        text_ids_neg = []
        text_atts_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_ids, text_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_atts, text_atts, text_atts_neg],
            dim=0,
        )

        query_embeds_itm = self.query_token_embeds.unsqueeze(0).expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_embeds_itm.size()[:-1], dtype=torch.long).to(
            graph_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        graph_embeds_all = torch.cat(
            [graph_embeds, graph_embeds_neg, graph_embeds], dim=0
        )  # pos, neg, pos
        graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(
            graph_embeds.device
        )

        output_itm = self.Qformer.roberta(
            text_ids_all,
            query_embeds=query_embeds_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=graph_embeds_all,
            encoder_attention_mask=graph_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_embeds_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
            dim=0,
        ).to(graph_embeds.device)
        itm_loss = F.cross_entropy(logits, itm_labels)
        
        return {
            "loss": lm_loss + itm_loss
        }

class LLaSA(nn.Module):
    def __init__(self, args, hypergraph_enc_config, llm_tokenizer, encoder_tokenizer, **kwargs) -> None:
        super().__init__()

        # set in init_tokenizer_and_embeds
        self.bert_graph_pad_token = None
        self.llm_graph_pad_token_id = None
        self.llm_pad_token_id = None

        self.args = args
        self.num_query_tokens = args.qformer.num_query_tokens

        # llm
        self.llm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.llm.model_name_or_path, 
            attn_implementation=args.llm.attn_implementation, 
            **kwargs
        )
        self.init_tokenizer_and_embeds(llm_tokenizer, encoder_tokenizer, DEFAULT_GRAPH_PAD_TOKEN)

        if args.llm.finetuning_type == "full":
            self.llm.requires_grad_(True)
        elif args.llm.finetuning_type == "lora":
            if args.llm.ckpt_path is not None:
                logger.info(f"loading lora ckpt from {args.llm.ckpt_path}")
                self.llm.load_adapter(args.llm.ckpt_path)
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
        # elif args.llm.finetuning_type == 'pt':
        #     self.llm.requires_grad_(False)

        #     self.query_token_embeds = nn.Parameter(torch.zeros(args.qformer.num_query_tokens, self.llm.config.hidden_size))
        #     self.query_token_embeds.data.normal_(mean=0.0, std=self.llm.config.initializer_range)
        elif args.llm.finetuning_type == "freeze":
            self.llm.requires_grad_(False)
        else:
            raise NotImplementedError

        # qformer
        if self.num_query_tokens > 0:

            qformer_kwargs = {}
            if self.args.qformer.strategy == 'pt':
                qformer_kwargs['llm_hidden_size'] = self.llm.config.hidden_size
                qformer_kwargs['llm_initializer_range'] = self.llm.config.initializer_range
            self.qformer = GFormer(args, encoder_tokenizer, hypergraph_enc_config, **qformer_kwargs)

            self.projector = nn.Linear(self.qformer.roberta_config.hidden_size, self.llm.config.hidden_size, dtype=kwargs["torch_dtype"])
            self.ln_norm = nn.LayerNorm(self.llm.config.hidden_size, dtype=kwargs["torch_dtype"])

            if args.qformer.ckpt_path is not None and not args.qformer.skip_graph_encoder:
                logger.info(f"loading qformer ckpt from {args.qformer.ckpt_path}")

                state_dict = torch.load(args.qformer.ckpt_path)
                self.qformer.load_state_dict(
                    state_dict,
                    # strict=False
                )

            self.qformer = self.qformer.to(kwargs["torch_dtype"])
        else:
            self.qformer = None

        if args.llm.ckpt_path is not None:
            logger.info(f"loading all ckpt from {args.llm.ckpt_path}")
            
            # this ckpt also include qformer
            self.load_state_dict(
                torch.load(os.path.join(args.llm.ckpt_path, "model.bin")),
                strict=False
            )

            self.to(kwargs["torch_dtype"])

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
        self.llm.gradient_checkpointing_enable()

    def init_tokenizer_and_embeds(
        self,
        llm_tokenizer: AutoTokenizer,
        bert_tokenizer,
        graph_pad_token=DEFAULT_GRAPH_PAD_TOKEN,
    ):
        llm = self.llm

        llm_tokenizer.add_tokens([graph_pad_token], special_tokens=True)
        self.llm_graph_pad_token_id = llm_tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PAD_TOKEN])[0]
        self.llm_pad_token_id = llm_tokenizer.pad_token_id
        llm.resize_token_embeddings(len(llm_tokenizer), mean_resizing=False)

    def construct_inputs_embeds(self, input_ids, qformer_inputs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.shape[0]

        query_embeds = self.qformer.gen_query_embeds(qformer_inputs, self.llm, self.llm_graph_pad_token_id).to(inputs_embeds.dtype)

        if query_embeds.shape[-1] != inputs_embeds.shape[-1]:
            # for v2
            res_embeds = self.projector(query_embeds)
            res_embeds = self.ln_norm(res_embeds)
        else:
            # prompt tunning
            res_embeds = query_embeds

        graph_pad_st_idx = torch.argmax((input_ids == self.llm_graph_pad_token_id).int(), dim=1)
        graph_pad_ed_idx = graph_pad_st_idx + self.num_query_tokens

        new_inputs_embeds = torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        for i in range(batch_size):
            cur_inputs_embeds = inputs_embeds[i]
            cur_graph_pad_st_idx, cur_graph_pad_ed_idx = graph_pad_st_idx[i], graph_pad_ed_idx[i]

            new_inputs_embeds[i][:cur_graph_pad_st_idx] += cur_inputs_embeds[:cur_graph_pad_st_idx]
            new_inputs_embeds[i][cur_graph_pad_st_idx:cur_graph_pad_ed_idx] += res_embeds[i]
            new_inputs_embeds[i][cur_graph_pad_ed_idx:] += cur_inputs_embeds[cur_graph_pad_ed_idx:]

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
        **loss_kwargs, # For compatibility with the transformers >= 4.46, https://github.com/huggingface/transformers/issues/34263
    ):
        if self.num_query_tokens > 0:
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
                **loss_kwargs,
            )
        else:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **loss_kwargs,
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
        if self.num_query_tokens > 0:
            inputs_embeds = self.construct_inputs_embeds(input_ids, qformer_inputs)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **gen_kwargs,
            )
        else:
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **gen_kwargs,
            )
            outputs = outputs[:, input_ids.shape[1]:]

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
