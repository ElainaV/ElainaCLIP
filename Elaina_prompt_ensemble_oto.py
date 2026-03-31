

import os
from typing import Union, List
from pkg_resources import packaging
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.Tensor:
    if isinstance(texts, str):
        texts = [texts]

    sot, eot = _tokenizer.encoder["<|startoftext|>"], _tokenizer.encoder["<|endoftext|>"]
    all_ids = [[sot] + _tokenizer.encode(t) + [eot] for t in texts]

    dtype = torch.long if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0") else torch.int
    out   = torch.zeros(len(all_ids), context_length, dtype=dtype)

    for i, ids in enumerate(all_ids):
        if len(ids) > context_length:
            if truncate:
                ids = ids[:context_length]
                ids[-1] = eot
            else:
                raise RuntimeError(f"Input '{texts[i]}' is too long ({len(ids)}) for context {context_length}")
        out[i, :len(ids)] = torch.tensor(ids)
    return out

def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class ElainaDynamicPromptGenerator(nn.Module):
    def __init__(self, input_dim=768, prompt_len=4, prompt_dim=768):
        super().__init__()
        self.prompt_len = prompt_len
        self.generator  = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, prompt_len * prompt_dim)
        )

    def forward(self, img_emb: torch.Tensor) -> torch.Tensor:
        B = img_emb.size(0)
        out = self.generator(img_emb).view(B, self.prompt_len, -1)
        return out

class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()

        # ---------- 0. 读取 design 参数 ----------
        self.n_ctx        = design_details["Prompt_length"]
        self.depth        = design_details.get("learnabel_text_embedding_depth", 1)
        self.text_n_ctx   = design_details.get("learnabel_text_embedding_length", 4)

        learn_ratio       = design_details.get("learnable_ratio", .5)
        self.learn_ctx    = int(round(self.n_ctx * learn_ratio))
        self.dyn_ctx      = self.n_ctx - self.learn_ctx

        ctx_dim           = clip_model.ln_final.weight.shape[0]
        dtype             = clip_model.transformer.get_cast_dtype()

        self.classnames            = ["object"]
        self.state_normal_list     = ["{}"]
        self.state_anomaly_list    = ["damaged {}"]
        self.normal_num            = len(self.state_normal_list)
        self.anomaly_num           = len(self.state_anomaly_list)

        ctx_pos = torch.empty(1, self.normal_num, self.learn_ctx, ctx_dim, dtype=dtype)
        ctx_neg = torch.empty(1, self.anomaly_num, self.learn_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_pos, std=0.02)
        nn.init.normal_(ctx_neg, std=0.02)
        self.ctx_pos = nn.Parameter(ctx_pos)
        self.ctx_neg = nn.Parameter(ctx_neg)

        tmpl_prefix = " ".join(["X"] * self.learn_ctx)
        def _build_templates(state_list):
            return [f"{tmpl_prefix} {tpl.format(nm)}." for tpl in state_list for nm in self.classnames]

        prompt_pos_ids = torch.cat([tokenize(p) for p in _build_templates(self.state_normal_list)])
        prompt_neg_ids = torch.cat([tokenize(p) for p in _build_templates(self.state_anomaly_list)])

        with torch.no_grad():
            emb_pos = clip_model.token_embedding(prompt_pos_ids).type(dtype)
            emb_neg = clip_model.token_embedding(prompt_neg_ids).type(dtype)
            emb_pos = emb_pos.view(self.normal_num, 1, -1, ctx_dim)
            emb_neg = emb_neg.view(self.anomaly_num,1, -1, ctx_dim)

        self.register_buffer("token_prefix_pos", emb_pos[:, :, :1, :])
        self.register_buffer("token_prefix_neg", emb_neg[:, :, :1, :])

        suffix_len = 77 - (1 + self.n_ctx)
        self.register_buffer("token_suffix_pos", emb_pos[:, :, 1+self.n_ctx : 1+self.n_ctx+suffix_len, :])
        self.register_buffer("token_suffix_neg", emb_neg[:, :, 1+self.n_ctx : 1+self.n_ctx+suffix_len, :])

        self.register_buffer("token_ids_pos", prompt_pos_ids[:self.normal_num])
        self.register_buffer("token_ids_neg", prompt_neg_ids[:self.anomaly_num])

        self.dynamic_gen = ElainaDynamicPromptGenerator(prompt_len=self.dyn_ctx, prompt_dim=ctx_dim)
        if design_details.get("freeze_dynamic", False):
            for p in self.dynamic_gen.parameters():
                p.requires_grad = False
            print("[DynPrompt] MLP frozen (inference mode)")

        self.compound_prompts_depth = self.depth
        if self.compound_prompts_depth > 1:
            self.compound_prompts_text = nn.ParameterList([
                nn.Parameter(torch.empty(self.text_n_ctx, ctx_dim))
                for _ in range(self.compound_prompts_depth - 1)
            ])
            for p in self.compound_prompts_text:
                nn.init.normal_(p, std=0.02)

            single_fc = nn.Linear(ctx_dim, 896)
            self.compound_prompt_projections = _get_clones(
                single_fc, self.compound_prompts_depth - 1
            )
        else:
            self.compound_prompts_text      = []
            self.compound_prompt_projections = nn.ModuleList()

    def forward(self, cls_id=None, image_embedding=None):
        ctx_dim = self.ctx_pos.shape[-1]

        if image_embedding is None:
            raise RuntimeError("image_embedding (batch_size=1)")

        if image_embedding.size(0) != 1:
            raise RuntimeError("batch_size==1")

        dyn_in = image_embedding
        dyn_prompt = self.dynamic_gen(dyn_in)

        ctx_pos = torch.cat([self.ctx_pos[0], dyn_prompt.expand(self.normal_num, -1, -1)], dim=1)
        ctx_neg = torch.cat([self.ctx_neg[0], dyn_prompt.expand(self.anomaly_num, -1, -1)], dim=1)

        prompts_pos = torch.cat([self.token_prefix_pos[0], ctx_pos, self.token_suffix_pos[0]], dim=1)
        prompts_neg = torch.cat([self.token_prefix_neg[0], ctx_neg, self.token_suffix_neg[0]], dim=1)

        prompts   = torch.cat([prompts_pos, prompts_neg], dim=0)
        token_ids = torch.cat([self.token_ids_pos, self.token_ids_neg], dim=0)

        return prompts, token_ids, self.compound_prompts_text
