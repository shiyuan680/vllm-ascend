# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from vllm/model_executor/models/qwen3_moe.py
# This file is a part of the vllm-ascend project.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
import vllm
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
                                               
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group
from vllm_ascend.ops.fused_moe import AscendFusedMoE

from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers import PretrainedConfig
from vllm.model_executor.layers.quantization import QuantizationConfig


class CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    }


class AscendQwen3MoeSparseMoeBlock(nn.Module):
    
    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = AscendFusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts")

        
        self.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        # TODO: need a better flag to indicate whether in profile run or not.
        if attn_metadata is None:
            # for profile run
            is_prefill = True
            enable_force_load_balance = True
        else:
            # is_prefill = attn_metadata.num_prefills > 0 is_prefill or
            enable_force_load_balance = False
            if hasattr(attn_metadata, 'with_prefill_across_dp'):
                is_prefill = attn_metadata.with_prefill_across_dp

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=self.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=None,
        )

        return hidden_states


vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock = AscendQwen3MoeSparseMoeBlock
