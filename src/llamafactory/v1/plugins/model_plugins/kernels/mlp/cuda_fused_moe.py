# Copyright 2025 the LlamaFactory team.
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
#
# CUDA Fused MoE Kernel for Qwen3-MoE models
# Based on https://github.com/woct0rdho/transformers-qwen3-moe-fused
# The grouped_gemm kernels are licensed under AGPLv3 (from Unsloth)

import types
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .....extras.packages import is_transformers_version_greater_than
from .....extras.types import HFModel
from ..constants import DeviceType, KernelType
from ..registry import MetaMoEKernel


if TYPE_CHECKING:
    from torch import nn


def _is_qwen3_moe_fused_available() -> bool:
    """Check if qwen3_moe_fused package is available."""
    try:
        from qwen3_moe_fused.functional import moe_fused_linear
        from qwen3_moe_fused.kernels.indexing import get_expert_counts_and_idx
        return True
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


class CudaMoeFused:
    """CUDA fused MoE implementation using qwen3_moe_fused kernels."""

    @staticmethod
    def _stack_expert_weights(experts: "nn.ModuleList") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack expert weights into fused format (num_experts, out_features, in_features)."""
        gate_weights = torch.stack([e.gate_proj.weight for e in experts], dim=0)
        up_weights = torch.stack([e.up_proj.weight for e in experts], dim=0)
        down_weights = torch.stack([e.down_proj.weight for e in experts], dim=0)
        return gate_weights, up_weights, down_weights

    @staticmethod
    def qwen3moe_sparse_moe_block_forward(self, hidden_states: torch.Tensor):
        """
        Fused forward pass for Qwen3MoeSparseMoeBlock.

        This replaces the inefficient for-loop over experts with a fused grouped GEMM
        kernel that processes all experts in parallel.
        """
        from qwen3_moe_fused.functional import moe_fused_linear
        from qwen3_moe_fused.kernels.indexing import get_expert_counts_and_idx

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        M = batch_size * sequence_length

        hidden_states = hidden_states.view(M, hidden_dim)

        # Router logits: (M, num_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        # routing_weights, selected_experts: (M, num_selected)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Expand hidden states for each selected expert
        num_selected = self.top_k
        hidden_states = hidden_states.unsqueeze(1).expand(M, num_selected, hidden_dim)
        hidden_states = hidden_states.reshape(M * num_selected, hidden_dim)
        selected_experts = selected_experts.view(M * num_selected)

        # Sort by expert for memory coalescence
        m_sizes, sort_idx, inv_sort_idx = get_expert_counts_and_idx(selected_experts, self.num_experts)
        hidden_states = hidden_states[sort_idx]

        # Stack expert weights if not already cached
        if not hasattr(self, '_fused_gate_weight'):
            gate_w, up_w, down_w = CudaMoeFused._stack_expert_weights(self.experts)
            # Register as buffers so they move with the model
            self.register_buffer('_fused_gate_weight', gate_w, persistent=False)
            self.register_buffer('_fused_up_weight', up_w, persistent=False)
            self.register_buffer('_fused_down_weight', down_w, persistent=False)

        # Fused computation
        gate_h = moe_fused_linear(hidden_states, self._fused_gate_weight, m_sizes)
        up_h = moe_fused_linear(hidden_states, self._fused_up_weight, m_sizes)
        hidden_states = F.silu(gate_h) * up_h
        del gate_h, up_h
        hidden_states = moe_fused_linear(hidden_states, self._fused_down_weight, m_sizes)

        # Unsort
        hidden_states = hidden_states[inv_sort_idx]

        # Combine expert outputs with routing weights
        hidden_states = hidden_states.view(M, num_selected, hidden_dim)
        hidden_states = torch.einsum("beo,be->bo", hidden_states, routing_weights)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits


# Model architecture to MoE block mapping
kernel_moe_mapping = {}

# Only register for transformers < 5.0.0 (Transformers 5 will have native fused MoE)
if not is_transformers_version_greater_than("5.0.0"):
    kernel_moe_mapping["Qwen3MoeForCausalLM"] = {
        "Qwen3MoeSparseMoeBlock": CudaMoeFused.qwen3moe_sparse_moe_block_forward
    }


class CudaFusedMoEKernel(MetaMoEKernel):
    """
    CUDA Fused MoE Kernel for Qwen3-MoE models.

    This kernel provides ~5x speedup over the default HuggingFace implementation
    by using Triton-based grouped GEMM kernels instead of a Python for-loop
    over experts.

    Requirements:
        - CUDA-capable GPU
        - qwen3_moe_fused package: pip install qwen3-moe-fused
          or: pip install git+https://github.com/woct0rdho/transformers-qwen3-moe-fused.git

    Note:
        The grouped_gemm kernels in qwen3_moe_fused are licensed under AGPLv3
        (derived from Unsloth). The rest of the package is Apache 2.0.
    """

    type = KernelType.MOE
    device = DeviceType.CUDA

    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        """Apply CUDA fused MoE kernel to the model."""
        # Check prerequisites
        if not _is_cuda_available():
            return model

        if not _is_qwen3_moe_fused_available():
            # Silently return - the kernel is optional
            return model

        # Check if model architecture is supported
        archs = getattr(model.config, "architectures", [])
        target_moe_mapping = None
        for arch in archs:
            if arch in kernel_moe_mapping:
                target_moe_mapping = kernel_moe_mapping[arch]
                break

        if target_moe_mapping is None:
            return model

        # Patch forward methods
        patched_count = 0
        for module in model.modules():
            class_name = module.__class__.__name__
            if class_name in target_moe_mapping:
                new_forward_func = target_moe_mapping[class_name]
                module.forward = types.MethodType(new_forward_func, module)
                patched_count += 1

        if patched_count > 0:
            print(f"Applied CUDA fused MoE kernel to {patched_count} MoE blocks.")

        return model
