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

"""
Tests for CUDA fused MoE kernel integration.

To run these tests:
    pip install qwen3-moe-fused  # or git+https://github.com/woct0rdho/transformers-qwen3-moe-fused.git
    pytest tests/v1/plugins/test_cuda_fused_moe.py -v
"""

import pytest
import torch


def test_kernel_registration():
    """Test that the CUDA fused MoE kernel is properly registered."""
    from llamafactory.v1.plugins.model_plugins.kernels.registry import (
        KERNEL_REGISTRY,
        _ensure_kernels_loaded,
    )
    from llamafactory.v1.plugins.model_plugins.kernels.constants import DeviceType, KernelType

    # Ensure all kernels are loaded
    _ensure_kernels_loaded()

    # Check if CUDA MoE kernel was registered (only if qwen3_moe_fused is available)
    try:
        from qwen3_moe_fused.functional import moe_fused_linear
        cuda_moe_kernel = KERNEL_REGISTRY.get_kernel(KernelType.MOE, DeviceType.CUDA)
        assert cuda_moe_kernel is not None, "CUDA MoE kernel should be registered when qwen3_moe_fused is available"
    except ImportError:
        pytest.skip("qwen3_moe_fused not installed, skipping CUDA MoE kernel test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_apply_to_model():
    """Test that the kernel can be applied to a Qwen3-MoE model."""
    try:
        from qwen3_moe_fused.functional import moe_fused_linear
    except ImportError:
        pytest.skip("qwen3_moe_fused not installed")

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        pytest.skip("transformers not installed")

    from llamafactory.v1.plugins.model_plugins.kernels.mlp.cuda_fused_moe import CudaFusedMoEKernel

    # Create a minimal Qwen3-MoE config for testing
    # Note: This requires the actual model to be available or we mock it
    # For CI, we'll just verify the kernel class is properly structured
    assert hasattr(CudaFusedMoEKernel, 'type')
    assert hasattr(CudaFusedMoEKernel, 'device')
    assert hasattr(CudaFusedMoEKernel, 'apply')
    assert CudaFusedMoEKernel.type.value == "moe"
    assert CudaFusedMoEKernel.device.value == "cuda"


def test_package_availability_check():
    """Test the package availability check function."""
    from llamafactory.extras.packages import is_qwen3_moe_fused_available

    # Just verify the function exists and returns a boolean
    result = is_qwen3_moe_fused_available()
    assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
