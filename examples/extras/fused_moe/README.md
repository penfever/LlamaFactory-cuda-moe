# CUDA Fused MoE Kernel for Qwen3-MoE

This directory contains example configurations for training Qwen3-MoE models with
the CUDA fused MoE kernel, which provides **~5x speedup** over the default
HuggingFace implementation.

## Background

The default HuggingFace MoE implementation uses a Python for-loop to iterate over
experts, which results in low GPU utilization. The fused kernel uses Triton-based
grouped GEMM operations to process all experts in parallel.

## Prerequisites

Install the fused kernel package:

```bash
pip install git+https://github.com/woct0rdho/transformers-qwen3-moe-fused.git
# or via LLaMA-Factory extras:
pip install llamafactory[qwen3-moe-fused]
```

## Usage

### Full Fine-tuning (Multi-GPU)

```bash
llamafactory-cli train examples/extras/fused_moe/qwen3_moe_fused_sft.yaml
```

### LoRA Fine-tuning (Single GPU)

```bash
llamafactory-cli train examples/extras/fused_moe/qwen3_moe_fused_lora.yaml
```

For QLoRA (4-bit quantization), uncomment the quantization settings in the config.

## Key Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `use_v1_kernels` | `true` | Enables the fused MoE kernel |
| `deepspeed` | `ds_z2_config.json` | Use ZeRO-2 (ZeRO-3 is incompatible with MoE) |
| `attn` | `fa2` | Flash Attention 2 for attention layers |
| `enable_liger_kernel` | `true` | Additional kernel optimizations |

## Performance

| Configuration | GPU Memory | Throughput |
|--------------|------------|------------|
| Unfused (default) | High | ~20% GPU util |
| Fused kernel | Same | ~100% GPU util, **5x faster** |
| Fused + QLoRA | ~20GB | ~100% GPU util |

## Important Notes

1. **ZeRO-3 Incompatibility**: MoE models must use ZeRO-2 or lower. ZeRO-3 will
   throw an assertion error.

2. **Transformers Version**: The fused kernel works with transformers < 5.0.0.
   Transformers 5.0 will include native fused MoE support.

3. **License**: The grouped GEMM kernels in `qwen3_moe_fused` are derived from
   Unsloth and licensed under AGPLv3. The rest of the package is Apache 2.0.

## Supported Models

- Qwen3-30B-A3B (Qwen/Qwen3-30B-A3B)
- Other Qwen3-MoE variants

## Troubleshooting

**Kernel not applied**: Check that:
- `qwen3-moe-fused` is installed: `python -c "import qwen3_moe_fused; print('OK')"`
- `use_v1_kernels: true` is set in your config
- You're using a CUDA GPU
- Model architecture is `Qwen3MoeForCausalLM`

**OOM errors**: Try:
- Reducing `per_device_train_batch_size`
- Enabling gradient checkpointing
- Using QLoRA (4-bit quantization)
- Using a smaller `cutoff_len`
