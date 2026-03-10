# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVAX model registration plugin for vLLM.

Registers HCX model architectures without modifying the upstream registry.py.
Loaded automatically via pyproject.toml entry_points.
"""


def register():
    from vllm import ModelRegistry

    # HyperCLOVAX text-only LLM (uses Llama architecture)
    ModelRegistry.register_model(
        "HyperCLOVAXForCausalLM",
        "vllm.model_executor.models.llama:LlamaForCausalLM",
    )
    # HCX Omni: vision + audio encoder + LLM (single-stage inference)
    ModelRegistry.register_model(
        "HCXOmniForCausalLM",
        "vllm.model_executor.models.hcx_omni:HCXOmniForCausalLM",
    )
