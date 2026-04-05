# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HCXOmni token boundary correctness."""
import pytest

DISCRETE_AUDIO_START = 128606
DISCRETE_AUDIO_VOCAB = 6561
DISCRETE_IMAGE_START = 135168
DISCRETE_IMAGE_VOCAB = 65536
IMAGE_TOKENS_PER_IMAGE = 729  # 27 × 27


def test_token_ranges_do_not_overlap():
    audio_end = DISCRETE_AUDIO_START + DISCRETE_AUDIO_VOCAB
    assert audio_end <= DISCRETE_IMAGE_START, (
        f"Audio range ends at {audio_end}, image starts at {DISCRETE_IMAGE_START}"
    )


def test_image_token_count_is_perfect_square():
    import math
    side = math.isqrt(IMAGE_TOKENS_PER_IMAGE)
    assert side * side == IMAGE_TOKENS_PER_IMAGE


def test_audio_token_extraction():
    seq = (
        [0] * 5
        + list(range(DISCRETE_AUDIO_START, DISCRETE_AUDIO_START + 10))
        + [0] * 3
    )
    extracted = [t - DISCRETE_AUDIO_START
                 for t in seq
                 if DISCRETE_AUDIO_START <= t < DISCRETE_AUDIO_START + DISCRETE_AUDIO_VOCAB]
    assert extracted == list(range(10))


def test_image_token_extraction():
    seq = (
        list(range(DISCRETE_IMAGE_START, DISCRETE_IMAGE_START + 5))
        + [DISCRETE_AUDIO_START + 1]
    )
    extracted = [t - DISCRETE_IMAGE_START
                 for t in seq
                 if DISCRETE_IMAGE_START <= t < DISCRETE_IMAGE_START + DISCRETE_IMAGE_VOCAB]
    assert extracted == [0, 1, 2, 3, 4]


def test_hcx_omni_classes_importable():
    from vllm.model_executor.models.hcx_omni import (  # noqa: F401
        HCXOmniForCausalLM,
        HCXOmniDummyInputsBuilder,
        HCXOmniMultiModalProcessor,
        HCXOmniProcessingInfo,
    )


def test_hcx_omni_registered_in_registry():
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    assert "HCXVisionV2ForCausalLM" in _MULTIMODAL_MODELS
    assert "HCXOmniForCausalLM" in _MULTIMODAL_MODELS
    mod, cls = _MULTIMODAL_MODELS["HCXVisionV2ForCausalLM"]
    assert mod == "hcx_omni"
    assert cls == "HCXOmniForCausalLM"


def test_hcx_omni_weight_mapper():
    from vllm.model_executor.models.hcx_omni import HCXOmniForCausalLM
    mapper = HCXOmniForCausalLM.hf_to_vllm_mapper
    assert "model.language_model." in mapper.orig_to_new_prefix
    assert "model.vision_model." in mapper.orig_to_new_prefix
    assert "model.audio_model." in mapper.orig_to_new_prefix
