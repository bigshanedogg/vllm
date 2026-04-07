# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only HCXOmni model (HyperCLOVAX + Qwen2.5-VL Vision + Qwen2Audio).

HyperCLOVAX + Qwen2.5-VL Vision + Qwen2Audio.
Supports dual-stream multimodal input:
  - Discrete tokens: pre-computed VQ indices embedded via LLM embedding table
  - Continuous embeddings: encoder outputs projected to LLM hidden dimension

Weight mapping from HF checkpoint:
  model.vision_model.*              -> visual.*
  model.mm_projector.*              -> mm_projector.*
  model.audio_model.*               -> audio_tower.*
  model.audio_projector.*           -> audio_projector.*
  model.video_audio_compressor.*    -> video_audio_compressor.*
  model.language_model.*            -> language_model.*
"""

import copy
import inspect
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias
import glob as _glob
import sys
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
from transformers import AutoConfig, AutoModel, BatchFeature, Siglip2VisionConfig, Siglip2VisionModel
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_audio import Qwen2AudioEncoder
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from .qwen2_vl import Qwen2VLProcessingInfo
from .qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoPixelInputs,
    Qwen2_5_VisionTransformer,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


# === TA-Tok Discrete Image Encoder === #


def _ta_tok_models_make(model_spec, args=None, load_sd=False) -> torch.nn.Module:
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    model_params = inspect.signature(_ta_tok_models[model_spec["name"]]).parameters
    if "kwargs" not in model_params:
        model_args = {k: v for k, v in model_args.items() if k in model_params}
    model = _ta_tok_models[model_spec["name"]](**model_args)
    if load_sd:
        if (
            ("abs_pe" in model_spec["sd"])
            and hasattr(model, "abs_pe")
            and model_spec["sd"]["abs_pe"].shape != model.abs_pe.shape
        ):
            del model_spec["sd"]["abs_pe"]
        msg = model.load_state_dict(model_spec["sd"], strict=False)
        print(msg)
    return model


class _TaTokBottleneck(nn.Module):
    def __init__(
        self,
        bottleneck_dim: int,
        input_dim: int,
        output_dim: int,
        token_nums: int,
        regularizer=None,
        **kwargs,
    ):
        super().__init__()
        self.token_nums = token_nums
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bottleneck_dim > 0:
            self.bottleneck_dim = bottleneck_dim
        else:
            assert (
                self.input_dim == self.output_dim
            ), "input_dim and output_dim must be the same when bottleneck_dim is not specified"
            self.bottleneck_dim = self.input_dim

        self.project_dim = self.bottleneck_dim

        if self.bottleneck_dim > 0:
            self.in_linear = nn.Linear(self.input_dim, self.project_dim)
            self.out_linear = nn.Linear(self.bottleneck_dim, self.output_dim)
        else:
            self.in_linear = self.out_linear = lambda x: x

        regularizer["args"]["dim"] = self.bottleneck_dim
        regularizer["args"]["token_nums"] = self.token_nums
        self.regularizer = _ta_tok_models_make(regularizer)

    def project_in(self, x):
        assert len(x.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        z = self.in_linear(x)
        return z

    def project_out(self, z_cat):
        z = self.out_linear(z_cat)
        return z

    def decode(self, bottleneck_rep):
        regularized_z = self.regularizer.decode(bottleneck_rep)
        return self.project_out(regularized_z)

    def forward(self, x):
        z = self.project_in(x)
        projected_z = z
        regularized_output = self.regularizer(z)
        x_hat = self.project_out(regularized_output["regularized_z"])
        bottleneck_rep = regularized_output.pop("bottleneck_rep")
        return {
            "output": x_hat,
            "bottleneck_rep": bottleneck_rep,
            "projected_z": projected_z,
            **regularized_output,
        }


class _TaTokSimVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        l2_normalized=False,
        same_index_shape=True,
        stochastic=False,
        stochastic_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        assert isinstance(l2_normalized, bool)
        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature

        if self.stochastic:
            if stochastic_temperature > 0:
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else:
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        self.embedding_proj = nn.Linear(self.dim, self.dim)

        self.same_index_shape = same_index_shape

    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic

    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1 / temperature

    @torch.autocast(device_type="cuda", enabled=False)
    def get_emb(self):
        emb = self.embedding_proj(self.embedding.weight)
        if self.l2_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z):
        emb = self.get_emb()
        z = z.to(emb)
        assert len(z.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        if self.l2_normalized:
            z = F.normalize(z, p=2, dim=-1)

        z_flattened = rearrange(z, "b n d -> (b n) d")

        if self.stochastic:
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, emb)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(emb**2, dim=1)
                - 2
                * torch.einsum("bd,dn->bn", z_flattened, rearrange(emb, "n d -> d n"))
            )
            q_indices = torch.argmin(d, dim=1)

        quantized = F.embedding(
            q_indices,
            emb,
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        ).view(z.shape)

        quantized = z + (quantized - z).detach()

        if self.same_index_shape:
            q_indices = q_indices.reshape(quantized.shape[0], quantized.shape[1])

        return_dict = {
            "unregularized_z": z,
            "emb": emb,
            "regularized_z": quantized,
            "bottleneck_rep": q_indices,
        }
        return return_dict

    def get_codebook_entry(self, indices, shape=None):
        indices_shape = indices.shape
        indices_flatten = rearrange(indices, "... -> (...)")

        emb = self.get_emb()
        z_q = F.embedding(indices_flatten, emb)
        if self.l2_normalized:
            z_q = F.normalize(z_q, p=2, dim=-1)

        if shape is not None:
            z_q = z_q.reshape(shape)
        else:
            z_q = z_q.reshape([*indices_shape, self.dim])
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)


_ta_tok_models = {
    "simvq": _TaTokSimVectorQuantizer,
    "bottleneck": _TaTokBottleneck,
}


class _TaTokScalingLayer(nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        self.register_buffer("shift", torch.Tensor(mean)[None, :, None, None])
        self.register_buffer("scale", torch.Tensor(std)[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

    def inv(self, inp):
        return inp * self.scale + self.shift


class TextAlignedTokenizer(nn.Module):
    def __init__(
        self,
        bottleneck,
        bottleneck_token_num=256,
        input_size=384,
        teacher="google/siglip2-so400m-patch14-384",
        input_type="quant",
        pool_scale=1,
        decoder_depth=3,
        select_layer_id=-2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.bottleneck_token_num = bottleneck_token_num
        self.teacher = teacher
        self.input_type = input_type
        self.pool_scale = pool_scale
        self.decoder_depth = decoder_depth
        self.select_layer_id = select_layer_id

        self.bottleneck_dim = bottleneck["args"]["bottleneck_dim"]

        self.encoder_config = AutoConfig.from_pretrained(teacher)
        self.encoder = AutoModel.from_config(self.encoder_config).vision_model

        self.encoder_hidden_dim = self.encoder.config.hidden_size

        self.decoder_config = Siglip2VisionConfig()
        self.decoder_config.update(
            {
                "patch_size": 1,
                "num_hidden_layers": self.decoder_depth,
                "num_channels": self.bottleneck_dim,
                "hidden_size": self.encoder_hidden_dim,
            }
        )
        self.decoder = Siglip2VisionModel(self.decoder_config)

        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim), nn.Tanh()
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
        )

        bottleneck_args = {
            "token_nums": self.bottleneck_token_num,
            "input_dim": self.encoder_hidden_dim,
            "output_dim": self.bottleneck_dim,
        }
        self.bottleneck = _ta_tok_models_make(bottleneck, args=bottleneck_args)

        self.scale_layer = _TaTokScalingLayer(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_resize = Resize((self.input_size, self.input_size))

    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_checkpoint(cls, ckpt, load_teacher=True, **kwargs):
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckpt_kwargs = ckpt["model"]["args"]
        print(ckpt_kwargs)
        model = cls(**kwargs, **ckpt_kwargs)
        sd = ckpt["model"]["sd"]
        if not load_teacher:
            sd = {k: v for k, v in sd.items() if not k.startswith("teacher")}
        model.load_state_dict(sd, strict=True)
        return model

    def encode(self, x, **kwargs):
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.scale_layer(x)
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = self.image_resize(x)
        # Manually iterate through encoder layers to collect hidden_states,
        # because SiglipEncoder.forward() ignores output_hidden_states.
        hidden_states = self.encoder.embeddings(x)
        all_hidden_states = [hidden_states]
        for layer in self.encoder.encoder.layers:
            hidden_states = layer(hidden_states, attention_mask=None)
            all_hidden_states.append(hidden_states)
        vq_feats = all_hidden_states[self.select_layer_id]

        pool_scale = self.pool_scale
        pool_scale = kwargs.get("pool_scale", pool_scale)
        if pool_scale != 1:
            vq_feats = self.avg_pool(vq_feats, pool_scale)
        vq_feats = self.encode_task_layer(vq_feats.to(x))

        bottleneck_out = self.bottleneck(vq_feats)
        z = bottleneck_out.pop("output")

        return {
            "encoded": z,
            "pool_scale": pool_scale,
            "vq_feats": vq_feats,
            **bottleneck_out,
        }

    def avg_pool(self, z, pool_scale=1):
        if z.ndim == 3:
            b, n, c = z.shape
            p = int(n**0.5)
            z = rearrange(z, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
        else:
            b, c, p, _ = z.shape
        p_s = int(p // pool_scale)
        z = F.avg_pool2d(
            z, kernel_size=(pool_scale, pool_scale), stride=(pool_scale, pool_scale)
        ).contiguous()
        z = rearrange(z, "b c p1 p2 -> b (p1 p2) c")
        return z

    def decode(self, z):
        if z.ndim == 4:
            z = rearrange(z, "b c p1 p2 -> b (p1 p2) c")
        attention_mask = torch.ones(z.shape[:2], dtype=torch.int, device=z.device)
        p = int(z.shape[1] ** 0.5)
        spatial_shape = torch.tensor([[p, p]] * z.shape[0], device=self.device)
        z = self.decoder(
            z, attention_mask, spatial_shape, output_hidden_states=True
        ).last_hidden_state
        z = self.decode_task_layer(z)
        return z

    def decode_from_bottleneck(self, bottleneck_rep):
        z = self.bottleneck.decode(bottleneck_rep)
        p = int(z.shape[1] ** 0.5)
        z = rearrange(z, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
        return self.decode(z)

    def forward(self, data, **kwargs):
        encode_output = self.encode(data, **kwargs)
        vq_feats = encode_output["encoded"]
        p = int(vq_feats.shape[1] ** 0.5)
        vq_feats = rearrange(vq_feats, "b (h w) c -> b c h w", h=p, w=p)
        pred_feats = self.decode(vq_feats)

        if self.input_type == "quant":
            z = encode_output["regularized_z"]
        elif self.input_type == "indices":
            z = encode_output["bottleneck_rep"]
        elif self.input_type == "rec":
            z = pred_feats
        encode_output["encoded"] = z
        return encode_output


# === TA-Tok Discrete Image Encoder (lazy singleton) === #

class _TaTokEncoderManager:
    """Lazily loads and caches the TA-Tok discrete image encoder.

    Thread-safe singleton that encodes PIL images into 729 VQ codebook indices.
    The decoder is deleted after loading to save ~1GB GPU memory.
    """

    _lock = threading.Lock()
    _model = None

    @classmethod
    def get_model(cls, checkpoint_path: str, device: str = "cuda"):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info(
                        "Loading TA-Tok encoder from %s", checkpoint_path
                    )
                    model = TextAlignedTokenizer.from_checkpoint(
                        checkpoint_path,
                        load_teacher=False,
                        input_type="indices",
                    )
                    model.eval()
                    model = model.to(device)
                    # Delete decoder components (not needed for encoding)
                    if hasattr(model, "decoder"):
                        del model.decoder
                    if hasattr(model, "decode_task_layer"):
                        del model.decode_task_layer
                    torch.cuda.empty_cache()
                    cls._model = model
                    logger.info("TA-Tok encoder loaded (device=%s)", device)
        return cls._model

    @classmethod
    def encode_images(
        cls,
        images: list,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> list[torch.Tensor]:
        """Encode PIL images to VQ codebook indices.

        Args:
            images: list of PIL Images
            checkpoint_path: path to ta_tok.pth
            device: cuda device string

        Returns:
            list of 1-D tensors, each shape [729] with indices in [0, 65535]
        """
        model = cls.get_model(checkpoint_path, device)
        tensors = []
        for img in images:
            img_resized = img.convert("RGB").resize(
                (384, 384), Image.BICUBIC
            )
            tensors.append(to_tensor(img_resized))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model.encode(batch)
        indices = output["bottleneck_rep"]  # [B, 729]
        return [indices[i].cpu() for i in range(len(images))]


# === CosyVoice2 Discrete Audio Encoder Manager === #

class _CosyvoiceEncoderManager:
    """Lazy singleton for CosyVoice2 (FSQ) discrete audio encoder.

    Extracts model.discrete_audio_model.* weights from the safetensors
    checkpoint and loads CosyvoiceEncoder on first use. Results are cached
    across calls so the GPU overhead is paid only once per server process.
    """

    _lock = threading.Lock()
    _model = None

    @classmethod
    def get_model(cls, model_dir: str, device: str = "cuda"):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    cls._model = cls._load(model_dir, device)
        return cls._model

    @classmethod
    def _load(cls, model_dir: str, device: str):
        from safetensors import safe_open

        prefix = "model.discrete_audio_model."
        shard_paths = sorted(_glob.glob(f"{model_dir}/model-*.safetensors"))
        if not shard_paths:
            shard_paths = [f"{model_dir}/model.safetensors"]

        state_dict: dict[str, "torch.Tensor"] = {}
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith(prefix):
                        state_dict[key[len(prefix):]] = sf.get_tensor(key)

        if not state_dict:
            raise RuntimeError(
                f"No '{prefix}*' weights found in {model_dir}. "
                "CosyVoice2 inline encoding unavailable."
            )

        # Import CosyvoiceEncoder from the model directory (trust_remote_code)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from cosyvoice import CosyvoiceEncoder  # noqa: PLC0415

        model = CosyvoiceEncoder()
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            logger.warning(
                "CosyvoiceEncoder load_state_dict: missing=%s unexpected=%s",
                missing, unexpected,
            )
        model.eval()
        model.freeze()
        model = model.to(device)
        logger.info(
            "CosyvoiceEncoder (CosyVoice2) loaded from %s on %s", model_dir, device
        )
        return model

    @classmethod
    def encode_audios(
        cls,
        audios: list,
        model_dir: str,
        device: str = "cuda",
    ) -> list:
        """Encode raw waveforms to CosyVoice2 discrete token tensors.

        Args:
            audios: list of 1-D float32 numpy arrays at 16 kHz.
            model_dir: HuggingFace model directory (contains cosyvoice.py
                and safetensors checkpoint).
            device: CUDA device string.

        Returns:
            list of 1-D torch.LongTensor with discrete indices in [0, 6561].
        """
        model = cls.get_model(model_dir, device)
        result = []
        for wav_np in audios:
            wav_t = (
                torch.tensor(wav_np, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                code = model(wav_t)  # returns (code,) or Tensor
            if isinstance(code, tuple):
                code = code[0]
            result.append(code.squeeze(0).cpu().long())
        return result


# === MambaMia Video-Audio Compressor === #

class _MambaMia2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(self.weight.dtype)


class _MambaMia2Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _MambaMia2Block(nn.Module):
    """Single MambaMia2 block with SSM mixer and gated pooling attention."""

    def __init__(self, hidden_size, layer_idx, chunk_size=25):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.mixer = Mamba2(
            d_model=hidden_size,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=1,
            layer_idx=layer_idx,
            chunk_size=256,
        )
        self.norm = _MambaMia2RMSNorm(hidden_size)

        # Gated Pooling Attention (GPA) components
        self.chunk_size_gpa = chunk_size
        self.weight_fc = nn.Linear(hidden_size, 1)
        self.gate_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states

        # Gated Pooling Attention
        bsz, seq_len, dim = hidden_states.shape
        chunk_plus_one = self.chunk_size_gpa + 1
        if seq_len % chunk_plus_one == 0:
            n_chunk = seq_len // chunk_plus_one
            h_4d = hidden_states.view(bsz, n_chunk, chunk_plus_one, dim)
            content = h_4d[:, :, :self.chunk_size_gpa, :]
            query = h_4d[:, :, self.chunk_size_gpa:, :]

            weights = torch.softmax(self.weight_fc(content), dim=2)
            pooled = (content * weights).sum(dim=2, keepdim=True)
            gate = torch.sigmoid(self.gate_fc(query))
            updated_query = query + gate * pooled

            h_4d = torch.cat([content, updated_query], dim=2)
            hidden_states = h_4d.view(bsz, seq_len, dim)

        return hidden_states


class _MambaMia2Model(nn.Module):
    """MambaMia2 Model backbone matching OmniServe weight structure."""

    def __init__(self, hidden_size, num_hidden_layers, chunk_size=25):
        super().__init__()
        self.layers = nn.ModuleList([
            _MambaMia2Block(hidden_size, layer_idx=idx, chunk_size=chunk_size)
            for idx in range(num_hidden_layers)
        ])
        self.norm_f = _MambaMia2RMSNorm(hidden_size)

    def forward(self, inputs_embeds, attention_mask=None):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm_f(hidden_states)
        return _MambaMia2Output(last_hidden_state=hidden_states)


class MambaMiaCompressorWrapper(nn.Module):
    """MambaMia Video/Audio Compressor matching OmniServe weight structure.

    Compresses temporal sequences (audio/video frames) with 25:1 ratio using
    MambaMia2 SSM blocks. Loaded from HF checkpoint weights under
    model.video_audio_compressor.* prefix.

    Weight structure (after HF->vLLM prefix mapping):
        video_audio_compressor.input_proj.{weight,bias}
        video_audio_compressor.query_token
        video_audio_compressor.input_norm.{weight,bias}
        video_audio_compressor.model.layers.N.{mixer,norm}.*
        video_audio_compressor.model.norm_f.weight
        video_audio_compressor.output_proj.{weight,bias}
    """

    def __init__(self, config):
        super().__init__()
        self.chunk_size = getattr(config, "chunk_size", 25)
        self.input_size = getattr(config, "input_size", 1280)
        self.output_size = getattr(config, "output_size", 2048)
        self.hidden_size = getattr(config, "hidden_size", 3072)
        num_hidden_layers = getattr(config, "num_hidden_layers", 4)

        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.query_token = nn.Parameter(torch.randn(self.hidden_size))
        self.input_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(self.hidden_size, self.output_size)

        self.model = self._build_backbone(config, num_hidden_layers)

    def _build_backbone(self, config, num_hidden_layers):
        """Build MambaMia2Model backbone if mamba_ssm is available."""
        try:
            from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401
        except ImportError:
            logger.warning(
                "mamba_ssm not installed. MambaMia compressor weights "
                "will be loaded but forward() will use identity fallback. "
                "Install: pip install mamba-ssm>=1.2.0 causal-conv1d>=1.2.0"
            )
            return None

        try:
            return _MambaMia2Model(
                hidden_size=self.hidden_size,
                num_hidden_layers=num_hidden_layers,
                chunk_size=self.chunk_size,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize MambaMia2 backbone: %s. "
                "Using identity fallback.", e,
            )
            return None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Compress temporal sequence with chunk_size:1 ratio.

        Args:
            inputs_embeds: (B, L, input_size) audio/video frame embeddings

        Returns:
            (B, L//chunk_size, output_size) compressed representations
        """
        bsz, seq_len, _ = inputs_embeds.shape

        # Pad to chunk_size boundary
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size - (seq_len % self.chunk_size)
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len))
            seq_len = inputs_embeds.shape[1]

        n_chunk = seq_len // self.chunk_size

        # Project input to hidden dimension
        hidden_states = self.input_proj(inputs_embeds)

        # Reshape to chunks and insert query tokens
        hidden_4d = hidden_states.view(
            bsz, n_chunk, self.chunk_size, self.hidden_size
        )
        query_expanded = self.query_token.view(1, 1, 1, -1).expand(
            bsz, n_chunk, 1, self.hidden_size
        )
        hidden_with_query = torch.cat([hidden_4d, query_expanded], dim=2)

        # Flatten, normalize
        model_input = hidden_with_query.view(bsz, -1, self.hidden_size)
        model_input = self.input_norm(model_input)

        # Process through MambaMia2 backbone
        if self.model is not None:
            outputs = self.model(inputs_embeds=model_input)
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = model_input

        # Handle NaN defensively
        if torch.isnan(hidden_states).any():
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        # Extract query positions (last token in each chunk)
        hidden_out_4d = hidden_states.view(
            bsz, n_chunk, self.chunk_size + 1, self.hidden_size
        )
        query_outputs = hidden_out_4d[:, :, self.chunk_size, :]

        # Project to output size
        compressed = self.output_proj(query_outputs)
        return compressed


# Type aliases (reuse Qwen2.5-VL types for image/video)
HCXOmniImageInputs: TypeAlias = (
    Qwen2_5_VLImagePixelInputs | Qwen2_5_VLImageEmbeddingInputs
)
HCXOmniVideoInputs: TypeAlias = (
    Qwen2_5_VLVideoPixelInputs | Qwen2_5_VLVideoEmbeddingInputs
)


# === Audio Input Types === #
class HCXOmniAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "nmb", 3000),
    ]
    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", 3000),
    ]


class HCXOmniAudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"] = "audio_embeds"
    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs", dynamic_dims={"naf"}),
    ]


HCXOmniAudioInputs: TypeAlias = (
    HCXOmniAudioFeatureInputs | HCXOmniAudioEmbeddingInputs
)


# === Discrete Token Input Types === #
class HCXOmniDiscreteImageTokenInputs(TensorSchema):
    """Pre-computed discrete image token indices from TA-Tok SimVQ encoder.
    Each tensor contains codebook indices [0, 65535] for one image.
    Mapped to LLM vocab IDs via: token_id = index + discrete_image_unit_0_id.
    """
    type: Literal["discrete_image_tokens"]
    discrete_image_tokens: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "ndt", dynamic_dims={"ndt"}),
    ]


class HCXOmniDiscreteAudioTokenInputs(TensorSchema):
    """Pre-computed discrete audio token indices from CosyVoice2 FSQ encoder.
    Each tensor contains codebook indices [0, 6561] for one audio.
    Mapped to LLM vocab IDs via: token_id = index + discrete_audio_unit_0_id.
    """
    type: Literal["discrete_audio_tokens"]
    discrete_audio_tokens: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "ndt", dynamic_dims={"ndt"}),
    ]


# === Audio MLP Projector === #
class HCXOmniAudioMLP(nn.Module):
    """MLP projector matching the source model's VLM_Mlp for audio."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# From Qwen2AudioEncoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    feat_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (feat_lengths - 2) // 2 + 1
    return feat_lengths, output_lengths


# === Discrete Token Data Items & Parser === #

class _DiscreteTokenItems(ProcessorBatchItems):
    """Data items for discrete tokens (VQ indices or raw PIL images).

    Overrides get_processor_data to use modality name directly (no 's' suffix)
    so _call_hf_processor can pop it with the same key.
    """

    def get_processor_data(self) -> dict[str, object]:
        return {self.modality: self.get_all()}


class _HCXOmniDataParser(MultiModalDataParser):
    """Extends MultiModalDataParser with discrete_image / discrete_audio."""

    def _parse_discrete_data(self, data, modality: str):
        if data is None or self._is_empty(data):
            return None
        if not isinstance(data, (list, tuple)):
            data = [data]
        return _DiscreteTokenItems(data, modality)

    def _get_subparsers(self):
        subparsers = dict(super()._get_subparsers())
        subparsers["discrete_image"] = lambda d: self._parse_discrete_data(
            d, "discrete_image"
        )
        subparsers["discrete_audio"] = lambda d: self._parse_discrete_data(
            d, "discrete_audio"
        )
        return subparsers


# === Processing Info === #
class HCXOmniProcessingInfo(Qwen2VLProcessingInfo):
    """Processing info for HCXOmni (vision + audio)."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        from vllm.transformers_utils.processor import cached_get_processor

        model_id = self.ctx.model_config.model
        revision = self.ctx.model_config.revision
        trust_remote_code = self.ctx.model_config.trust_remote_code
        use_fast = kwargs.pop("use_fast", True)
        return cached_get_processor(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {
            "image": None,
            "audio": None,
            "discrete_image": None,
            "discrete_audio": None,
        }

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_config = self.get_hf_config()
        audio_config = hf_config.audio_config
        return WhisperFeatureExtractor(
            feature_size=audio_config.num_mel_bins,
            sampling_rate=16000,
        )

    def get_data_parser(self):
        return _HCXOmniDataParser(
            target_sr=16000,
            target_channels=1,
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        # Audio: max mel length 3000 -> after 2 convolutions -> 750 tokens
        max_audio_tokens = 750
        # Discrete tokens are embedded by the LLM table (is_embed=all_false),
        # so they need 0 encoder cache budget.
        max_discrete_image_tokens = 0
        max_discrete_audio_tokens = 0
        return {
            "image": max_image_tokens,
            "audio": max_audio_tokens,
            "discrete_image": max_discrete_image_tokens,
            "discrete_audio": max_discrete_audio_tokens,
        }


# === Dummy Inputs Builder === #
class HCXOmniDummyInputsBuilder(
    BaseDummyInputsBuilder[HCXOmniProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_discrete_images = mm_counts.get("discrete_image", 0)
        num_discrete_audios = mm_counts.get("discrete_audio", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token

        # Use <|AUDIO_PAD|> as audio placeholder
        audio_token = "<|AUDIO_PAD|>"
        discrete_image_token = "<|DISCRETE_IMAGE_PAD|>"
        discrete_audio_token = "<|DISCRETE_AUDIO_PAD|>"

        return (
            discrete_image_token * num_discrete_images
            + image_token * num_images
            + discrete_audio_token * num_discrete_audios
            + audio_token * num_audios
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_discrete_images = mm_counts.get("discrete_image", 0)
        num_discrete_audios = mm_counts.get("discrete_audio", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None

        # Dummy audio: 30s at 16kHz
        audio_len = 30 * 16000

        mm_data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }
        if num_audios > 0:
            mm_data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        # Dummy discrete tokens (pre-computed VQ indices)
        if num_discrete_images > 0:
            mm_data["discrete_image"] = [
                torch.zeros(729, dtype=torch.long)
                for _ in range(num_discrete_images)
            ]
        if num_discrete_audios > 0:
            mm_data["discrete_audio"] = [
                torch.zeros(375, dtype=torch.long)
                for _ in range(num_discrete_audios)
            ]
        return mm_data


# === Multimodal Processor === #
class HCXOmniMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    """Extends Qwen2.5-VL processor with audio support."""

    def apply(
        self,
        inputs,
        timing_ctx=None,
    ):
        # The vLLM multimodal processor API changed from a 4-argument form
        # (self, prompt_token_ids, mm_data, hf_processor_mm_kwargs) to a
        # 2-argument form (self, inputs: ProcessorInputs, timing_ctx).
        # This override uses the new signature and injects None placeholders
        # for `discrete_audio` so that get_replacement_discrete_audio() is
        # invoked for each audio item even before the actual discrete tokens
        # are available (they are filled in later by the thinker stage).
        from copy import copy as _copy
        from dataclasses import replace as _dc_replace
        from vllm.multimodal.processing.context import TimingContext

        mm_items = inputs.mm_data_items
        if "audio" in mm_items and "discrete_audio" not in mm_items:
            mm_items = _copy(mm_items)  # shallow copy
            n_audio = mm_items.get_count("audio")
            mm_items["discrete_audio"] = _DiscreteTokenItems(
                [None] * n_audio, "discrete_audio"
            )
            inputs = _dc_replace(inputs, mm_data_items=mm_items)

        if timing_ctx is None:
            timing_ctx = TimingContext(enabled=False)
        return super().apply(inputs, timing_ctx)

    def _get_cache_missing_items(self, cache, mm_data_items, mm_hashes):
        # Override to allow None placeholders for discrete_audio.
        # These are injected by apply() so that _call_hf_processor
        # can run CosyVoice2 inline encoding on the audio items.
        mm_is_cached = {
            modality: cache.is_cached(hashes)
            for modality, hashes in mm_hashes.items()
        }
        mm_missing_idxs = {
            modality: [
                idx for idx, item_is_cached in enumerate(items_is_cached)
                if not item_is_cached
            ]
            for modality, items_is_cached in mm_is_cached.items()
        }
        mm_missing_data: dict = {}
        for modality, idxs in mm_missing_idxs.items():
            missing_modality_data = []
            for idx in idxs:
                data = mm_data_items[modality][idx]
                if data is None:
                    if modality == "discrete_audio":
                        # None placeholder injected by apply(); CosyVoice2
                        # inline encoding will fire in _call_hf_processor
                        missing_modality_data.append(None)
                    else:
                        raise ValueError(
                            f"Cache miss for {modality} at index {idx} "
                            f"but data is not provided."
                        )
                else:
                    missing_modality_data.append(data)
            mm_missing_data[modality] = missing_modality_data
        mm_missing_items = self.info.parse_mm_data(mm_missing_data, validate=False)
        return mm_is_cached, mm_missing_items

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Pop audio and discrete data before calling parent
        # (Qwen2.5-VL doesn't know audio or discrete tokens)
        mm_data = dict(mm_data)  # make mutable copy
        audios = mm_data.pop("audios", [])
        discrete_images = mm_data.pop("discrete_image", [])
        discrete_audios = mm_data.pop("discrete_audio", [])

        # Fix token case mismatch: HCX processor uses lowercase <|image_pad|>
        # but tokenizer vocab only has uppercase <|IMAGE_PAD|> (ID 128062).
        # Patch processor to use uppercase tokens so _check_special_mm_tokens
        # can correctly match tokens in text vs input_ids.
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        _orig_image_token = hf_processor.image_token
        _orig_video_token = hf_processor.video_token
        _orig_image_token_id = getattr(hf_processor, 'image_token_id', None)

        if "<|IMAGE_PAD|>" in vocab:
            hf_processor.image_token = "<|IMAGE_PAD|>"
            hf_processor.image_token_id = vocab["<|IMAGE_PAD|>"]
            prompt = prompt.replace(_orig_image_token, "<|IMAGE_PAD|>")
        if "<|VIDEO_PAD|>" in vocab:
            hf_processor.video_token = "<|VIDEO_PAD|>"
            prompt = prompt.replace(_orig_video_token, "<|VIDEO_PAD|>")

        try:
            # Call parent for image/video processing
            result = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
        finally:
            # Restore original tokens
            hf_processor.image_token = _orig_image_token
            hf_processor.video_token = _orig_video_token
            if _orig_image_token_id is not None:
                hf_processor.image_token_id = _orig_image_token_id

        # Process audio separately with WhisperFeatureExtractor
        if audios:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            audio_inputs = feature_extractor(
                audios,
                sampling_rate=feature_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )
            result["input_features"] = audio_inputs["input_features"]
            result["feature_attention_mask"] = audio_inputs["attention_mask"]

        # Inline CosyVoice2 discrete audio encoding:
        # Triggered when:
        #   (a) discrete_audios is empty — text-only request with audio (shouldn't happen)
        #   (b) discrete_audios contains None placeholders injected by apply()
#        (c) NOT triggered when discrete_audios has pre-encoded tensors
        _needs_cosyvoice = audios and (
            not discrete_audios
            or all(not isinstance(da, torch.Tensor) for da in discrete_audios)
        )
        if _needs_cosyvoice:
            model_dir = self.info.ctx.model_config.model
            try:
                discrete_audios = _CosyvoiceEncoderManager.encode_audios(
                    audios, model_dir
                )
            except Exception as _e:
                logger.warning(
                    "CosyVoice2 inline encoding failed (discrete audio skipped): %s",
                    _e,
                )
                discrete_audios = []

        # Handle discrete images: pre-computed tensors or raw PIL images
        if discrete_images:
            encoded = []
            raw_pils: list[tuple[int, Image.Image]] = []
            for item in discrete_images:
                if isinstance(item, torch.Tensor):
                    encoded.append(item)
                else:
                    # PIL Image — needs TA-Tok encoding
                    raw_pils.append((len(encoded), item))
                    encoded.append(None)  # placeholder

            if raw_pils:
                hf_config = self.info.get_hf_config()
                dv_config = getattr(
                    hf_config, "discrete_vision_config", None
                )
                if dv_config is None or not isinstance(dv_config, dict):
                    raise ValueError(
                        "discrete_vision_config not set in model config. "
                        "Cannot encode raw images without TA-Tok."
                    )
                ta_tok_path = dv_config.get("model_name_or_path")
                if ta_tok_path is None:
                    raise ValueError(
                        "discrete_vision_config.model_name_or_path not set."
                    )
                pil_images = [img for _, img in raw_pils]
                indices_list = _TaTokEncoderManager.encode_images(
                    pil_images, ta_tok_path
                )
                for (idx, _), indices in zip(raw_pils, indices_list):
                    encoded[idx] = indices

            result["discrete_image_tokens"] = encoded
        if discrete_audios:
            result["discrete_audio_tokens"] = discrete_audios

        return result

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(
            super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        )
        # Add audio fields
        fields["input_features"] = MultiModalFieldConfig.batched("audio")
        fields["feature_attention_mask"] = MultiModalFieldConfig.batched("audio")
        # Add discrete token fields
        fields["discrete_image_tokens"] = MultiModalFieldConfig.batched(
            "discrete_image"
        )
        fields["discrete_audio_tokens"] = MultiModalFieldConfig.batched(
            "discrete_audio"
        )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # Build all prompt updates using uppercase token names from vocab
        # (HCX processor returns lowercase image_token but vocab has uppercase)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use uppercase tokens from vocab (HCX uses <|IMAGE_PAD|> not <|image_pad|>)
        image_token_id = vocab.get("<|IMAGE_PAD|>", vocab.get(hf_processor.image_token))
        video_token_id = vocab.get("<|VIDEO_PAD|>", vocab.get(hf_processor.video_token))
        merge_length = image_processor.merge_size ** 2

        def get_replacement_vision(item_idx: int, modality: str):
            token_id = image_token_id if modality == "image" else video_token_id
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            return [token_id] * num_tokens

        updates: list[PromptUpdate] = [
            PromptReplacement(
                modality=modality,
                target=[image_token_id if modality == "image" else video_token_id],
                replacement=partial(get_replacement_vision, modality=modality),
            )
            for modality in ("image", "video")
        ]

        # Audio prompt update
        audio_token = "<|AUDIO_PAD|>"
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None:
            return updates

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths: list[int] = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_audio(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                raise ValueError("The audio is too short to be represented")

            audio_tokens = [audio_token_id] * num_features
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        updates.append(
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_audio,
            )
        )

        # --- Discrete image prompt update --- #
        hf_config = self.info.get_hf_config()
        discrete_image_pad_token = "<|DISCRETE_IMAGE_PAD|>"
        discrete_image_pad_id = vocab.get(discrete_image_pad_token)
        discrete_image_unit_0_id = getattr(
            hf_config, "discrete_image_unit_0_id", 135168
        )
        # Structural tokens matching training format
        vision_eol_id = vocab.get("<|vision_eol|>")
        vision_eof_id = vocab.get("<|vision_eof|>")
        vision_ratio_1_1_id = vocab.get("<|vision_ratio_1:1|>")

        if discrete_image_pad_id is not None:
            def get_replacement_discrete_image(item_idx: int):
                out_item = out_mm_kwargs["discrete_image"][item_idx]
                tokens = out_item["discrete_image_tokens"].data
                assert isinstance(tokens, torch.Tensor)
                flat = tokens.flatten().tolist()

                # Build token sequence matching training format:
                # [ratio] + ([codebook_token]*27 + [eol])*27 + [eof]
                result_ids: list[int] = []
                if vision_ratio_1_1_id is not None:
                    result_ids.append(vision_ratio_1_1_id)

                for i, t in enumerate(flat):
                    result_ids.append(discrete_image_unit_0_id + int(t))
                    if (i + 1) % 27 == 0 and vision_eol_id is not None:
                        result_ids.append(vision_eol_id)

                if vision_eof_id is not None:
                    result_ids.append(vision_eof_id)

                # Discrete tokens are embedded by the LLM table directly;
                # no multimodal encoder embeddings needed (is_embed=all False).
                return PromptUpdateDetails(
                    full=result_ids,
                    is_embed=lambda _tok, full: torch.zeros(
                        len(full), dtype=torch.bool
                    ),
                )

            updates.append(
                PromptReplacement(
                    modality="discrete_image",
                    target=[discrete_image_pad_id],
                    replacement=get_replacement_discrete_image,
                )
            )

        # --- Discrete audio prompt update --- #
        discrete_audio_pad_token = "<|DISCRETE_AUDIO_PAD|>"
        discrete_audio_pad_id = vocab.get(discrete_audio_pad_token)
        discrete_audio_unit_0_id = getattr(
            hf_config, "discrete_audio_unit_0_id", 128606
        )

        if discrete_audio_pad_id is not None:
            def get_replacement_discrete_audio(item_idx: int):
                out_item = out_mm_kwargs["discrete_audio"][item_idx]
                tokens = out_item["discrete_audio_tokens"].data
                assert isinstance(tokens, torch.Tensor)
                # Map codebook indices [0, 6561] to LLM vocab token IDs
                result_ids = [
                    discrete_audio_unit_0_id + int(t)
                    for t in tokens.flatten()
                ]
                return PromptUpdateDetails(
                    full=result_ids,
                    is_embed=lambda _tok, full: torch.zeros(
                        len(full), dtype=torch.bool
                    ),
                )

            updates.append(
                PromptReplacement(
                    modality="discrete_audio",
                    target=[discrete_audio_pad_id],
                    replacement=get_replacement_discrete_audio,
                )
            )

        return updates


# === Model === #
@MULTIMODAL_REGISTRY.register_processor(
    HCXOmniMultiModalProcessor,
    info=HCXOmniProcessingInfo,
    dummy_inputs=HCXOmniDummyInputsBuilder,
)
class HCXOmniForCausalLM(
    nn.Module,
    SupportsMultiModal,
    SupportsMRoPE,
    SupportsPP,
    SupportsQuant,
):
    """
    HCXOmni model: Qwen2.5-VL vision encoder + Qwen2AudioEncoder +
    projectors + HyperCLOVAX (Llama-style) language model.

    Supports image-to-text and audio-to-text inference.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv": ["qkv"],  # vision tower uses pre-packed qkv
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_model.": "visual.",
            "model.mm_projector.": "mm_projector.",
            "model.audio_model.": "audio_tower.",
            "model.audio_projector.": "audio_projector.",
            "model.video_audio_compressor.": "video_audio_compressor.",
            "model.language_model.": "language_model.",
        }
    )

    # --- MRoPE helpers --- #

    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int, int, float]]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        tokens_per_second = getattr(
            self.config.vision_config, "tokens_per_second", 1.0
        )
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                yield (
                    offset,
                    1,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                    1.0,
                )
            elif mm_feature.modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                second_per_grid_ts = 1.0
                if mm_feature.data.get("second_per_grid_ts", None):
                    second_per_grid_ts = mm_feature.data[
                        "second_per_grid_ts"
                    ].data.item()
                t_factor = second_per_grid_ts * tokens_per_second
                yield (
                    offset,
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                    t_factor,
                )
            elif mm_feature.modality in (
                "audio", "discrete_image", "discrete_audio",
            ):
                # Audio and discrete tokens use flat text positions (no 3D grid)
                continue
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        llm_pos_ids_list: list = []
        st = 0

        for (
            offset,
            llm_grid_t,
            llm_grid_h,
            llm_grid_w,
            t_factor,
        ) in self.iter_mm_grid_thw(mm_features):
            text_len = offset - st
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0
                else 0
            )
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

            grid_indices = np.indices((llm_grid_t, llm_grid_h, llm_grid_w))
            if t_factor != 1.0:
                grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
            llm_pos_ids_list.append(
                grid_indices.reshape(3, -1) + text_len + st_idx
            )
            st = offset + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0
                else 0
            )
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (
            llm_positions.max() + 1 - len(input_tokens)
        ).item()

        return torch.from_numpy(llm_positions), mrope_position_delta

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "discrete_image":
            return "<|DISCRETE_IMAGE_PAD|>"
        if modality == "discrete_audio":
            return "<|DISCRETE_AUDIO_PAD|>"
        if modality.startswith("image"):
            return "<|IMAGE_PAD|>"
        if modality.startswith("video"):
            return "<|VIDEO_PAD|>"
        if modality.startswith("audio"):
            return "<|AUDIO_PAD|>"
        raise ValueError(f"Unsupported modality: {modality}")

    # --- Init --- #

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        # Vision encoder + projector
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

            vision_out_size = getattr(
                config.vision_config,
                "out_hidden_size",
                config.vision_config.hidden_size,
            )
            text_hidden_size = config.text_config.hidden_size
            self.mm_projector = nn.Linear(
                vision_out_size, text_hidden_size, bias=True
            )

        # Audio encoder + MLP projector
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen2AudioEncoder(config.audio_config)
            audio_d_model = config.audio_config.d_model  # 1280
            self.audio_projector = HCXOmniAudioMLP(
                in_features=audio_d_model,
                hidden_features=audio_d_model,
                out_features=text_hidden_size,
            )

        # MambaMia Video/Audio Compressor (optional, 25:1 temporal compression)
        compressor_config = getattr(config, "video_audio_compressor_config", None)
        if compressor_config is not None:
            self.video_audio_compressor = MambaMiaCompressorWrapper(
                compressor_config
            )
        else:
            self.video_audio_compressor = None

        # Language model
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # --- Image/Video processing --- #

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> HCXOmniImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        return Qwen2_5_VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> HCXOmniVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        return Qwen2_5_VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _process_image_input(
        self, image_input: HCXOmniImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_forward_context(None, self.vllm_config):
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        image_embeds = self.mm_projector(
            image_embeds.to(self.mm_projector.weight.dtype)
        )

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: HCXOmniVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            with set_forward_context(None, self.vllm_config):
                video_embeds = self.visual(
                    pixel_values_videos, grid_thw=grid_thw_list
                )

        video_embeds = self.mm_projector(
            video_embeds.to(self.mm_projector.weight.dtype)
        )

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    # --- Audio processing (adapted from qwen2_audio.py) --- #

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> HCXOmniAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            return HCXOmniAudioEmbeddingInputs(
                type="audio_embeds", audio_embeds=audio_embeds
            )

        if input_features is not None:
            return HCXOmniAudioFeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: HCXOmniAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return tuple(audio_input["audio_embeds"])

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
        )

        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=audio_feat_lengths.dtype,
                device=audio_feat_lengths.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(
            batch_size, max_seq_len
        )
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len
        ).expand(batch_size, 1, max_seq_len, max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(
            input_features, attention_mask=audio_attention_mask
        )
        selected_audio_feature = audio_outputs.last_hidden_state

        # Apply MLP projector
        audio_features = self.audio_projector(selected_audio_feature)

        # Mask out padding tokens
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_output_lengths = audio_output_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens)
            .expand(num_audios, max_audio_tokens)
            .to(audio_output_lengths.device)
            < audio_output_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(
            -1, embed_dim
        )

        return torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )

    # --- Combined multimodal dispatch --- #

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = (
                    self._parse_and_validate_image_input(**kwargs)
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = (
                    self._parse_and_validate_video_input(**kwargs)
                )
            if (
                input_key in ("input_features", "audio_embeds")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = (
                    self._parse_and_validate_audio_input(**kwargs)
                )
            # Discrete tokens: no real encoding needed, just track count
            if (
                input_key == "discrete_image_tokens"
                and "discrete_image" not in mm_input_by_modality
            ):
                mm_input_by_modality["discrete_image"] = kwargs[
                    "discrete_image_tokens"
                ]
            if (
                input_key == "discrete_audio_tokens"
                and "discrete_audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["discrete_audio"] = kwargs[
                    "discrete_audio_tokens"
                ]
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs
        )
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if multimodal_input is None:
                continue
            if modality == "image":
                multimodal_embeddings += tuple(
                    self._process_image_input(multimodal_input)
                )
            elif modality == "video":
                multimodal_embeddings += tuple(
                    self._process_video_input(multimodal_input)
                )
            elif modality == "audio":
                multimodal_embeddings += tuple(
                    self._process_audio_input(multimodal_input)
                )
            elif modality in ("discrete_image", "discrete_audio"):
                # Discrete tokens are embedded by the LLM's embedding table.
                # Return dummy tensors to satisfy the encoder pipeline check.
                # These are never used because is_embed=all_false in
                # PromptUpdateDetails.
                if isinstance(multimodal_input, (list, tuple)):
                    n_items = len(multimodal_input)
                else:
                    n_items = 1
                _p = next(self.visual.parameters())
                for _ in range(n_items):
                    multimodal_embeddings += (
                        torch.empty(0, 0, device=_p.device, dtype=_p.dtype),
                    )

        return multimodal_embeddings

    # --- Forward / logits / weights --- #

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        skip_prefixes = [
            # Discrete encoders run externally (OmniServe encoder service),
            # not inside this vLLM model.
            "model.discrete_audio_encoder.",
            "model.discrete_vision_encoder.",
            "model.discrete_audio_model.",
            "model.discrete_vision_model.",
        ]
        # Skip compressor weights if module is not loaded
        if self.video_audio_compressor is None:
            skip_prefixes.append("model.video_audio_compressor.")
            skip_prefixes.append("video_audio_compressor.")

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        connector = ["mm_projector.", "audio_projector."]
        tower_model = ["visual.", "audio_tower."]
        if self.video_audio_compressor is not None:
            connector.append("video_audio_compressor.")
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=connector,
            tower_model=tower_model,
        )
