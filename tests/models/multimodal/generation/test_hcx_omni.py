"""
HCXOmni vLLM porting test script (TDD - tests first).
Tests the omni model with image-to-text and audio-to-text.

Usage (inside kje-vllm-dev-build container):
  cd /workspace
  python /mnt/local/kje/20260227/test_hcx_omni.py
"""

import sys
import os

# Remove script directory from sys.path to avoid shadowing vllm package
# with the vllm/ repo directory in the same folder.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

import numpy as np

MODEL_PATH = "/mnt/local/kje/omni-model-encoder-llm"

results = {"image": None, "audio": None}

print("=" * 60)
print("HCXOmni vLLM Port Test (TDD)")
print("=" * 60)

# ------------------------------------------------------------------ #
# Step 1: Check imports
# ------------------------------------------------------------------ #
print("\n[Step 1] Checking imports...")
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.hcx_omni import HCXOmniForCausalLM
    print("  OK: vLLM and HCXOmniForCausalLM imported")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------ #
# Step 2: Check registry
# ------------------------------------------------------------------ #
print("\n[Step 2] Checking model registry...")
try:
    from vllm.model_executor.models.registry import (
        _MULTIMODAL_MODELS,
        _TEXT_GENERATION_MODELS,
    )

    assert "HCXOmniForCausalLM" in _MULTIMODAL_MODELS, \
        "HCXOmniForCausalLM not in _MULTIMODAL_MODELS"

    print(f"  OK: HCXOmniForCausalLM -> {_MULTIMODAL_MODELS['HCXOmniForCausalLM']}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------ #
# Step 3: Check processor loading
# ------------------------------------------------------------------ #
print("\n[Step 3] Checking processor loading...")
try:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  OK: Processor type = {type(processor).__name__}")
    print(f"  OK: image_token = {processor.image_token!r}")
    print(f"  OK: video_token = {processor.video_token!r}")
    print(f"  OK: merge_size = {processor.image_processor.merge_size}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------ #
# Step 4: Load model with vLLM
# ------------------------------------------------------------------ #
print("\n[Step 4] Loading model with vLLM (this may take a few minutes)...")
try:
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1, "audio": 1},
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )
    print("  OK: Model loaded successfully!")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------------ #
# Step 5: Image-to-Text inference
# ------------------------------------------------------------------ #
print("\n[Step 5] Running image-to-text inference...")
try:
    from PIL import Image
    from io import BytesIO

    # Load sample image (URL with synthetic fallback)
    try:
        import urllib.request
        img_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"
        with urllib.request.urlopen(img_url, timeout=5) as resp:
            img_data = resp.read()
        image = Image.open(BytesIO(img_data)).convert("RGB")
        print(f"  OK: Loaded image from URL, size = {image.size}")
    except Exception:
        image = Image.new("RGB", (448, 448), color=(128, 64, 32))
        print(f"  INFO: Using synthetic image, size = {image.size}")

    # Resize if too large (keep under ~4M pixels for 8192 context)
    MAX_PIXELS = 4_000_000
    w, h = image.size
    if w * h > MAX_PIXELS:
        ratio = (MAX_PIXELS / (w * h)) ** 0.5
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        print(f"  INFO: Resized large image {(w, h)} -> {image.size}")

    image_token = "<|IMAGE_PAD|>"
    prompt = (
        f"<|im_start|>user\n{image_token}\n"
        f"이 이미지에 무엇이 보이나요?<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        sampling_params=sampling_params,
    )

    response = outputs[0].outputs[0].text
    assert len(response.strip()) > 0, "Image response is empty"

    print(f"  OK: Image response:\n{'-'*40}")
    print(f"  {response.strip()}")
    print(f"  {'-'*40}")
    results["image"] = "PASS"

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    results["image"] = "FAIL"

# ------------------------------------------------------------------ #
# Step 6: Audio-to-Text inference
# ------------------------------------------------------------------ #
print("\n[Step 6] Running audio-to-text inference...")
# Load real audio from URL
audio_url = "http://wbl.s3-website.kr.object.ncloudstorage.com/wbl/example/example.wav"
import urllib.request, io, soundfile as sf
with urllib.request.urlopen(audio_url, timeout=10) as resp:
    audio_bytes = resp.read()
audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
if sample_rate != 16000:
    import scipy.signal
    num_samples = int(len(audio) * 16000 / sample_rate)
    audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    sample_rate = 16000
duration = len(audio) / sample_rate
print(f"  OK: Loaded audio from URL: {audio.shape[0]} samples, {sample_rate}Hz, {duration:.1f}s")

audio_token = "<|AUDIO_PAD|>"
prompt = (
    f"<|im_start|>user\n{audio_token}\n"
    f"이 오디오에서 무엇이 들리나요?<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"audio": audio}},
    sampling_params=sampling_params,
)

response = outputs[0].outputs[0].text
assert len(response.strip()) > 0, "Audio response is empty"

print(f"  OK: Audio response:\n{'-'*40}")
print(f"  {response.strip()}")
print(f"  {'-'*40}")
results["audio"] = "PASS"



# ------------------------------------------------------------------ #
# Step 7: Summary
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("Test Results Summary")
print("=" * 60)
print(f"  Image-to-Text : {results['image']}")
print(f"  Audio-to-Text : {results['audio']}")
print("-" * 60)

if all(v == "PASS" for v in results.values()):
    print("[SUCCESS] All HCXOmni tests passed!")
else:
    failed = [k for k, v in results.items() if v != "PASS"]
    print(f"[PARTIAL] Failed tests: {', '.join(failed)}")
    sys.exit(1)
