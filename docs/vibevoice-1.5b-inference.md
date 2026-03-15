# VibeVoice 1.5B Inference API

This document covers the Python API for `VibeVoiceForConditionalGenerationInference`, the inference class for the VibeVoice-1.5B TTS model restored in [davidamacey/VibeVoice](https://github.com/davidamacey/VibeVoice).

For a higher-level overview see [vibevoice-tts.md](vibevoice-tts.md). For LoRA fine-tuning see [finetuning-tts/README.md](../finetuning-tts/README.md).

---

## Installation

```bash
git clone https://github.com/davidamacey/VibeVoice
cd VibeVoice
pip install -e .

huggingface-cli download microsoft/VibeVoice-1.5B --local-dir ./models/VibeVoice-1.5B
```

---

## Loading the Model

### Standard load (recommended)

```python
import torch
from vibevoice import VibeVoiceForConditionalGenerationInference
from vibevoice.processor import VibeVoiceProcessor

model_path = "./models/VibeVoice-1.5B"

processor = VibeVoiceProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
    model_path,
    device="cuda",
    torch_dtype=torch.bfloat16,
)
model.eval()
```

`from_pretrained_hf` wraps the standard HuggingFace `from_pretrained` and reads the sharded `model.safetensors.index.json` automatically.

### Load with CPU layer offloading (low VRAM)

For GPUs with limited VRAM, keep only N transformer layers on the GPU and move the rest to CPU. Layers are prefetched asynchronously during the forward pass.

```python
from vibevoice import VibeVoiceForConditionalGenerationInference, OffloadConfig
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

config = VibeVoiceConfig.from_pretrained(model_path)
offload_config = OffloadConfig(
    enabled=True,
    num_layers_on_gpu=12,       # keep 12 of 28 layers on GPU (~10 GB VRAM)
    offload_prediction_head=False,
    pin_memory=True,            # required for async transfers
    prefetch_next_layer=True,
    async_transfer=True,
)

model = VibeVoiceForConditionalGenerationInference.from_pretrained_file(
    model_path,
    config=config,
    device="cuda",
    offload_config=offload_config,
)
model.eval()
```

### Load with merged LoRA weights

```python
model = VibeVoiceForConditionalGenerationInference.from_pretrained_file(
    model_path,
    config=config,
    device="cuda",
    lora_model_path="./checkpoints/tts_lora/tts_lora_final.safetensors",
    lora_weight=1.0,
)
```

---

## Text Format

The processor requires text formatted as speaker-labelled lines:

```
Speaker 0: Hello, welcome to the show.
Speaker 1: Thanks for having me, it's great to be here.
Speaker 0: Let's dive right in.
```

- Speaker IDs are integers starting at 0
- Up to 4 speakers are supported
- Single-speaker scripts use `Speaker 0:` throughout
- Punctuation controls pacing — commas and periods work best

---

## Basic TTS

```python
model.set_ddpm_inference_steps(20)  # 10–30; more steps = higher quality, slower

inputs = processor(
    text="Speaker 0: Hello, this is VibeVoice text to speech.",
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        tokenizer=processor.tokenizer,
        cfg_scale=3.0,       # classifier-free guidance scale; 2.0–5.0
        return_speech=True,
    )

import torchaudio
audio = output.speech_outputs[0].cpu()
torchaudio.save("output.wav", audio.unsqueeze(0) if audio.ndim == 1 else audio, 24000)
```

---

## Voice Cloning

Voice cloning is zero-shot. Pass a reference audio via `voice_samples` in the processor call and the model will match that speaker's vocal characteristics — no training required.

```python
import numpy as np
import torchaudio

# Load and prepare reference audio (must be 24 kHz mono)
waveform, sr = torchaudio.load("reference.wav")
if sr != 24000:
    waveform = torchaudio.functional.resample(waveform, sr, 24000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(0, keepdim=True)
voice_samples = [waveform.squeeze(0).numpy()]

inputs = processor(
    text="Speaker 0: This speech will match the reference voice.",
    voice_samples=voice_samples,
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        tokenizer=processor.tokenizer,
        cfg_scale=3.0,
        return_speech=True,
    )

audio = output.speech_outputs[0].cpu()
torchaudio.save("cloned.wav", audio.unsqueeze(0) if audio.ndim == 1 else audio, 24000)
```

Tips for best voice cloning quality:
- Use 5–30 seconds of clean reference audio (no background noise or music)
- Match the language of the reference to the target text
- Try `cfg_scale` between 2.0 and 4.0; higher values enforce the voice more strongly

---

## Multi-speaker Generation

```python
script = """Speaker 0: Welcome back to the podcast.
Speaker 1: Great to be here, thanks for having me.
Speaker 0: Today we're talking about voice AI.
Speaker 1: It's a fascinating topic with a lot of recent progress."""

inputs = processor(text=script, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, tokenizer=processor.tokenizer, cfg_scale=3.0)
```

---

## `generate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | `VibeVoiceTokenizerProcessor` | required | The processor's tokenizer |
| `cfg_scale` | `float` | `3.0` | Classifier-free guidance scale. Higher = more faithful to the text/voice prompt |
| `return_speech` | `bool` | `True` | Whether to decode and return waveform audio |
| `ddpm_steps` | `int` | `20` | Diffusion steps (set once via `model.set_ddpm_inference_steps()`) |
| `visitor` | `GenerationVisitor` | `None` | Optional callback for progress tracking |
| `max_length` | `int` | from config | Maximum token sequence length |

---

## `VibeVoiceGenerationOutput`

The object returned by `generate()`:

```python
@dataclass
class VibeVoiceGenerationOutput:
    sequences: Optional[torch.LongTensor]      # generated token IDs
    speech_outputs: Optional[List[torch.Tensor]]  # decoded audio waveforms (one per batch item)
```

---

## `OffloadConfig` Reference

```python
@dataclass
class OffloadConfig:
    enabled: bool = False
    num_layers_on_gpu: int = 8        # transformer layers to keep on GPU
    offload_prediction_head: bool = False  # also offload diffusion head (~3 GB)
    offload_kv_cache: bool = False    # offload KV cache for CPU layers
    pin_memory: bool = True           # required for async transfers
    prefetch_next_layer: bool = True  # overlap compute and transfer
    async_transfer: bool = True       # ThreadPoolExecutor-based async transfer
    cache_clear_interval: int = 50    # clear CUDA cache every N layer transfers
    verbose: bool = False
    profile: bool = False             # print timing breakdown
```

VRAM usage guide for the 1.5B model (bfloat16):

| `num_layers_on_gpu` | Approx VRAM |
|---------------------|-------------|
| 28 (all, no offload) | ~5 GB |
| 16 | ~3.5 GB |
| 8 | ~2 GB |
| 4 | ~1.5 GB |

---

## Progress Callbacks (`GenerationVisitor`)

Implement `GenerationVisitor` to receive step-level events during generation:

```python
from vibevoice import GenerationVisitor

class MyVisitor(GenerationVisitor):
    def visit_preprocessing(self, *a, **kw): pass
    def visit_inference_start(self, *a, **kw):
        print("Generation started")
    def visit_inference_batch_start(self, *a, **kw): pass
    def visit_inference_batch_end(self, *a, **kw): pass
    def visit_inference_save_audio_file(self, *a, **kw): pass
    def visit_inference_step_start(self, step, total, *a, **kw):
        print(f"Step {step}/{total}")
    def visit_inference_step_end(self, *a, **kw): pass
    def visit_completed(self, *a, **kw):
        print("Done")
    def visit_failed(self, error, *a, **kw):
        print(f"Failed: {error}")

output = model.generate(**inputs, tokenizer=processor.tokenizer, visitor=MyVisitor())
```

---

## Merging LoRA Weights

To bake a trained LoRA into the model weights for zero-overhead inference:

```python
from vibevoice.utils.model_utils import merge_lora_weights

model = merge_lora_weights(
    model,
    lora_path="./checkpoints/tts_lora/tts_lora_final.safetensors",
    lora_weight=1.0,
)
```

---

## CLI Demo

```bash
# Basic TTS
python demo/tts_1p5b_inference.py \
    --model ./models/VibeVoice-1.5B \
    --text "Speaker 0: Hello world." \
    --output output.wav

# Voice cloning
python demo/tts_1p5b_inference.py \
    --model ./models/VibeVoice-1.5B \
    --text "Speaker 0: Hello world." \
    --voice reference.wav \
    --output cloned.wav \
    --cfg-scale 3.0 \
    --steps 20

# CPU layer offloading
python demo/tts_1p5b_inference.py \
    --model ./models/VibeVoice-1.5B \
    --text "Speaker 0: Hello world." \
    --output output.wav \
    --offload --gpu-layers 8
```

---

## Running Tests

```bash
# Unit tests only (no model required)
pytest tests/test_utils.py tests/test_inference.py -k "not Integration"

# All tests including end-to-end generation (requires model at /mnt/nas/models/vibevoice/VibeVoice-1.5B)
pytest tests/ -v
```
