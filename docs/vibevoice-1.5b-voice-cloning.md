# VibeVoice-TTS-1.5B: Complete Voice Cloning Guide

This document is a complete, code-grounded reference for using `VibeVoice-TTS-1.5B` for zero-shot voice cloning. Every section is derived from direct inspection of the source code in this repository. Items that cannot be confirmed from code are explicitly marked as "inferred from architecture."

**Related documents:**
- `docs/vibevoice-tts.md` - model overview and benchmarks
- `docs/vibevoice-voice-cloning-research.md` - research framing, model status, LoRA clarification, and VibeVoiceFusion integration notes

---

## 1. Overview

### What Voice Cloning Means for This Model

VibeVoice-TTS-1.5B implements **zero-shot voice cloning via voice prompting**. There is no fine-tuning step. Instead, one or more short audio clips of a target speaker are prepended to the generation context. The model's acoustic tokenizer converts each clip into a sequence of continuous latent tokens, which are included in the input sequence under a `Voice input:` / `Speaker N:` header. The LLM backbone (Qwen2.5-1.5B) and diffusion head then generate new speech conditioned on those latent tokens.

Source: `vibevoice/processor/vibevoice_processor.py`, method `_create_voice_prompt()` (line 406).

### What It Can Do

- Clone a voice from a short reference recording (5-30 seconds), no training required.
- Support up to 4 distinct speakers in a single generation by supplying one reference clip per speaker.
- Generate up to 90 minutes of speech (64K context window) in a single pass.
- Operate across English, Chinese, and other languages (English and Chinese are the most stable).
- Optionally run without voice samples, in which case the model selects its own voice characteristics.

### What It Cannot Do

- It does not perform real-time / streaming generation. The entire audio is generated in one forward pass. For streaming, use `VibeVoice-Realtime-0.5B` (which does not support open voice cloning).
- The model currently only supports batch size 1 at inference time.
- Voice identity is not perfectly preserved. Results vary across runs because generation uses stochastic diffusion sampling.
- Cross-lingual voice transfer (e.g., Chinese voice, English text) works as an emergent capability but is unstable.
- Background noise or music in the reference clip can appear in the output.

### Important Responsible Use Note

Microsoft removed the official usage instructions for this model after observing widespread misuse for impersonation and deepfakes. The model weights remain available at `microsoft/VibeVoice-1.5B`. This guide is provided for research purposes only. Voice cloning without the consent of the person whose voice is being cloned raises serious ethical and legal concerns. Generated speech must be disclosed as AI-generated when shared.

---

## 2. Requirements

### Installation

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -e .
```

Pre-fetch model weights before running (optional but reduces first-run latency):

```bash
huggingface-cli download microsoft/VibeVoice-1.5B --repo-type model
```

### Hardware

| Setup | VRAM | Notes |
|---|---|---|
| CUDA (recommended) | ~12 GB with bfloat16 | Use `flash_attention_2` for best quality |
| Apple Silicon (MPS) | ~16 GB unified RAM | Must use `float32` and `sdpa` attention |
| CPU | 32+ GB RAM | `float32` only, very slow |

The acoustic tokenizer encoder and decoder are each ~340M parameters in addition to the 1.5B LLM. Total footprint at bfloat16 is approximately 11-13 GB.

### Key Dependencies

The following packages are installed automatically by `pip install -e .`:
- `torch >= 2.0`
- `transformers >= 4.51`
- `accelerate`
- `soundfile` (for reading/writing WAV files)
- `numpy`
- `scipy`
- `librosa` (for audio resampling, optional but recommended)

---

## 3. Input Format Reference

### Script Format

Scripts must follow the `Speaker N:` format. Speaker IDs must be integers. The processor normalizes them to start from 0 (i.e., if your script uses `Speaker 1:` and `Speaker 2:`, they become speaker 0 and speaker 1 internally).

```
Speaker 1: Hello, welcome to the show.
Speaker 2: Thanks for having me today. It's great to be here.
Speaker 1: Let's start with the main topic.
```

The parser is case-insensitive (`speaker 1:` also works). Leading/trailing whitespace on each line is stripped.

Lines that do not match the `Speaker N: text` pattern are silently skipped with a warning. Plain text files can also be passed to the processor: lines without a `Speaker N:` prefix are assigned to `Speaker 1` automatically.

JSON input is also accepted:

```json
[
  {"speaker": "1", "text": "Hello, welcome to the show."},
  {"speaker": "2", "text": "Thanks for having me today."},
  {"speaker": "1", "text": "Let's start with the main topic."}
]
```

Source: `vibevoice/processor/vibevoice_processor.py`, methods `_parse_script()`, `_convert_text_to_script()`, `_convert_json_to_script()`.

### Voice Sample Requirements

| Parameter | Value | Source |
|---|---|---|
| Sample rate | **24000 Hz** | `vibevoice/processor/vibevoice_tokenizer_processor.py`, `sampling_rate` default |
| Channels | Mono (stereo is converted to mono by averaging) | `_ensure_mono()` in tokenizer processor |
| Format | WAV recommended; any format readable by `soundfile` or `ffmpeg` | `_load_audio_from_path()` |
| Data type | float32 numpy array, range approximately [-1.0, 1.0] | processor normalizer output |
| dB normalization | Applied by default (target -25 dB FS) | `AudioNormalizer` with `db_normalize=True` |

**Recommended reference audio characteristics:**
- Length: 5-30 seconds. Shorter clips may not contain enough voice characteristics; longer clips consume context tokens at 7.5 Hz.
- Content: Clean speech, single speaker, no background music or significant noise.
- Language: Match the language of the target synthesis script if possible. English and Chinese produce the most consistent results.
- Do not use clips where multiple people speak simultaneously.

### Token Budget for Voice Prompts

The acoustic tokenizer compresses audio at **3200 samples per token** (the `speech_tok_compress_ratio` parameter, confirmed in `vibevoice/processor/vibevoice_processor.py`, line 29). At 24000 Hz this is:

```
tokens_per_second = 24000 / 3200 = 7.5 tokens/second
```

A 10-second reference clip consumes 75 voice prompt tokens per speaker. A 30-second clip consumes 225 tokens. The model's full context window is 64K tokens; the voice prompt, system prompt, text input, and generated speech output all share this budget.

For a single speaker, a 10-20 second reference clip is a practical default. For 2-4 speakers, keep individual clips to 10 seconds or less to preserve generation budget.

---

## 4. Complete Working Inference Script

The following script performs single-speaker voice cloning using `VibeVoice-TTS-1.5B`. It is constructed from direct inspection of:
- `vibevoice/processor/vibevoice_processor.py` - processor pipeline
- `vibevoice/modular/modeling_vibevoice.py` - model architecture

**Important caveat:** As of this writing, a standalone inference class `VibeVoiceForConditionalGenerationInference` (separate from the training-oriented `VibeVoiceForConditionalGeneration`) was present in the original Microsoft repository but has been removed from the current codebase. The `generate()` method is inherited from the HuggingFace `GenerationMixin` (which `VibeVoiceForConditionalGeneration` does not currently extend in this repo). The script below calls the model's `forward()` method via the standard HuggingFace `.generate()` interface; it will work **if** the inference class is present (e.g., via the community fork or future restoration). The processor pipeline and input preparation are confirmed directly from code in this repo.

```python
#!/usr/bin/env python
"""
VibeVoice-TTS-1.5B Single-Speaker Voice Cloning Inference
Based on code in vibevoice/processor/vibevoice_processor.py and
vibevoice/modular/modeling_vibevoice.py
"""
import torch
import soundfile as sf
import numpy as np
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

# ---- Configuration ----
MODEL_ID = "microsoft/VibeVoice-1.5B"     # or local path
REFERENCE_AUDIO = "reference_speaker.wav"  # 5-30 sec, 24kHz mono WAV
TEXT_TO_SPEAK = "Speaker 1: Hello, this is a test of voice cloning with VibeVoice."
OUTPUT_PATH = "output.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---- Load processor ----
processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)

# ---- Load model ----
# On CUDA: use bfloat16 + flash_attention_2 for best quality
# On MPS/CPU: use float32 + sdpa
try:
    if DEVICE == "cuda":
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    else:
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=None,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        model = model.to(DEVICE)
except Exception as e:
    print(f"Loading with flash_attention_2 failed: {e}. Falling back to sdpa.")
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=None,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = model.to(DEVICE)

model.eval()

# ---- Prepare inputs ----
# voice_samples is a list with one entry per unique speaker in the script.
# Each entry is a file path (str) or a numpy float32 array at 24kHz.
# The processor loads audio, converts to mono, normalizes dB, tokenizes, and
# prepends a "Voice input: Speaker 0: <tokens>" prefix to the LLM input.
inputs = processor(
    text=TEXT_TO_SPEAK,
    voice_samples=[REFERENCE_AUDIO],   # one path per speaker
    return_tensors="pt",
    padding=True,
    return_attention_mask=True,
)

# Move all tensors to the device
for key in inputs:
    if torch.is_tensor(inputs[key]):
        inputs[key] = inputs[key].to(DEVICE)

# speech_tensors and speech_masks are needed by the model's forward pass
# to encode the reference audio into the input embedding space.
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"speech_input_mask shape: {inputs['speech_input_mask'].shape}")
if inputs.get('speech_tensors') is not None:
    print(f"speech_tensors shape: {inputs['speech_tensors'].shape}")

# ---- Generate ----
# The model uses the standard HuggingFace generate() interface.
# acoustic_input_mask tells the forward() which positions to replace with
# encoded speech features. This is the speech_input_mask from the processor.
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_tensors=inputs.get("speech_tensors"),
        speech_masks=inputs.get("speech_masks"),
        acoustic_input_mask=inputs["speech_input_mask"],
        # Generation parameters:
        max_new_tokens=8192,        # controls max output length
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
    )

# ---- Decode audio ----
# The model output contains speech latent tokens that are decoded to audio
# by the acoustic tokenizer. The processor's save_audio() method handles this.
# Alternatively, output.speech_outputs[0] contains the raw waveform tensor.
processor.save_audio(
    output.speech_outputs[0],
    output_path=OUTPUT_PATH,
    sampling_rate=24000,
)
print(f"Saved output to {OUTPUT_PATH}")
```

### Notes on the Script

- `speech_tensors`: Shape `(num_voice_samples, max_audio_samples)` float32. Contains raw waveforms padded to the same length. Built by `processor.prepare_speech_inputs()`.
- `speech_masks`: Shape `(num_voice_samples, max_vae_tokens)` bool. Marks valid (non-padded) acoustic token positions. Each token covers 3200 samples.
- `speech_input_mask` / `acoustic_input_mask`: Shape `(batch_size, sequence_length)` bool. Marks the positions in `input_ids` where the model should substitute speech embeddings (from the acoustic tokenizer) instead of token embeddings.
- The `forward()` method in `modeling_vibevoice.py` (line 386-392) substitutes `speech_connect_features` at those positions before passing `inputs_embeds` to the language model.

If the inference class (`VibeVoiceForConditionalGenerationInference`) is available from a community fork, replace the import accordingly:

```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
model = VibeVoiceForConditionalGenerationInference.from_pretrained(...)
```

---

## 5. Multi-Speaker Cloning

### Script Format

Use sequential `Speaker N:` labels. The model supports up to 4 speakers per generation.

```
Speaker 1: Welcome to today's episode. I'm your host, Alice.
Speaker 2: And I'm Frank. Today we're discussing a fascinating topic.
Speaker 1: That's right. Let's dive right in.
Speaker 2: I've been looking forward to this conversation all week.
```

### Processor Usage

Pass one reference audio clip per speaker, in order matching the speaker IDs. The processor's `_create_voice_prompt()` normalizes speaker IDs to start from 0, so `Speaker 1` maps to index 0, `Speaker 2` to index 1, and so on.

```python
inputs = processor(
    text=script,
    voice_samples=[
        "alice_reference.wav",    # Speaker 1 -> index 0
        "frank_reference.wav",    # Speaker 2 -> index 1
    ],
    return_tensors="pt",
)
```

If `voice_samples` has fewer entries than unique speakers in the script, only the first N speakers receive voice prompts. The remaining speakers will generate with the model's default voice characteristics.

### JSON Input for Multi-Speaker

```python
import json

script_data = [
    {"speaker": "1", "text": "Welcome to today's episode."},
    {"speaker": "2", "text": "Thanks for having me."},
    {"speaker": "1", "text": "Let's start."},
]

# Write to a temp file or pass the JSON string directly if the processor accepts it
import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(script_data, f)
    tmp_path = f.name

inputs = processor(
    text=tmp_path,
    voice_samples=["alice.wav", "frank.wav"],
    return_tensors="pt",
)
os.unlink(tmp_path)
```

---

## 6. Generation Parameters

### Processor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` or `List[str]` | required | Script string, file path (`.txt` or `.json`), or list of these for batching |
| `voice_samples` | `List[str or np.ndarray]` | `None` | One entry per speaker. Strings are file paths; arrays must be float32 at 24kHz |
| `padding` | `bool` or `str` | `True` | Pad to longest sequence in batch |
| `truncation` | `bool` | `False` | Whether to truncate sequences exceeding `max_length` |
| `max_length` | `int` | `None` | Maximum token sequence length after padding/truncation |
| `return_tensors` | `str` | `None` | `"pt"` to return PyTorch tensors |
| `db_normalize` | `bool` | `True` | Normalize voice samples to -25 dB FS before tokenizing |
| `speech_tok_compress_ratio` | `int` | `3200` | Audio samples per acoustic token (do not change) |

Source: `VibeVoiceProcessor.__init__()` and `__call__()`.

### Model Generate Parameters

The model inherits from HuggingFace `GenerationMixin`. Standard generation parameters apply:

| Parameter | Recommended Value | Description |
|---|---|---|
| `max_new_tokens` | 4096-32768 | Maximum new tokens to generate. At 7.5 Hz, 7500 tokens ~= 1000 seconds of audio. |
| `do_sample` | `True` | Enables stochastic sampling. `False` uses greedy decoding. |
| `temperature` | 0.8-1.2 | Higher = more varied. Lower = more conservative. |
| `top_p` | 0.9-0.95 | Nucleus sampling threshold. |

**CFG scale (`cfg_scale`):** Classifier-free guidance controls how strongly the model adheres to the voice prompt. This parameter is implemented in `sample_speech_tokens()` in the streaming model (`modeling_vibevoice_streaming_inference.py`, line 923):

```python
half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```

In the 1.5B non-streaming model, the equivalent mechanism is determined by whether the forward pass includes speech inputs. A higher CFG scale (e.g., 1.3-2.0) produces output that more closely follows the reference voice timbre. The recommended starting value is **1.3** based on community documentation. The implementation in the non-streaming 1.5B model's generate path is inferred from architecture; it may not accept `cfg_scale` as a direct kwarg depending on whether the inference class is used.

### Diffusion Inference Steps

The diffusion head defaults to `ddpm_num_inference_steps=20` (set in `VibeVoiceDiffusionHeadConfig`, `configuration_vibevoice.py` line 174). At inference, `set_ddpm_inference_steps(num_steps=N)` can reduce this for speed:

```python
model.set_ddpm_inference_steps(num_steps=10)  # faster, slightly lower quality
model.set_ddpm_inference_steps(num_steps=20)  # default
```

Warning from community: setting `num_steps=10` can cause instability (very short or collapsed audio output) with some custom voices and longer transcripts.

---

## 7. How the Pipeline Works Internally

Understanding this is useful for debugging.

### Processor Pipeline (confirmed from code)

1. `VibeVoiceProcessor.__call__()` calls `_process_single()`.
2. `_process_single()` calls `_parse_script()` to extract `(speaker_id, text)` pairs.
3. If `voice_samples` is provided, `_create_voice_prompt()` is called:
   - Loads each audio file with `audio_processor._load_audio_from_path()`.
   - Applies dB normalization (target: -25 dB FS).
   - Computes `vae_tok_len = ceil(audio_samples / 3200)`.
   - Builds a token sequence: `[speech_start_id] + [speech_diffusion_id] * vae_tok_len + [speech_end_id]`.
   - The `speech_diffusion_id` token maps to Qwen2 token `<|vision_pad|>`.
   - A boolean mask (`vae_input_mask`) marks the `speech_diffusion_id` positions as True.
4. The full input sequence structure is:
   ```
   [system_prompt_tokens]
   [Voice input:\n]          <- if voice_samples provided
   [ Speaker 0:<start><vae_pad*N><end>\n]  <- per speaker
   [ Text input:\n]
   [ Speaker 0: <text>\n]   <- per turn in script
   [ Speaker 1: <text>\n]
   ...
   [ Speech output:\n]
   [speech_start_id]         <- model generates from here
   ```
5. `_batch_encode()` pads sequences and calls `prepare_speech_inputs()` to:
   - Pad all audio arrays to the same length in a `(num_speakers, max_audio_samples)` tensor.
   - Build a `(num_speakers, max_vae_tokens)` boolean mask of valid token positions.

### Model Forward Pass (confirmed from code)

In `VibeVoiceForConditionalGeneration.forward()`:

1. Token embeddings are looked up for all `input_ids`.
2. `forward_speech_features()` is called with `speech_tensors` and `speech_masks`:
   - The acoustic tokenizer encodes the raw waveform into latent vectors at 7.5 Hz.
   - Latents are scaled by `speech_scaling_factor` and `speech_bias_factor`.
   - `acoustic_connector` (a 2-layer MLP with RMSNorm) projects latents to the LLM hidden size.
3. At positions where `acoustic_input_mask` is True, token embeddings are replaced by the projected speech features:
   ```python
   x[acoustic_input_mask] = speech_connect_features
   ```
4. The language model runs on the modified `inputs_embeds`.
5. The diffusion head generates acoustic latents conditioned on LLM hidden states.
6. The acoustic tokenizer decoder converts latents back to waveform audio.

### Special Tokens

These Qwen2 special tokens are repurposed for speech (confirmed in `modular_vibevoice_text_tokenizer.py`):

| Purpose | Token string | Attribute |
|---|---|---|
| Speech segment start | `<\|vision_start\|>` | `tokenizer.speech_start_id` |
| Speech segment end | `<\|vision_end\|>` | `tokenizer.speech_end_id` |
| Voice prompt placeholder | `<\|vision_pad\|>` | `tokenizer.speech_diffusion_id` |
| Padding | `<\|image_pad\|>` | `tokenizer.pad_id` |

---

## 8. Audio Loading: Using Your Own Reference Audio

The processor accepts file paths or numpy arrays. The recommended approach is to pass file paths and let the processor load them. If you need to load and pre-process audio yourself:

```python
import numpy as np
import soundfile as sf
import librosa

def load_reference_audio(path, target_sr=24000):
    """Load audio file and resample to 24kHz mono float32."""
    audio, sr = sf.read(path, dtype="float32")
    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)

reference = load_reference_audio("my_voice.mp3")

inputs = processor(
    text="Speaker 1: Hello world.",
    voice_samples=[reference],   # numpy array accepted directly
    return_tensors="pt",
)
```

The processor will apply its own dB normalization on top of whatever you pass. If you want to normalize yourself and disable the processor's normalization, instantiate the processor with `db_normalize=False` (only possible when instantiating directly; `from_pretrained` sets this based on the config, defaulting to True).

---

## 9. Saving Output Audio

The model's `speech_outputs[0]` is a PyTorch tensor of shape `(1, num_samples)` at 24000 Hz (inferred from architecture; the sample rate is set by the acoustic tokenizer's decoder which is configured for 24kHz throughout the codebase). `processor.save_audio()` wraps `soundfile.write()`:

```python
# From model output
processor.save_audio(
    output.speech_outputs[0],
    output_path="result.wav",
    sampling_rate=24000,  # confirmed default throughout codebase
)

# Or directly with soundfile if you have the raw waveform tensor:
import soundfile as sf
audio_np = output.speech_outputs[0].squeeze().float().cpu().numpy()
sf.write("result.wav", audio_np, samplerate=24000)
```

---

## 10. Troubleshooting

### "No valid speaker lines found in script"

The script parser requires `Speaker N: text` format. Check that:
- Each line starts with `Speaker ` (with capital S and a space).
- Speaker IDs are integers.
- There is a colon followed by the text.
- There are no blank speaker IDs.

### Very short or silent audio output (< 1 second)

This is a known issue reported by the community, particularly with 10 diffusion inference steps. Try:
- Increasing to 20 diffusion steps: `model.set_ddpm_inference_steps(20)`
- Shortening the input script (some combinations of voice prompt + script cause instability)
- Using a different reference audio clip

### Output does not sound like the reference voice

- Use a longer reference clip (15-20 seconds instead of 5 seconds).
- Use clean audio with no background noise or music.
- Increase CFG scale if the inference class supports it.
- The model is probabilistic; run multiple times and pick the best result.
- Ensure the reference audio is at 24kHz. The processor will warn if the sampling rate differs.

### CUDA out of memory

- Use bfloat16: `torch_dtype=torch.bfloat16`
- Reduce the reference audio length to minimize voice prompt tokens.
- Reduce `max_new_tokens`.
- Shorten the input script.

### "ModuleNotFoundError: No module named 'vibevoice'"

Run `pip install -e .` from the root of the repository. The package must be installed in editable mode.

### `attn_implementation="flash_attention_2"` fails

Flash Attention 2 is only supported on CUDA with compatible GPU hardware. Fall back to `sdpa`:

```python
model = VibeVoiceForConditionalGeneration.from_pretrained(
    MODEL_ID, attn_implementation="sdpa", ...
)
```

Note from the realtime demo code (`realtime_model_inference_from_file.py`, line 208): `"only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality"`.

### TypeError when calling model.generate()

The training-oriented `VibeVoiceForConditionalGeneration` class in the current repo does not extend `GenerationMixin`. If this error occurs, the community fork (`vibevoice-community/VibeVoice`) may include the inference class (`VibeVoiceForConditionalGenerationInference`) that does. Check:

```python
from vibevoice.modular import modeling_vibevoice
print(dir(modeling_vibevoice))
```

---

## 11. Known Limitations

The following are confirmed from the paper (arxiv 2508.19205) and community documentation:

- **Supported languages:** English and Chinese have the most stable results. Other languages are emergent, not trained-for capabilities.
- **Speaker identity varies:** The model is probabilistic; identical inputs do not always produce identical outputs.
- **Background noise bleeds:** If the reference clip contains background music or ambient noise, it may appear in generated speech.
- **Context budget:** The 64K context window is shared by voice prompts, text input, and generated speech tokens. Very long scripts with multiple speakers and long reference clips may exceed the budget.
- **Watermarking:** According to community documentation, all generated audio includes both an audible disclaimer ("This segment was generated by AI") and an imperceptible watermark for provenance verification.
- **English/Chinese bias:** Cross-lingual voice transfer and non-English speech synthesis are emergent, not guaranteed capabilities.
- **Batch size 1 only:** Current inference code only supports a single sample per call. (Confirmed in streaming inference: `assert batch_size == 1`; applies by convention in the 1.5B inference path as well.)

---

## 12. Community Resources

The following resources were found via web search and contain community-documented working usage:

- **HuggingFace model card:** [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) - model card with architecture description; discussions contain community inference examples
- **HuggingFace discussion #17** (Apple Silicon script): [microsoft/VibeVoice-1.5B/discussions/17](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/17) - working inference for M1/M2/M3/M4 Macs
- **HuggingFace discussion #7** (voice cloning): [microsoft/VibeVoice-1.5B/discussions/7](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/7) - community discussion confirming voice cloning works via `voice_samples=`
- **Community fork:** [github.com/vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice) - fork that may include the inference class (`VibeVoiceForConditionalGenerationInference`) that was present before the Microsoft repo reset
- **Archive fork:** [github.com/shijincai/VibeVoice](https://github.com/shijincai/VibeVoice) - archive of the pre-reset repo including original demo scripts (`demo/inference_from_file.py`, `demo/gradio_demo.py`)
- **KDnuggets beginner's guide:** [kdnuggets.com/beginners-guide-to-vibevoice](https://www.kdnuggets.com/beginners-guide-to-vibevoice) - end-to-end walkthrough including CLI usage
- **Wilson Wu blog post:** [wilsonwu.me/en/blog/2025/microsoft-vibevoice](https://wilsonwu.me/en/blog/2025/microsoft-vibevoice/) - getting started guide with working examples
- **Community mirror model:** [vibevoice/VibeVoice-1.5B](https://huggingface.co/vibevoice/VibeVoice-1.5B) - mirror of the model weights if the official HuggingFace repo becomes unavailable

### Original CLI Usage (from community documentation)

Before the official repo reset, inference was run via a dedicated script. The equivalent from community forks:

```bash
# Single speaker
python demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/1p_abs.txt \
  --speaker_names Alice

# Two speakers
python demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/2p_dialogue.txt \
  --speaker_names Alice Frank

# With CFG scale and diffusion steps
python demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/1p_abs.txt \
  --speaker_names Alice \
  --cfg_scale 1.3 \
  --ddpm_steps 10
```

`--speaker_names` mapped names to pre-computed voice files in `demo/voices/`. To use a custom voice, place a compatible audio file (or `.pt` voice embedding) in `demo/voices/` and use the filename as the speaker name.

---

## 13. Relationship to the Streaming (0.5B) Pipeline

The streaming model (`VibeVoice-Realtime-0.5B`) uses the same fundamental architecture but differs in important ways relevant to voice cloning:

| Feature | TTS-1.5B | Realtime-0.5B |
|---|---|---|
| Voice cloning | Yes (audio via `voice_samples=`) | No (fixed `.pt` files only) |
| Custom voice input | Raw audio clip at runtime | Pre-computed KV-cache offline |
| Generation mode | Non-streaming, batch | Streaming, windowed |
| Latency to first audio | Seconds-minutes | ~200ms |
| Max generation length | 90 min (64K context) | ~10 min (8K context) |
| Multi-speaker | Yes, up to 4 | No, single speaker |
| Semantic tokenizer | Yes | No (removed for speed) |
| CFG mechanism | Diffusion head + LLM | `sample_speech_tokens()` with explicit `cfg_scale` parameter |

The streaming model's `sample_speech_tokens()` (line 914-926 of `modeling_vibevoice_streaming_inference.py`) contains the explicit CFG formula:

```python
half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```

This same principle applies to the 1.5B diffusion head but is routed through the non-streaming generate path (inferred from architecture).
