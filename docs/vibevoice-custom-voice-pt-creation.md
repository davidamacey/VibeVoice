# VibeVoice Realtime-0.5B: Custom Voice .pt File Format and Creation

This document is a complete reverse-engineering of the `.pt` voice file format used by
`microsoft/VibeVoice-Realtime-0.5B`. It explains the exact binary layout, the data each
field holds, and provides a step-by-step procedure and a complete Python script for
creating a new `.pt` voice file from arbitrary audio.

---

## 1. File Format: Exact Structure

A `.pt` file is a PyTorch serialised Python `dict` with exactly **four keys**.
It is saved and loaded with `torch.save` / `torch.load(..., weights_only=False)`.

```
{
    "lm":        BaseModelOutputWithPast,   # positive conditioning - lower LM layers
    "tts_lm":    VibeVoiceCausalLMOutputWithPast,  # positive conditioning - upper TTS layers
    "neg_lm":    BaseModelOutputWithPast,   # negative conditioning - lower LM layers
    "neg_tts_lm": VibeVoiceCausalLMOutputWithPast, # negative conditioning - upper TTS layers
}
```

Every value is a `transformers.modeling_outputs.ModelOutput` subclass, which supports
both attribute-style (`v.last_hidden_state`) and dict-style (`v['last_hidden_state']`)
access. The fields that must be present on each value are:

| Field | Type | Description |
|---|---|---|
| `last_hidden_state` | `torch.Tensor` shape `(1, S, H)` | Final layer hidden states for all S prompt positions. H is the model hidden dimension. |
| `past_key_values` | `DynamicCache` or nested tuple | KV-cache for all transformer layers covering the S prompt positions. |
| `attentions` | `None` | Not stored (output_attentions=False at prefill time). |

### Concrete shape values (VibeVoice-Realtime-0.5B)

The 0.5B model uses a Qwen2-based backbone split into two halves:

- **Lower LM** (`language_model`): handles the `lm` / `neg_lm` keys. H = 896.
- **Upper TTS LM** (`tts_language_model`): handles the `tts_lm` / `neg_tts_lm` keys. H = 896.

For a ~10-second reference clip the positive keys (`lm`, `tts_lm`) will have S roughly
in the range 50-120 depending on audio length. The negative keys (`neg_lm`, `neg_tts_lm`)
always have S = 1 (a single padding token).

Example from a shipped preset:

```
lm.last_hidden_state:        shape=(1, N_prompt, 896), dtype=bfloat16
lm.past_key_values:          DynamicCache, num_layers = (total_layers - tts_backbone_layers)
tts_lm.last_hidden_state:    shape=(1, N_prompt, 896), dtype=bfloat16
tts_lm.past_key_values:      DynamicCache, num_layers = tts_backbone_layers
neg_lm.last_hidden_state:    shape=(1, 1, 896), dtype=bfloat16
neg_lm.past_key_values:      DynamicCache (single-position KV)
neg_tts_lm.last_hidden_state: shape=(1, 1, 896), dtype=bfloat16
neg_tts_lm.past_key_values:  DynamicCache (single-position KV)
```

---

## 2. How the Model Uses the .pt File at Inference Time

Understanding the consumption path explains why the format must be exactly this shape.

### 2a. Processor: `process_input_with_cached_prompt`

File: `vibevoice/processor/vibevoice_streaming_processor.py`

```python
input_id_length     = cached_prompt['lm']['last_hidden_state'].size(1)
tts_lm_input_id_length = cached_prompt['tts_lm']['last_hidden_state'].size(1)
```

The processor uses `last_hidden_state.size(1)` (the sequence length) to construct
pseudo `input_ids` of the correct length for the cached prefix. No actual audio is
re-processed at inference time; the KV-cache absorbs all voice context.

### 2b. Model: `generate`

File: `vibevoice/modular/modeling_vibevoice_streaming_inference.py`

```python
outputs              = all_prefilled_outputs["lm"]
tts_lm_outputs       = all_prefilled_outputs["tts_lm"]
negative_outputs     = all_prefilled_outputs["neg_lm"]
tts_lm_negative_outputs = all_prefilled_outputs["neg_tts_lm"]
```

Immediately after extraction, each output is fed into `_update_model_kwargs_for_generation`
which installs `past_key_values` into the generation state. From that point on the
generate loop never re-runs the voice prefix; it only extends the KV-cache with new
text and speech tokens.

The `neg_lm` / `neg_tts_lm` caches seed the negative branch that runs in parallel for
Classifier-Free Guidance (CFG). At each step the positive and negative hidden states
are extracted and combined:

```python
positive_condition = tts_lm_outputs.last_hidden_state[..., -1, :]
negative_condition = tts_lm_negative_outputs.last_hidden_state[..., -1, :]
# CFG formula inside sample_speech_tokens:
# latent = neg + cfg_scale * (pos - neg)
```

---

## 3. What Prompt Produces the .pt Tensors

### 3a. Positive conditioning (lm + tts_lm)

The full token sequence fed through the model during the prefill pass is:

```
[system_prompt_tokens]
[" Voice input:\n" tokens]
[" Speaker 0:" tokens]  [<|vision_start|>]  [<|vision_pad|>] * N_vae  [<|vision_end|>]  ["\n" token]
[" Text input:\n" tokens]
[" Speech output:\n" tokens]
[<|vision_start|>]
```

Where:

- `<|vision_start|>` = token ID for speech start (tokenizer.speech_start_id)
- `<|vision_pad|>` = token ID for speech diffusion placeholder (tokenizer.speech_diffusion_id)
- `<|vision_end|>` = token ID for speech end (tokenizer.speech_end_id)
- `N_vae` = `ceil(audio_samples / 3200)` (the acoustic VAE compression ratio is 3200 samples
  per latent frame at 24 kHz, i.e. 7.5 Hz)

The speech-position tokens (`<|vision_pad|>`) are not actually fed as token embeddings;
they act as placeholder slots whose embeddings are overwritten by the acoustic connector
output. This is the same pattern as the 1.5B TTS model's voice prompt processing.

**System prompt text** (from `VibeVoiceProcessor`):

```
 Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.
```

Note the leading space; the tokenizer encodes it with `add_special_tokens=True` (calls
`self.tokenizer.encode(self.system_prompt)` not `add_special_tokens=False`).

### 3b. Negative conditioning (neg_lm + neg_tts_lm)

The negative branch uses a single-token input: the special padding token `<|image_pad|>`
(token ID 151655). This is the `tokenizer.pad_id` which is used as a "null" conditioning
signal for CFG. It always has sequence length 1.

```python
neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
```

---

## 4. Audio Preprocessing Requirements

The acoustic tokenizer is a streaming convolutional VAE operating at 24 kHz.

| Parameter | Value |
|---|---|
| Sample rate | 24000 Hz |
| Channels | Mono (stereo is averaged to mono) |
| dtype | float32 numpy array, values in [-1, 1] |
| Normalization | dB FS normalization to -25 dBFS (default, configurable) |
| Min useful length | ~1 second (7-8 VAE frames) |
| Max recommended length | 30 seconds (to keep the KV-cache compact) |

dB normalization is handled by `vibevoice/processor/audio_utils.py:AudioNormalizer`
(target_dB_FS=-25, eps=1e-6).

The compression ratio is 3200 raw audio samples per VAE latent frame:

```
N_vae_frames = ceil(len(audio_array) / 3200)
```

---

## 5. Step-by-Step: Creating a .pt Voice File from a WAV

### Prerequisites

```bash
pip install -e /path/to/VibeVoice
# Model must be downloaded (local path or HuggingFace hub ID)
MODEL_PATH="microsoft/VibeVoice-Realtime-0.5B"
```

### High-Level Procedure

1. Load and preprocess the reference audio clip (resample to 24 kHz, mono, dB-normalize).
2. Build the positive prompt token sequence (system + Voice input section + Text input prefix + Speech output start).
3. Build the speech input tensor by running the audio through the acoustic tokenizer's encoder.
4. Run a single forward pass through `model.forward_lm(...)` over the full positive prompt. Save the output as `lm`.
5. Run a single forward pass through `model.forward_tts_lm(...)` using the LM's hidden states and the positive prompt for the TTS upper layers. Save the output as `tts_lm`.
6. Build the negative prompt (single `<|image_pad|>` token). Run the same two forward passes. Save as `neg_lm` and `neg_tts_lm`.
7. Assemble the dict and call `torch.save`.

### Complete Python Script

```python
"""
create_voice_pt.py
------------------
Creates a VibeVoice Realtime-0.5B compatible .pt voice file from an audio clip.

Usage:
    python create_voice_pt.py \
        --audio /path/to/reference.wav \
        --model microsoft/VibeVoice-Realtime-0.5B \
        --output my_voice.pt \
        [--device cuda|cpu|mps]

The audio should be 5-30 seconds of clean single-speaker speech at any sample
rate (the script resamples to 24 kHz). Stereo is converted to mono.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

# ---- audio loading -------------------------------------------------------

def load_audio(path: str, target_sr: int = 24000) -> np.ndarray:
    """Load and resample audio to target_sr, convert to mono float32."""
    try:
        import librosa
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
    except ImportError:
        import soundfile as sf
        wav, sr = sf.read(path, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != target_sr:
            raise RuntimeError(
                f"soundfile loaded audio at {sr} Hz but target is {target_sr} Hz. "
                "Install librosa for automatic resampling: pip install librosa"
            )
    return wav.astype(np.float32)


def db_normalize(wav: np.ndarray, target_dB_FS: float = -25.0, eps: float = 1e-6) -> np.ndarray:
    """Normalize audio to target dBFS RMS level."""
    rms = np.sqrt(np.mean(wav ** 2) + eps)
    target_rms = 10 ** (target_dB_FS / 20.0)
    return wav * (target_rms / rms)


# ---- prompt construction -------------------------------------------------

SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)

def build_positive_prompt_tokens(tokenizer, n_vae_frames: int):
    """
    Build the full positive voice-prompt token sequence.

    Layout:
        [system_prompt]
        " Voice input:\n"
        " Speaker 0:"  <|vision_start|>  [<|vision_pad|>]*N  <|vision_end|>  "\n"
        " Text input:\n"
        " Speech output:\n"
        <|vision_start|>

    Returns:
        input_ids       : list[int]  token id sequence
        speech_positions: list[bool] True at positions occupied by VAE placeholder tokens
    """
    speech_start_id = tokenizer.speech_start_id    # <|vision_start|>
    speech_end_id   = tokenizer.speech_end_id      # <|vision_end|>
    vae_token_id    = tokenizer.speech_diffusion_id # <|vision_pad|>

    # 1. System prompt (add_special_tokens=True to match training)
    system_tokens = tokenizer.encode(SYSTEM_PROMPT)
    speech_pos = [False] * len(system_tokens)

    # 2. Voice input header
    vi_header = tokenizer.encode(" Voice input:\n", add_special_tokens=False)
    system_tokens += vi_header
    speech_pos += [False] * len(vi_header)

    # 3. Speaker 0 voice segment
    spk_prefix = tokenizer.encode(" Speaker 0:", add_special_tokens=False)
    nl_token   = tokenizer.encode("\n", add_special_tokens=False)
    speaker_tokens = (
        spk_prefix
        + [speech_start_id]
        + [vae_token_id] * n_vae_frames
        + [speech_end_id]
        + nl_token
    )
    speaker_mask = (
        [False] * len(spk_prefix)
        + [False]                      # <|vision_start|>
        + [True] * n_vae_frames        # VAE placeholder positions
        + [False]                      # <|vision_end|>
        + [False] * len(nl_token)
    )
    system_tokens += speaker_tokens
    speech_pos    += speaker_mask

    # 4. Text input + Speech output prefix
    ti = tokenizer.encode(" Text input:\n", add_special_tokens=False)
    so = tokenizer.encode(" Speech output:\n", add_special_tokens=False)
    tail = ti + so + [speech_start_id]
    system_tokens += tail
    speech_pos    += [False] * len(tail)

    return system_tokens, speech_pos


def build_negative_prompt_tokens(tokenizer):
    """Single <|image_pad|> token - the null/negative conditioning signal."""
    pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return [pad_id], [False]


# ---- acoustic encoding ---------------------------------------------------

def encode_audio_to_speech_embeds(model, wav: np.ndarray, device, dtype):
    """
    Run audio through the acoustic tokenizer and project to LM hidden dim.

    Returns:
        acoustic_embeds: Tensor (1, N_vae, H) - speech latents projected to LM dim
        n_vae_frames:    int
    """
    n_vae_frames = math.ceil(len(wav) / 3200)

    wav_tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    # shape: (1, 1, T_audio) - expected by acoustic_tokenizer

    with torch.no_grad():
        # Encode to continuous latent
        latent = model.acoustic_tokenizer.encode(wav_tensor)  # (1, vae_dim, N_vae)
        # Scale latent (model stores scaling factors as buffers)
        sf = model.speech_scaling_factor.to(device)
        bf = model.speech_bias_factor.to(device)
        scaled = (latent - bf) * sf            # normalisation applied during training
        scaled = scaled.permute(0, 2, 1)       # (1, N_vae, vae_dim)
        # Project to transformer hidden dim
        acoustic_embeds = model.acoustic_connector(scaled.to(dtype))  # (1, N_vae, H)

    return acoustic_embeds, n_vae_frames


# ---- forward helpers -----------------------------------------------------

def prefill_lm(model, input_ids_list, speech_positions, acoustic_embeds, device, dtype):
    """
    Forward pass through the lower LM (model.model.language_model).

    Non-speech positions use token embeddings; speech positions are replaced
    with acoustic_embeds slices (matching the True positions in speech_positions).

    Returns:
        BaseModelOutputWithPast with last_hidden_state and past_key_values.
    """
    ids_tensor  = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attn_tensor = torch.ones_like(ids_tensor)

    # Build inputs_embeds: start from token embeddings then splice acoustic
    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids_tensor)  # (1, S, H)

    # Splice acoustic embeddings into the VAE-placeholder positions
    speech_idx = [i for i, v in enumerate(speech_positions) if v]
    if speech_idx and acoustic_embeds is not None:
        n = min(len(speech_idx), acoustic_embeds.shape[1])
        embeds[:, speech_idx[:n], :] = acoustic_embeds[:, :n, :].to(dtype=embeds.dtype)

    with torch.no_grad():
        output = model.forward_lm(
            inputs_embeds=embeds,
            attention_mask=attn_tensor,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    return output


def prefill_tts_lm(model, input_ids_list, speech_positions, lm_hidden, acoustic_embeds,
                   device, dtype):
    """
    Forward pass through the upper TTS LM (model.model.tts_language_model).

    The TTS LM replaces the last K positions of inputs_embeds with the LM's
    last_hidden_state and adds type embeddings (text=1, speech=0).

    For the prefill pass we feed:
      - All positions: token embeddings (will be overwritten by splice)
      - The function forward_tts_lm receives lm_last_hidden_state (= all LM hidden states)
        and the full speech_positions mask as tts_text_masks.

    Returns:
        VibeVoiceCausalLMOutputWithPast with last_hidden_state and past_key_values.
    """
    S = len(input_ids_list)
    ids_tensor  = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attn_tensor = torch.ones_like(ids_tensor)

    # tts_text_masks: 1 = text position, 0 = speech position
    # shape (1, S) matching input sequence
    text_mask = torch.tensor(
        [[0 if v else 1 for v in speech_positions]], dtype=torch.long, device=device
    )

    with torch.no_grad():
        output = model.forward_tts_lm(
            input_ids=ids_tensor,
            attention_mask=attn_tensor,
            lm_last_hidden_state=lm_hidden,   # (1, S, H) - full sequence
            tts_text_masks=text_mask,          # (1, S)
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    return output


# ---- main logic ----------------------------------------------------------

def create_voice_pt(
    audio_path: str,
    model_path: str,
    output_path: str,
    device: str = "cuda",
    db_normalize_audio: bool = True,
):
    """
    Full pipeline: audio -> .pt voice file.

    Args:
        audio_path:       Path to reference WAV (any sample rate, any channel count).
        model_path:       HuggingFace model ID or local path.
        output_path:      Destination .pt file path.
        device:           'cuda', 'cpu', or 'mps'.
        db_normalize_audio: Whether to apply dBFS normalisation (matches default).
    """
    print(f"Loading model from {model_path} ...")

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    if device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl  = "flash_attention_2"
    elif device == "mps":
        load_dtype = torch.float32
        attn_impl  = "sdpa"
    else:
        load_dtype = torch.float32
        attn_impl  = "sdpa"

    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map=device if device in ("cuda", "cpu") else None,
            attn_implementation=attn_impl,
        )
        if device == "mps":
            model.to("mps")
    except Exception:
        print("flash_attention_2 unavailable, falling back to sdpa")
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map=device if device in ("cuda", "cpu") else None,
            attn_implementation="sdpa",
        )
        if device == "mps":
            model.to("mps")

    model.eval()

    # ------------------------------------------------------------------
    # 1. Load and preprocess reference audio
    # ------------------------------------------------------------------
    print(f"Loading audio from {audio_path} ...")
    wav = load_audio(audio_path, target_sr=24000)
    if db_normalize_audio:
        wav = db_normalize(wav)
    print(f"  Audio: {len(wav)/24000:.2f}s, {len(wav)} samples")

    n_vae_frames = math.ceil(len(wav) / 3200)
    print(f"  VAE frames: {n_vae_frames}")

    # ------------------------------------------------------------------
    # 2. Encode audio through acoustic tokenizer
    # ------------------------------------------------------------------
    print("Encoding audio through acoustic tokenizer ...")
    acoustic_embeds, _ = encode_audio_to_speech_embeds(
        model, wav, device=model.device, dtype=load_dtype
    )
    print(f"  acoustic_embeds shape: {acoustic_embeds.shape}")

    # ------------------------------------------------------------------
    # 3. Build positive prompt tokens
    # ------------------------------------------------------------------
    pos_ids, pos_speech_mask = build_positive_prompt_tokens(tokenizer, n_vae_frames)
    print(f"  Positive prompt length: {len(pos_ids)} tokens")

    # ------------------------------------------------------------------
    # 4. Positive prefill - lower LM
    # ------------------------------------------------------------------
    print("Prefilling positive LM (lower layers) ...")
    lm_output = prefill_lm(
        model, pos_ids, pos_speech_mask, acoustic_embeds,
        device=model.device, dtype=load_dtype
    )
    print(f"  lm.last_hidden_state: {lm_output.last_hidden_state.shape}")

    # ------------------------------------------------------------------
    # 5. Positive prefill - upper TTS LM
    # ------------------------------------------------------------------
    print("Prefilling positive TTS LM (upper layers) ...")
    tts_lm_output = prefill_tts_lm(
        model, pos_ids, pos_speech_mask,
        lm_hidden=lm_output.last_hidden_state,
        acoustic_embeds=acoustic_embeds,
        device=model.device, dtype=load_dtype
    )
    print(f"  tts_lm.last_hidden_state: {tts_lm_output.last_hidden_state.shape}")

    # ------------------------------------------------------------------
    # 6. Build negative prompt and prefill
    # ------------------------------------------------------------------
    neg_ids, neg_speech_mask = build_negative_prompt_tokens(tokenizer)
    print(f"  Negative prompt length: {len(neg_ids)} tokens (always 1)")

    print("Prefilling negative LM (lower layers) ...")
    neg_lm_output = prefill_lm(
        model, neg_ids, neg_speech_mask, acoustic_embeds=None,
        device=model.device, dtype=load_dtype
    )

    print("Prefilling negative TTS LM (upper layers) ...")
    neg_tts_lm_output = prefill_tts_lm(
        model, neg_ids, neg_speech_mask,
        lm_hidden=neg_lm_output.last_hidden_state,
        acoustic_embeds=None,
        device=model.device, dtype=load_dtype
    )

    # ------------------------------------------------------------------
    # 7. Assemble and save
    # ------------------------------------------------------------------
    all_prefilled = {
        "lm":        lm_output,
        "tts_lm":    tts_lm_output,
        "neg_lm":    neg_lm_output,
        "neg_tts_lm": neg_tts_lm_output,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(all_prefilled, output_path)
    print(f"Saved voice file to {output_path}")

    # Sanity check
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    for k in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
        lhs = loaded[k]["last_hidden_state"] if isinstance(loaded[k], dict) \
              else loaded[k].last_hidden_state
        print(f"  {k}: last_hidden_state shape = {lhs.shape}")
    print("Done.")


# ---- CLI -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Create a VibeVoice Realtime-0.5B voice .pt file from an audio clip."
    )
    p.add_argument("--audio",  required=True, help="Path to reference audio file (WAV/MP3/FLAC)")
    p.add_argument("--model",  default="microsoft/VibeVoice-Realtime-0.5B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--output", required=True, help="Output .pt file path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable dBFS normalisation")
    args = p.parse_args()

    create_voice_pt(
        audio_path=args.audio,
        model_path=args.model,
        output_path=args.output,
        device=args.device,
        db_normalize_audio=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
```

Save this script as, e.g., `tools/create_voice_pt.py` and run:

```bash
python tools/create_voice_pt.py \
    --audio reference_speaker.wav \
    --model microsoft/VibeVoice-Realtime-0.5B \
    --output demo/voices/streaming_model/my_custom_voice.pt \
    --device cuda
```

Then pass `--speaker_name my_custom_voice` to `demo/realtime_model_inference_from_file.py`
or drop the `.pt` file into `demo/voices/streaming_model/` for the web server to
discover it automatically.

---

## 6. Important Implementation Notes and Caveats

### 6a. forward_tts_lm splice semantics

`forward_tts_lm` is documented as accepting `lm_last_hidden_state` with shape `(B, K, H)` and
uses it to overwrite the **tail** K positions of the inputs_embeds:

```python
start_idx = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
inputs_embeds[:, start_idx:, :] = lm_last_hidden_state
```

For the prefill pass K = S (the full sequence), so start_idx = 0 and the entire
embedding sequence is replaced by the LM hidden states. The token embeddings built
from `input_ids` serve only as shape/device scaffolding for this pass.

### 6b. tts_text_masks dimension at prefill vs decode time

At decode time `tts_text_masks` has shape `(B, 1)` (a single token per step). At
prefill time you need shape `(B, S)` to cover the full prompt. The script above handles
this correctly. If you see shape mismatches in `tts_input_types` embedding lookups,
check that your mask has the right length.

### 6c. Acoustic tokenizer encode API

The exact API of `model.acoustic_tokenizer.encode(wav_tensor)` depends on the specific
VAE architecture loaded with the model. The `wav_tensor` must have shape `(1, 1, T)` -
batch, channel, time. The output is typically `(1, vae_dim, N_vae)`. Consult the model's
`configuration_vibevoice.py:VibeVoiceAcousticTokenizerConfig` for the `vae_dim` value
(default 64).

### 6d. Scaling factors

`model.speech_scaling_factor` and `model.speech_bias_factor` are learned buffers that
normalize the VAE latents. They are stored inside the model checkpoint. The exact
normalization formula applied during training is:

```python
normalised_latent = (latent - bias_factor) * scaling_factor
```

Both factors are scalar tensors registered with `register_buffer`. If either evaluates
to `nan` the model has not been trained with scaling, and you should skip that step.

### 6e. CPU/MPS limitations

On CPU the prefill is slow (~30-60 seconds for a 10-second reference clip). MPS
(Apple Silicon) works but must use `torch.float32` and `attn_implementation="sdpa"`.
The shipped preset `.pt` files use `bfloat16` tensors (CUDA training precision); when
loaded on CPU/MPS with `map_location="cpu"` they are cast automatically.

### 6f. Voice quality factors

- Best reference length: 10-20 seconds of clean, noise-free single-speaker speech.
- The model does not separate speakers; feeding multi-speaker audio will average voice
  characteristics and produce inconsistent results.
- English reference audio produces the most reliable cloning. Other languages may work
  but have not been validated with the 0.5B model.
- dBFS normalisation is strongly recommended; without it very quiet or very loud
  references produce degraded conditioning.
- The reference clip's text content does not matter; only acoustic voice characteristics
  (pitch, timbre, speaking style) are captured in the KV-cache conditioning.

### 6g. No public torch.save in the codebase

There is no `torch.save` call anywhere in the public VibeVoice repository that produces
the voice `.pt` format. The prefill tool is internal to Microsoft. The script above was
derived entirely by reverse-engineering:

- The `.pt` file's key names from `modeling_vibevoice_streaming_inference.py` lines 678-681
- The expected field access patterns from `vibevoice_streaming_processor.py` lines 220-221
- The token sequence from `vibevoice_processor.py:_create_voice_prompt` (lines 406-459)
- The acoustic encoding path from `VibeVoiceStreamingModel` forward implementations
- The negative conditioning token from line 620 of the inference file

The script has not been run against the live model and should be treated as a
**research-grade starting point** that may require debugging against the actual model
checkpoint's acoustic tokenizer API.

---

## 7. File Placement

Drop the new `.pt` into:

```
demo/voices/streaming_model/<voice_name>.pt
```

Both the web server (`demo/web/app.py`) and the batch script
(`demo/realtime_model_inference_from_file.py`) auto-discover all `.pt` files in that
directory via glob. The voice key used in API calls and `--speaker_name` is the filename
stem without extension.
