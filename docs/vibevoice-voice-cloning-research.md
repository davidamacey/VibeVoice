# VibeVoice: Voice Cloning Research Reference

This document captures findings on voice cloning capabilities across the VibeVoice model family, explains what LoRA is and is not, clarifies the status of the 1.5B model, and provides guidance on integrating VibeVoiceFusion with the latest codebase.

---

## 1. Model Overview and Status

| Model | Size | Voice Cloning | Status |
|---|---|---|---|
| VibeVoice-TTS-1.5B | 1.5B | Yes (zero-shot via audio prompt) | Weights on HF; inference code present but usage docs removed |
| VibeVoice-Realtime-0.5B | 0.5B | No (fixed pre-computed voices only) | Fully active |
| VibeVoice-ASR-7B | 7B | N/A (speech recognition) | Fully active |

---

## 2. Does VibeVoice Support Voice Cloning?

**Short answer: Yes, but only through the 1.5B TTS model, and only with caveats.**

### 2a. VibeVoice-TTS-1.5B - Voice Cloning via Voice Prompts

The 1.5B TTS model supports **zero-shot voice cloning**. The mechanism is a "voice prompt" - one or more raw audio clips (e.g., WAV, MP3) of the target speaker are prepended to the input context before generation. The model conditions its output on the acoustic characteristics of those samples and attempts to match the voice.

**How it works architecturally:**

1. Audio clips are loaded and (optionally) dB-normalized.
2. Each clip is tokenized by the acoustic tokenizer at 7.5 Hz into a sequence of continuous latent tokens.
3. These tokens are prepended to the input sequence under a `Voice input:` header with `Speaker N:` labels matching speakers in the transcript.
4. At inference, the LLM and diffusion head generate speech that is conditioned on those voice latents.

**Relevant code:**

- Processor: `vibevoice/processor/vibevoice_processor.py`
  - `VibeVoiceProcessor.__call__()` - accepts `voice_samples=` parameter (list of audio paths or numpy arrays, one per speaker)
  - `_create_voice_prompt()` (line ~406) - tokenizes the audio, builds the `Voice input:` prefix sequence
- Model: `vibevoice/modular/modeling_vibevoice.py` - `VibeVoiceForConditionalGeneration`

**Usage pattern (code exists, officially undocumented):**

```python
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
model = VibeVoiceForConditionalGeneration.from_pretrained(
    "microsoft/VibeVoice-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

script = """
Speaker 0: Hello, welcome to the show.
Speaker 1: Thanks for having me today.
"""

# voice_samples: one audio file path per unique speaker in the script
inputs = processor(
    text=script,
    voice_samples=["path/to/speaker0_sample.wav", "path/to/speaker1_sample.wav"],
    return_tensors="pt"
)

outputs = model.generate(**inputs)
```

**Why usage was disabled:** On 2025-09-05 Microsoft removed the inference documentation (but not the model weights or code) after observing widespread misuse for impersonation and deepfakes. The weights remain accessible at `microsoft/VibeVoice-1.5B` on HuggingFace. Community members have independently documented usage via HuggingFace discussions.

**Voice quality for cloning:**

- Works best with 5-30 seconds of clean, single-speaker audio
- Cross-lingual voice transfer is supported but unstable (emergent capability)
- Background music/noise in the reference sample can bleed into output
- English and Chinese are the most stable; other languages produce variable results
- The (disabled) `VibeVoice-Large` (32K context, ~45 min generation) was reportedly more stable than 1.5B

---

### 2b. VibeVoice-Realtime-0.5B - NO Open Voice Cloning

The streaming model does **not** support arbitrary voice cloning. Voices are supplied as **pre-computed `.pt` files** (PyTorch tensors) stored in `demo/voices/streaming_model/`.

**What the `.pt` files contain:** A Python dict with four keys:
- `lm` - prefilled hidden states from the main language model
- `tts_lm` - prefilled hidden states from the TTS language model head
- `neg_lm` - negative conditioning hidden states (for CFG)
- `neg_tts_lm` - negative TTS conditioning hidden states

These represent the **KV-cache state** produced by running a short voice audio clip through the model offline, effectively "pre-warming" the model context with the target speaker's voice characteristics. At inference time the model picks up from these cached states rather than processing audio in real time, which enables the ~200ms first-audio latency.

**Why cloning is restricted on 0.5B:**

The documentation explicitly states: _"To mitigate deepfake risks and ensure low latency for the first speech chunk, voice prompts are provided in an embedded format. For users requiring voice customization, please reach out to our team."_

The tool to create new `.pt` voice files from arbitrary audio is not included in this repository. It is internal to Microsoft. There is no `torch.save` call anywhere in the public codebase that produces the expected format.

**Available preset voices (25 total):**

English: Carter (man), Davis (man), Emma (woman), Frank (man), Grace (woman), Mike (man)
Plus experimental multilingual voices in: German, French, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish (download via `demo/download_experimental_voices.sh`)

---

### 2c. VibeVoice-ASR-7B - Not TTS, No Voice Cloning

ASR is a speech-to-text model. It produces transcripts with speaker diarization and timestamps. It does not generate speech and has no cloning capability.

---

## 3. What is LoRA, and Is It Voice Cloning?

**LoRA is NOT voice cloning.** It is domain adaptation fine-tuning for the **ASR model only**.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that inserts small trainable rank-decomposition matrices into a frozen pre-trained model. It allows adapting a model to a new domain with far fewer trainable parameters and GPU memory than full fine-tuning.

**What LoRA is used for in this repo:**

The `finetuning-asr/` directory provides LoRA fine-tuning scripts for `VibeVoice-ASR-7B`. The purpose is to improve **speech recognition accuracy** for:
- Custom hotwords and technical terminology
- Specific speaker characteristics (recognition accuracy, not synthesis)
- Domain-specific audio (e.g., medical, legal, call center)
- Non-standard accents or speaking styles

**What LoRA is NOT:**

- It does not teach the model to speak in a new voice
- It does not enable TTS or synthesis of any kind
- It is applied to the ASR model, not the TTS models

**Training data format for ASR LoRA:**

```json
{
  "audio_duration": 351.73,
  "audio_path": "recording.mp3",
  "segments": [
    {"speaker": 0, "text": "Transcribed text...", "start": 0.0, "end": 38.68}
  ],
  "customized_context": ["TechCorp", "product name"]
}
```

**Training command:**

```bash
torchrun --nproc_per_node=1 finetuning-asr/lora_finetune.py \
    --model_path microsoft/VibeVoice-ASR \
    --data_dir ./my_dataset \
    --output_dir ./output \
    --lora_r 16 --lora_alpha 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --bf16
```

---

## 4. How to Clone a Voice Locally for Research

Given the current state of the repo, there are two practical approaches for local research:

### Option A: Use VibeVoice-TTS-1.5B with Voice Prompts (Most Capable)

The model weights are available on HuggingFace. The inference code is present in this repo. This is the recommended path for voice cloning research.

**Steps:**

1. Install the package:
   ```bash
   git clone https://github.com/microsoft/VibeVoice.git
   cd VibeVoice
   pip install -e .
   ```

2. Prepare a reference audio clip of the target voice (5-30 sec, clean speech, 24kHz WAV recommended).

3. Write a multi-speaker script with `Speaker 0:` labels. If cloning a single voice, use one speaker.

4. Load the processor and model, pass the reference audio as `voice_samples`, and generate.

5. Refer to community discussions at `microsoft/VibeVoice-1.5B` on HuggingFace for working inference examples (community has shared working scripts in discussions #12 and elsewhere).

**Tips for best clone quality:**
- Use clean, noise-free audio with no background music
- 10-20 seconds of audio is a good target (shorter may miss voice characteristics; longer wastes context)
- Avoid clips with multiple speakers
- English prompts produce the most reliable results
- Try multiple samples; results vary per run (the model is probabilistic)
- Increasing CFG scale (if exposed) can strengthen voice conditioning

### Option B: Streaming 0.5B with Custom .pt Files

The `.pt` format has been fully reverse-engineered from the codebase. A complete
implementation guide and working Python script are available at:

**`docs/vibevoice-custom-voice-pt-creation.md`**

#### What was discovered

The `.pt` file is a `torch.save`d Python dict with exactly four keys:

| Key | Contents |
|---|---|
| `lm` | `BaseModelOutputWithPast` from a single prefill forward pass through the lower LM layers |
| `tts_lm` | `VibeVoiceCausalLMOutputWithPast` from the upper TTS LM layers |
| `neg_lm` | Same as `lm` but for a single-token negative conditioning input |
| `neg_tts_lm` | Same as `tts_lm` but for negative conditioning |

Each object carries `.last_hidden_state` (shape `(1, S, 896)` in bfloat16) and
`.past_key_values` (a `DynamicCache` covering all transformer layers up to that split).

#### How the positive conditioning is built

The voice reference audio is processed through a three-stage pipeline:

1. **Audio preprocessing**: resample to 24 kHz mono, apply dBFS normalisation (target -25 dBFS).
2. **Acoustic encoding**: run through the model's VAE encoder (`model.acoustic_tokenizer.encode()`), scale the latent with the model's `speech_scaling_factor` and `speech_bias_factor` buffers, project to LM hidden dimension via `model.acoustic_connector`.
3. **Token sequence prefill**: the full prompt sequence is:

   ```
   [system_prompt]
   " Voice input:\n"
   " Speaker 0:" <|vision_start|> [<|vision_pad|>]*N_vae <|vision_end|> "\n"
   " Text input:\n"
   " Speech output:\n"
   <|vision_start|>
   ```

   where N_vae = ceil(audio_samples / 3200). The acoustic embeddings are spliced into
   the VAE-placeholder positions before the forward passes.

#### How the negative conditioning is built

The negative branch uses a single `<|image_pad|>` token (token ID 151655, the
tokenizer's `pad_id`). The same two forward passes (`forward_lm` / `forward_tts_lm`)
are run on this single-token input, producing KV-caches of sequence length 1. This
seeding value acts as a "null" signal for the Classifier-Free Guidance (CFG) formula
used during speech diffusion sampling.

#### Caveats

- There is no `torch.save` call anywhere in the public codebase; the prefill tool is
  internal to Microsoft. The reverse-engineered script in the linked doc is a
  research-grade starting point that may require debugging against the exact
  acoustic tokenizer API exposed by the checkpoint.
- Voice quality is best with 10-20 seconds of clean, noise-free, single-speaker
  English audio at 24 kHz.
- The reference clip's text content is irrelevant; only acoustic characteristics are
  captured.

For research purposes, Option A (1.5B TTS model) remains the most validated path.
Option B is now also viable for researchers willing to verify the prefill script against
the live model checkpoint.

---

## 5. The 1.5B Model on HuggingFace

The model weights at `microsoft/VibeVoice-1.5B` are still live as of the time of this writing. What was removed:
- The installation and usage instructions in `docs/vibevoice-tts.md` (replaced with "Disabled due to widespread misuse")
- The mention of the `VibeVoice-Large` model (also removed, listed as "Disabled")

What remains:
- Full model weights on HuggingFace
- All inference code in this repo (`vibevoice/modular/modeling_vibevoice.py`, `vibevoice/processor/vibevoice_processor.py`, etc.)
- The paper describing the architecture (arxiv 2508.19205)
- Community usage examples in HuggingFace model card discussions

The `VibeVoice-Large` model (32K context, Qwen2.5-7B backbone, ~45 min generation, more stable) is listed as "Disabled" in the table and its weights are not publicly available.

---

## 6. VibeVoiceFusion Integration

VibeVoiceFusion (`https://github.com/zhao-kun/VibeVoiceFusion`) is an open-source full-stack web application for multi-speaker TTS generation and LoRA fine-tuning built on top of VibeVoice. The following is based on direct inspection of its source code (453 stars, 56 forks, 283+ commits; MIT licensed).

**Quick start:**
```bash
docker pull zhaokundev/vibevoicefusion
docker run -d --gpus all -p 9527:9527 \
  -v $(pwd)/workspace:/workspace/zhao-kun/vibevoice/workspace \
  zhaokundev/vibevoicefusion:latest
# Access at http://localhost:9527
```

### Real Tech Stack

**Backend:** Python 3.9+, Flask 3.0+, Flask-CORS, PyTorch, HuggingFace Transformers 4.51.3 (pinned), bitsandbytes, TensorBoard 2.20.0

**Frontend:** Next.js 15.5 (App Router), React 19, TypeScript 5, Tailwind CSS 4

**Infra:** Docker + Docker Compose; single-port deployment (port 9527 serves both static frontend and REST API)

**Target model:** VibeVoice-7B (bf16 or float8_e4m3fn), NOT the 1.5B or 0.5B models

### Real Architecture

VibeVoiceFusion ships its **own extended copy** of the `vibevoice` package alongside the Flask backend. It is a fork rather than a thin downstream consumer—it adds several modules that are absent from `microsoft/VibeVoice`:

| Module in VibeVoiceFusion | Status in microsoft/VibeVoice |
|---|---|
| `vibevoice/modular/modeling_vibevoice_inference.py` | **Missing** — upstream has `VibeVoiceForConditionalGeneration` in `modeling_vibevoice.py` |
| `vibevoice/modular/custom_offloading_utils.py` | **Missing** |
| `vibevoice/modular/adaptive_offload.py` | **Missing** |
| `vibevoice/training/trainer.py` (TTS LoRA) | **Missing** — upstream only has `finetuning-asr/` for ASR LoRA |
| `vibevoice/lora/lora_network.py` | **Missing** |
| `config/configuration_vibevoice.py` (with `DEFAULT_CONFIG`, `InferencePhase`) | **Missing** — upstream config lives at `vibevoice/modular/configuration_vibevoice.py` |
| `vibevoice/modular/modeling_vibevoice.py` | Present in both (may have diverged) |
| `vibevoice/processor/vibevoice_processor.py` | Present in both (may have diverged) |

**The central inference class** used by VibeVoiceFusion is `VibeVoiceForConditionalInference` (from `vibevoice.modular.modeling_vibevoice_inference`), not `VibeVoiceForConditionalGeneration` from the upstream repo. Key differences:

- Loaded via `from_pretrain(model_path, config, device, offload_config, lora_model_path, lora_weight)` — note `from_pretrain` (not `from_pretrained`); takes a local `.safetensors` path, not a HuggingFace hub ID
- Accepts an `OffloadConfig` dataclass for GPU layer offloading
- Supports dynamic LoRA injection at inference time via `lora_model_path` and `lora_weight` arguments

### Flask REST API Routes

All API routes are registered at `/api/v1` (via `api_bp`). An OpenAI-compatible endpoint lives separately at `/v1`.

**Projects** (`/api/v1/projects`):
- `GET /projects` — list all projects
- `POST /projects` — create project (name: must start with letter, alphanum/hyphen/underscore/space only)
- `GET /projects/<project_id>` — get project
- `PUT /projects/<project_id>` — update name/description
- `DELETE /projects/<project_id>` — delete project and its workspace directory

**Speakers / voice files** (`/api/v1/projects/<project_id>/speakers`):
- `GET /speakers` — list speakers with voice file info
- `POST /speakers` — add speaker (multipart form: optional `description` + required `voice_file`; accepts WAV/MP3/M4A/FLAC/WebM; saved with UUID filename)
- `GET /speakers/<speaker_id>` — get speaker
- `PUT /speakers/<speaker_id>` — update metadata
- `DELETE /speakers/<speaker_id>` — delete speaker and voice file; remaining speakers are reindexed
- `GET /speakers/<speaker_id>/voice` — download voice file
- `PUT /speakers/<speaker_id>/voice` — replace voice file (ID unchanged)
- `POST /speakers/from-preset` — create speaker from an existing preset voice
- `POST /speakers/<speaker_id>/voice/trim` — trim audio to range (params: `start_time`, `end_time`)

**Dialog Sessions** (`/api/v1/projects/<project_id>/sessions`):
- `GET /sessions` — list sessions
- `POST /sessions` — create session
- `GET /sessions/<session_id>` — get session
- `PUT /sessions/<session_id>` — update text/metadata
- `DELETE /sessions/<session_id>` — delete
- `GET /sessions/<session_id>/text` — fetch dialog text content
- `GET /sessions/<session_id>/download` — download dialog text file

**Generation** (`/api/v1/projects/<project_id>/generations`):
- `POST /generations` — start generation; body params: `dialog_session_id`, `seeds` (list), `cfg_scale`, `model_dtype` (`bfloat16` | `float8_e4m3fn`), `attn_implementation`, `lora_model_path`, `lora_weight`, `batch_size`, `offloading` (object: `type` = `preset` with value `balanced`/`aggressive`/`extreme`, or `type` = `manual` with `num_layers_on_gpu` 1-28)
- `GET /generations` — list generations (stale PENDING auto-marked FAILED)
- `GET /generations/current` — any actively running generation (global, across all projects)
- `GET /generations/<request_id>` — get generation details
- `GET /generations/<request_id>/download` — stream audio inline (`?download=true` forces attachment)
- `GET /generations/<request_id>/items/<item_index>/download` — download individual item from batch
- `DELETE /generations/<request_id>` — delete generation and audio file
- `POST /generations/batch-delete` — bulk delete (body: `request_ids` array)

**Quick Generate** (`/api/v1/quick-generate`) — no project required:
- `POST /quick-generate` — start generation (multipart: up to 4 voice files (WAV/MP3/M4A/FLAC/WebM/OGG) + `text`; params: `batch_size` 1-20, `cfg_scale`, offloading config)
- `GET /quick-generate/<request_id>` — get status
- `GET /quick-generate/current` — get active task
- `GET /quick-generate/history` — list history (pagination: `limit`, `offset`)
- `GET /quick-generate/<request_id>/download` — download first audio file
- `GET /quick-generate/<request_id>/items/<item_index>/download` — download by index
- `DELETE /quick-generate/<request_id>` — delete generation
- `GET /quick-generate/<request_id>/voice/preview` — preview first reference voice
- `GET /quick-generate/<request_id>/voice/<voice_index>/preview` — preview voice by index (0-3)

**Preset Voices** (`/api/v1/preset-voices`):
- `GET /preset-voices` — list with pagination and filters (`language`, `gender`, `has_bgm`, `offset`, `limit`)
- `POST /preset-voices` — upload new preset (multipart: voice file)
- `GET /preset-voices/<filename>` — get preset metadata
- `DELETE /preset-voices/<filename>` — delete preset
- `POST /preset-voices/batch-delete` — bulk delete
- `GET /preset-voices/languages` — list available language codes
- `GET /preset-voices/<filename>/preview` — stream WAV for playback
- Filename convention: `{language}-{name}_{gender}[_bgm].wav`

**Training / LoRA** (`/api/v1/projects/<project_id>/training`):
- `POST /training` — create training job (body: `job_name`, training config via `TrainConfig.from_dict()`)
- `GET /training` — list jobs with states and LoRA files
- `GET /training/current` — active job with live metrics
- `GET /training/<job_id>` — job details
- `DELETE /training/<job_id>` — delete completed jobs only
- `POST /training/batch-delete` — bulk delete
- `GET /training/<job_id>/lora/<filename>` — download a LoRA checkpoint file
- `GET /training/lora-files` — list all LoRA files in project output dir
- `GET /training/<job_id>/metrics` — TensorBoard metrics (params: `max_points`, `metrics` = `loss`/`learning_rate`/`timing`/`all`)

**Tasks:** `GET /api/v1/tasks` and related endpoints for the async task manager.

**OpenAI-compatible** (`/v1`):
- `POST /v1/audio/speech` — OpenAI TTS API shape; Bearer auth required
  - Request: `model` (required), `input` (required, max 4096 chars), `voice` (required, preset voice name), `response_format` (optional: wav/mp3/flac/opus/aac/pcm), `speed` (accepted but ignored)
  - Model mapping: `vibevoice-7b` → `bf16`, `vibevoice-7b-hd` → `float8_e4m3fn`, `tts-1` → `bf16`, `tts-1-hd` → `float8_e4m3fn`
  - Returns binary audio with appropriate Content-Type
- `GET /v1/models` — list available models

**Health:** `GET /health` — service status and version.

### How VibeVoiceFusion Calls VibeVoice (Actual Code Path)

```
Flask API endpoint
  -> VoiceGenerationService.generation()  (backend/services/voice_gerneration_service.py)
  |    or QuickGenerateService.start_generation()  (backend/services/quick_generate_service.py)
  -> InferenceBase.create(offload_config, fake=False)  (backend/inference/inference.py)
  -> InferenceEngine._load_model()
       # lazy import to avoid circular imports:
       from vibevoice.modular.modeling_vibevoice_inference import (
           VibeVoiceForConditionalInference, VibeVoiceGenerationOutput
       )
       from vibevoice.modular.custom_offloading_utils import OffloadConfig
       from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
       from config.configuration_vibevoice import DEFAULT_CONFIG, VibeVoiceConfig, InferencePhase

       model = VibeVoiceForConditionalInference.from_pretrain(
           model_path,        # abs path to vibevoice7b_bf16.safetensors or
                              #   vibevoice7b_float8_e4m3fn.safetensors
           config,            # VibeVoiceConfig instance built from DEFAULT_CONFIG
           device="cuda",
           offload_config=offload_config,   # OffloadConfig dataclass
           lora_model_path=lora_model_path, # optional .safetensors LoRA path
           lora_weight=lora_weight          # float, default 1.0
       )
  -> InferenceEngine.run_inference()  (visitor pattern for progress callbacks)
  -> audio file saved to disk
  -> API returns file path for /download endpoint
```

**How voice files are passed to the model:**
- Project-based generation: `SpeakerService` stores one UUID-named audio file per speaker. Before inference, service resolves speaker IDs to absolute file paths.
- Quick-generate: up to 4 uploaded voice files saved with UUID names; paths resolved as `[str(voices_dir / vf) for vf in voice_files]`.
- Voice paths are passed as a list directly to `VibeVoiceForConditionalInference` (which handles voice encoding internally), NOT passed to `VibeVoiceProcessor` via `voice_samples=` as in the upstream 1.5B pattern.

**LoRA training (what it actually trains):**
- `vibevoice.training.trainer.VibeVoiceTrainer` trains `VibeVoiceForConditionalInference` (7B TTS model).
- `TrainConfig` key fields: `lora_dim`, `lora_alpha`, `lora_dropout`, `multiplier`, `epochs`, `batch_size`, `learning_rate`, `gradient_accumulation_steps` (default 16), `dtype`, `optimizer_type` (default `AdamW8bit`), `speech_compress_ratio` (default 3200), `diffusion_loss_weight`, `ce_loss_weight`, `save_model_per_num_epoch` (default 10).
- This is **TTS LoRA** (voice style fine-tuning on the 7B model), entirely separate from the ASR LoRA in the upstream `finetuning-asr/` directory.
- LoRA network class: `LoRANetwork` in `vibevoice/lora/lora_network.py`.
- LoRA checkpoints saved to: `{workspace}/{project}/training/lora_output/{job_name}/` as `.safetensors` files.

### Offloading System

VibeVoiceFusion adds a custom GPU layer offloading system (`custom_offloading_utils.py`, `LayerOffloader` class) absent from the upstream repo. `OffloadConfig` key fields: `enabled`, `num_layers_on_gpu` (default 8), `offload_prediction_head`, `pin_memory`, `prefetch_next_layer`, `async_transfer`, `cache_clear_interval` (default 50).

Named presets (from `backend/inference/inference.py` `OFFLOAD_PRESETS`):

| Preset | GPU layers | CPU layers | VRAM savings | Speed penalty |
|---|---|---|---|---|
| `balanced` | 12 | 16 | ~5 GB | ~2x slower |
| `aggressive` | 8 | 20 | ~6 GB | ~2.5x slower |
| `extreme` | 4 | 24 | ~7 GB | ~3.5x slower |
| none | all | 0 | 0 | baseline (needs 11-14 GB) |

~310 MB VRAM savings per offloaded layer. Transfer overhead: ~37.8 ms CPU→GPU, ~35.8 ms GPU→CPU per layer. Async prefetching is supported to hide transfer latency.

### Gaps and Incompatibilities with the Upstream VibeVoice Repo

Running VibeVoiceFusion against unmodified `microsoft/VibeVoice` will fail at import time:

| VibeVoiceFusion import | Problem |
|---|---|
| `from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference` | Module does not exist in upstream; upstream equivalent is `VibeVoiceForConditionalGeneration` in `modeling_vibevoice.py` |
| `from vibevoice.modular.custom_offloading_utils import OffloadConfig` | Module does not exist in upstream |
| `from config.configuration_vibevoice import DEFAULT_CONFIG, InferencePhase` | `config/` package does not exist in upstream; `VibeVoiceConfig` is at `vibevoice/modular/configuration_vibevoice.py` |
| `from vibevoice.training.trainer import TrainConfig` | `vibevoice/training/` does not exist in upstream |
| `VibeVoiceForConditionalInference.from_pretrain(local_path, config, ...)` | Upstream uses `VibeVoiceForConditionalGeneration.from_pretrained(hub_id, ...)` — different class, different method name, different signature |
| Model files: `vibevoice7b_bf16.safetensors`, `vibevoice7b_float8_e4m3fn.safetensors` | References locally-stored 7B model files. Microsoft removed 7B weights from public access; only the 1.5B model is publicly available at `microsoft/VibeVoice-1.5B` on HuggingFace. |

**The most critical gap:** VibeVoiceFusion targets the publicly-removed 7B model. The upstream repo's publicly available weights are the 1.5B model. The 7B weights must be obtained separately (e.g., community mirrors).

### How to Wire VibeVoiceFusion to the Latest VibeVoice Backend

Because VibeVoiceFusion is effectively a fork, updating requires syncing specific files rather than a simple version bump.

**Step 1: Sync the shared files**

Pull the latest versions of these files from `microsoft/VibeVoice` into VibeVoiceFusion and check for API changes before merging:
- `vibevoice/modular/modeling_vibevoice.py`
- `vibevoice/processor/vibevoice_processor.py`
- `vibevoice/modular/configuration_vibevoice.py`

Pay particular attention to changes in `VibeVoiceForConditionalGeneration.generate()` and in `VibeVoiceProcessor.__call__()`, especially the `voice_samples=` parameter shape.

**Step 2: Adapt `VibeVoiceForConditionalInference` for the 1.5B model**

The upstream 1.5B model uses the standard HuggingFace loading pattern:
```python
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

model = VibeVoiceForConditionalGeneration.from_pretrained(
    "microsoft/VibeVoice-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

To adapt VibeVoiceFusion's `from_pretrain()` shim: wrap `from_pretrained(local_dir)` (local directory containing safetensors + config.json), then install the `LayerOffloader` hooks post-load. The 1.5B model has fewer transformer layers than 7B, so offloading presets need recalibration.

**Step 3: Wire voice file paths to `VibeVoiceProcessor.voice_samples`**

The upstream processor accepts voice cloning samples as:
```python
inputs = processor(
    text=script,
    voice_samples=voice_paths,  # list of str paths or numpy arrays, one per speaker
    return_tensors="pt"
)
outputs = model.generate(**inputs)
```

VibeVoiceFusion currently passes voice paths into `VibeVoiceForConditionalInference` (which handles encoding internally). To use the upstream pattern, update `InferenceEngine.run_inference()` to call `VibeVoiceProcessor.__call__` with `voice_samples=voice_paths`, then call `model.generate()`.

**Step 4: Update model file references**

In `backend/config.py` and `backend/services/training_service.py`, replace `vibevoice7b_bf16.safetensors` / `vibevoice7b_float8_e4m3fn.safetensors` with paths pointing to the 1.5B model directory.

**Step 5: Keep TTS LoRA code as-is, or replace with ASR LoRA scripts**

There is no TTS LoRA training code in the upstream repo. VibeVoiceFusion's `vibevoice/training/` and `vibevoice/lora/` are custom additions. If TTS LoRA is needed, preserve these files. If only ASR fine-tuning is needed, use `finetuning-asr/lora_finetune.py` from the upstream instead.

**Step 6: Do not use `demo/web/app.py` as VibeVoiceFusion's TTS backend**

The upstream `demo/web/app.py` is a FastAPI WebSocket server for the Realtime-0.5B streaming model only. VibeVoiceFusion uses a completely different architecture: Flask REST + async task queue + file-based audio delivery. There is no WebSocket TTS path in VibeVoiceFusion. These two systems are architecturally incompatible and cannot be directly merged without significant redesign.

### Actual Architecture Diagram

```
   User Browser
        |
        | HTTP (REST JSON + static frontend assets)
        v
  Next.js 15.5 frontend   <--->   Flask 3.0 backend  (port 9527)
  (React 19, TypeScript 5,              |
   Tailwind CSS 4)                      | blueprints registered at /api/v1
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
             Task Manager        InferenceEngine      TrainingService
             (async queue;       VibeVoiceFor           VibeVoiceTrainer
              one job at a       ConditionalInference   (TTS LoRA on 7B;
              time globally)     .from_pretrain()        vibevoice/training/
                    |            local safetensors        trainer.py)
                    |                   |                   |
                    |            LayerOffloader      LoRANetwork
                    |            (custom_offloading_  (vibevoice/lora/
                    |             utils.py; NOT in    lora_network.py;
                    |             upstream repo)      NOT in upstream repo)
                    |
             TensorBoard (training metrics at /training/<job_id>/metrics)
```

---

## 7. Summary of Answers to Key Questions

| Question | Answer |
|---|---|
| Does this repo support voice cloning? | Yes, via the TTS 1.5B model (voice_samples= parameter). Code is present; usage docs were removed. |
| How do I clone a voice? | Pass a reference audio clip as voice_samples= to VibeVoiceProcessor; the model conditions synthesis on that voice. |
| Is the 1.5B model actually disabled? | Weights are still on HuggingFace. Only the official usage instructions were removed. Community has working examples. |
| What is LoRA for? | Fine-tuning the ASR model for domain accuracy. NOT for voice cloning or TTS. |
| Does the 0.5B streaming model do cloning? | No. It uses pre-computed .pt voice files (fixed speaker set). Custom voice creation is not publicly available. |
| Does the 1.5B model do cloning? | Yes - it is specifically designed for zero-shot multi-speaker TTS with voice prompting. |
| Can VibeVoiceFusion use this repo? | Not directly — it is a fork with its own extended vibevoice/ package. It targets the (removed) 7B model and uses classes (VibeVoiceForConditionalInference, OffloadConfig) absent from the upstream. Adaptation requires syncing shared files and bridging the from_pretrain() / from_pretrained() API difference. See Section 6 for the full gap analysis and migration steps. |

---

## 8. Responsible Use Notes

This documentation is intended for research purposes only. Key guidelines:

- The model is **not cleared for commercial use** without further testing and safety evaluation.
- Generated speech should always be disclosed as AI-generated when shared.
- Voice cloning without consent of the voice owner raises serious ethical and legal issues in most jurisdictions.
- Microsoft disabled the 1.5B usage docs specifically due to impersonation misuse. Researchers should take this seriously.
- Do not use these capabilities for fraud, disinformation, or harassment.
