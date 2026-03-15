> **Community Fork** — This is [davidamacey/VibeVoice](https://github.com/davidamacey/VibeVoice), a fork of [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) that restores the TTS-1.5B inference code, adds voice cloning, a LoRA fine-tuning pipeline, an extended web UI, and a full test suite. All original model weights remain on the [Microsoft HuggingFace collection](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f). Attribution and credit for the original research belongs entirely to Microsoft Research.

<div align="center">

## 🎙️ VibeVoice: Open-Source Frontier Voice AI
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=githubpages)](https://microsoft.github.io/VibeVoice)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
[![TTS Report](https://img.shields.io/badge/TTS-Report-red?logo=arxiv)](https://arxiv.org/pdf/2508.19205)
[![ASR Report](https://img.shields.io/badge/ASR-Report-yellow?logo=arxiv)](https://arxiv.org/pdf/2601.18184)
[![Colab](https://img.shields.io/badge/StreamingTTS-Colab-green?logo=googlecolab)](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb)
[![ASR Playground](https://img.shields.io/badge/ASR-Playground-6F42C1?logo=gradio)](https://aka.ms/vibevoice-asr)

</div>


<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/VibeVoice_logo_white.png">
  <img src="Figures/VibeVoice_logo.png" alt="VibeVoice Logo" width="300">
</picture>
</div>

<div align="left">

<h3>📰 News</h3>

<strong>2026-03-15 (this fork): The TTS-1.5B inference code removed by Microsoft has been restored and extended. This fork adds <a href="docs/vibevoice-1.5b-inference.md">zero-shot voice cloning</a>, a LoRA fine-tuning pipeline (<a href="finetuning-tts/README.md">finetuning-tts/</a>), an extended web UI with a <code>/generate</code> endpoint, CPU layer offloading for low-VRAM GPUs, and a full pytest test suite. See the <a href="docs/vibevoice-tts.md">TTS documentation</a> to get started.</strong>

<strong>2026-03-06: 🚀 VibeVoice ASR is now part of a <a href="https://github.com/huggingface/transformers/releases/tag/v5.3.0">Transformers release</a>! You can now use our speech recognition model directly through the Hugging Face Transformers library for seamless integration into your projects.</strong>

<strong>2026-01-21: 📣 We open-sourced <a href="docs/vibevoice-asr.md"><strong>VibeVoice-ASR</strong></a>, a unified speech-to-text model designed to handle 60-minute long-form audio in a single pass, generating structured transcriptions containing Who (Speaker), When (Timestamps), and What (Content), with support for User-Customized Context. Try it in [Playground](https://aka.ms/vibevoice-asr)</strong>.
- ⭐️ VibeVoice-ASR is natively multilingual, supporting over 50 languages — check the [supported languages](docs/vibevoice-asr.md#language-distribution) for details.
- 🔥 The VibeVoice-ASR [finetuning code](finetuning-asr/README.md) is now available!
- ⚡️ **vLLM inference** is now supported for faster inference; see [vllm-asr](docs/vibevoice-vllm-asr.md) for more details.
- 📑 [VibeVoice-ASR Technique Report](https://arxiv.org/pdf/2601.18184) is available.

2025-12-16: 📣 We added experimental speakers to <a href="docs/vibevoice-realtime-0.5b.md"><strong>VibeVoice‑Realtime‑0.5B</strong></a> for exploration, including multilingual voices in nine languages (DE, FR, IT, JP, KR, NL, PL, PT, ES) and 11 distinct English style voices. [Try it](docs/vibevoice-realtime-0.5b.md#optional-more-experimental-voices). More speaker types will be added over time.

2025-12-03: 📣 We open-sourced <a href="docs/vibevoice-realtime-0.5b.md"><strong>VibeVoice‑Realtime‑0.5B</strong></a>, a real‑time text‑to‑speech model that supports streaming text input and robust long-form speech generation. Try it on [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb).


2025-09-05 (upstream): The original Microsoft repository removed the VibeVoice-TTS inference code citing misuse concerns. The model weights remain publicly available on Hugging Face. This fork restores the inference code for research and development use — please use responsibly.


2025-08-25: 📣 We open-sourced <a href="docs/vibevoice-tts.md"><strong>VibeVoice-TTS</strong></a>, a long-form multi-speaker text-to-speech model that can synthesize speech up to 90 minutes long with up to 4 distinct speakers.

</div>

## Overview

VibeVoice is a **family of open-source frontier voice AI models** that includes both Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) models. 

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of **7.5 Hz**. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

For more information, demos, and examples, please visit our [Project Page](https://microsoft.github.io/VibeVoice).


<div align="center">

| Model |   Weight | Quick Try |
|-------|--------------|---------|
| VibeVoice-ASR-7B | [HF Link](https://huggingface.co/microsoft/VibeVoice-ASR) |  [Playground](https://aka.ms/vibevoice-asr) |
| VibeVoice-TTS-1.5B | [HF Link](https://huggingface.co/microsoft/VibeVoice-1.5B) | [Inference Guide](docs/vibevoice-1.5b-inference.md) |
| VibeVoice-Realtime-0.5B | [HF Link](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb) |

</div>

## Models


### 1. 📖 [VibeVoice-ASR](docs/vibevoice-asr.md) - Long-form Speech Recognition

**VibeVoice-ASR** is a unified speech-to-text model designed to handle **60-minute long-form audio** in a single pass, generating structured transcriptions containing **Who (Speaker), When (Timestamps), and What (Content)**, with support for **Customized Hotwords**.

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.

[📖 Documentation](docs/vibevoice-asr.md) | [🤗 Hugging Face](https://huggingface.co/microsoft/VibeVoice-ASR) | [🎮 Playground](https://aka.ms/vibevoice-asr) | [🛠️ Finetuning](finetuning-asr/README.md) |  [📊 Paper](docs/VibeVoice-ASR-Report.pdf)


<p align="center">
  <img src="Figures/DER.jpg" alt="DER" width="50%"><br>
  <img src="Figures/cpWER.jpg" alt="cpWER" width="50%"><br>
  <img src="Figures/tcpWER.jpg" alt="tcpWER" width="50%">
</p>


<div align="center" id="vibevoice-asr">

https://github.com/user-attachments/assets/acde5602-dc17-4314-9e3b-c630bc84aefa

</div>
<br>

### 2. 🎙️ [VibeVoice-TTS](docs/vibevoice-tts.md) - Long-form Multi-speaker TTS

**Best for**: Long-form conversational audio, podcasts, multi-speaker dialogues

- **⏱️ 90-minute Long-form Generation**:
  Synthesizes conversational/single-speaker speech up to **90 minutes** in a single pass, maintaining speaker consistency and semantic coherence throughout.

- **👥 Multi-speaker Support**:
  Supports up to **4 distinct speakers** in a single conversation, with natural turn-taking and speaker consistency across long dialogues.

- **🎭 Expressive Speech**:
  Generates expressive, natural-sounding speech that captures conversational dynamics and emotional nuances.

- **🌐 Multi-lingual Support**:
  Supports English, Chinese and other languages.


[📖 Documentation](docs/vibevoice-tts.md) | [🔧 Inference API](docs/vibevoice-1.5b-inference.md) | [🤗 Hugging Face](https://huggingface.co/microsoft/VibeVoice-1.5B) | [🛠️ LoRA Fine-tuning](finetuning-tts/README.md) | [📊 Paper](https://arxiv.org/pdf/2508.19205)


<div align="center">
  <img src="Figures/VibeVoice-TTS-results.jpg" alt="VibeVoice Results" width="80%">
</div>


**English**
<div align="center">

https://github.com/user-attachments/assets/0967027c-141e-4909-bec8-091558b1b784

</div>


**Chinese**
<div align="center">

https://github.com/user-attachments/assets/322280b7-3093-4c67-86e3-10be4746c88f

</div>

**Cross-Lingual**
<div align="center">

https://github.com/user-attachments/assets/838d8ad9-a201-4dde-bb45-8cd3f59ce722

</div>

**Spontaneous Singing**
<div align="center">

https://github.com/user-attachments/assets/6f27a8a5-0c60-4f57-87f3-7dea2e11c730

</div>


**Long Conversation with 4 people**
<div align="center">

https://github.com/user-attachments/assets/a357c4b6-9768-495c-a576-1618f6275727

</div>





<br>

### 3. ⚡ [VibeVoice-Streaming](docs/vibevoice-realtime-0.5b.md) - Real-time Streaming TTS

VibeVoice-Realtime is a **lightweight real‑time** text-to-speech model supporting **streaming text input** and **robust long-form speech generation**.

- Parameter size: 0.5B (deployment-friendly)
- Real-time TTS (~300 milliseconds first audible latency)
- Streaming text input
- Robust long-form speech generation (~10 minutes)

[📖 Documentation](docs/vibevoice-realtime-0.5b.md) | [🤗 Hugging Face](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | [🚀 Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)


<div align="center" id="generated-example-audio-vibevoice-realtime">

https://github.com/user-attachments/assets/0901d274-f6ae-46ef-a0fd-3c4fba4f76dc

</div>

<br>

## What This Fork Adds

This fork ([davidamacey/VibeVoice](https://github.com/davidamacey/VibeVoice)) restores and extends the TTS-1.5B functionality that was removed from the upstream Microsoft repo. The original ASR and Realtime-0.5B code is unchanged.

| Addition | Description |
|----------|-------------|
| `vibevoice/modular/modeling_vibevoice_inference.py` | `VibeVoiceForConditionalGenerationInference` — HuggingFace-compatible inference class with CFG diffusion, voice cloning, float8 quantization, and CPU layer offloading |
| `vibevoice/modular/custom_offloading_utils.py` | `OffloadConfig` + `LayerOffloader` — async CPU↔GPU layer offloading for low-VRAM inference |
| `vibevoice/utils/` | Shared utilities: memory-efficient safetensors reader, multi-shard loader, LoRA weight merge, seeded DDPM noise, float8 AutoCast wrappers, logger |
| `vibevoice/lora/` | `LoRANetwork` + `LoRAModule` — LoRA targeting Qwen2.5 attention/MLP and diffusion head layers |
| `vibevoice/generation/` | `GenerationVisitor` ABC — step-level callback hooks for progress, timing, and audio file events |
| `vibevoice/modular/word_timing.py` | Word-level timestamp extraction from token windows |
| `demo/tts_1p5b_inference.py` | CLI demo for zero-shot TTS and voice cloning |
| `demo/web/app.py` | Extended web server: `POST /generate` (voice upload + TTS), `GET /models` |
| `finetuning-tts/` | TTS LoRA training script + dataset format docs |
| `tests/` | 55 pytest tests including end-to-end generation against the 1.5B model |
| `docs/vibevoice-1.5b-inference.md` | Full Python API reference for the inference class |

### Quick Start (TTS)

```bash
git clone https://github.com/davidamacey/VibeVoice
cd VibeVoice
pip install -e .

# Download weights from HuggingFace
huggingface-cli download microsoft/VibeVoice-1.5B --local-dir ./models/VibeVoice-1.5B

# Basic TTS
python demo/tts_1p5b_inference.py \
    --model ./models/VibeVoice-1.5B \
    --text "Speaker 0: Hello, this is a test." \
    --output hello.wav

# Voice cloning
python demo/tts_1p5b_inference.py \
    --model ./models/VibeVoice-1.5B \
    --text "Speaker 0: Hello, this is a test." \
    --voice reference.wav \
    --output cloned.wav
```

See [docs/vibevoice-1.5b-inference.md](docs/vibevoice-1.5b-inference.md) for the full Python API.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.



## ⚠️ Risks and Limitations


While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.


We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only. Please use responsibly.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Microsoft/vibevoice&type=date&legend=top-left)
