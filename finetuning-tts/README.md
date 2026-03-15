# VibeVoice TTS LoRA Fine-tuning

This directory contains the script for LoRA (Low-Rank Adaptation) fine-tuning of the VibeVoice 1.5B TTS model.

Unlike the ASR fine-tuning script (which uses HuggingFace PEFT / Trainer), this script uses VibeVoice's own `LoRANetwork` implementation from `vibevoice/lora/`. This targets **both** the Qwen2.5 backbone attention/MLP layers **and** the diffusion prediction head — the layers that control how speech sounds.

> **Note on voice cloning**: LoRA fine-tuning adjusts the model's general speech quality and style for all speakers. It is **not** how voice cloning works. Voice cloning is zero-shot: pass a reference audio via `voice_samples=` to the processor and the model clones that speaker without any training.

## Requirements

```bash
pip install -e .
pip install safetensors torchaudio
```

## Data Format

Training data is a directory of JSON + audio pairs. Each JSON file describes one sample:

```
tts_dataset/
├── sample_001.wav
├── sample_001.json
├── sample_002.wav
├── sample_002.json
└── ...
```

### JSON Format

```json
{
    "audio_path": "sample_001.wav",
    "text": "Hello, this is a training sample."
}
```

The audio file path in `audio_path` is relative to the directory containing the JSON file. Audio is automatically resampled to 24 kHz mono.

## Training

```bash
python finetuning-tts/lora_finetune.py \
    --model_path /mnt/nas/models/vibevoice/VibeVoice-1.5B \
    --data_dir ./tts_dataset \
    --output_dir ./checkpoints/tts_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 3 \
    --batch_size 1 \
    --lr 1e-4
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 16 | LoRA rank (lower = fewer params, higher = more expressive) |
| `--lora_alpha` | 16.0 | LoRA alpha scaling factor |
| `--lora_dropout` | 0.05 | Dropout applied to LoRA layers |
| `--num_epochs` | 3 | Number of training epochs |
| `--batch_size` | 1 | Samples per GPU step |
| `--grad_accum_steps` | 4 | Effective batch = batch_size × grad_accum_steps |
| `--lr` | 1e-4 | Learning rate |
| `--max_audio_seconds` | 20.0 | Skip audio longer than this (seconds) |
| `--save_every_n_epochs` | 1 | Save a checkpoint every N epochs |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

## Output

Checkpoints are saved as safetensors files with the `vibevoice_lora-` key prefix:

```
checkpoints/tts_lora/
├── tts_lora_epoch1.safetensors
├── tts_lora_epoch2.safetensors
├── tts_lora_epoch3.safetensors
└── tts_lora_final.safetensors  ← symlink to the last epoch
```

## Using the Fine-tuned LoRA

Load the LoRA at inference time using `from_pretrained_file()`:

```python
from vibevoice import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

config = VibeVoiceConfig.from_pretrained("/mnt/nas/models/vibevoice/VibeVoice-1.5B")
model = VibeVoiceForConditionalGenerationInference.from_pretrained_file(
    "/mnt/nas/models/vibevoice/VibeVoice-1.5B",
    config=config,
    device="cuda",
    lora_model_path="./checkpoints/tts_lora/tts_lora_final.safetensors",
    lora_weight=1.0,
)
model.eval()
```

Or merge the LoRA directly into the base weights for zero-overhead inference:

```python
from vibevoice.utils.model_utils import merge_lora_weights

model = merge_lora_weights(model, "./checkpoints/tts_lora/tts_lora_final.safetensors")
```

## LoRA Architecture

The `LoRANetwork` targets these layer patterns in the 1.5B model:

- `model.language_model.layers.*.self_attn.{q,k,v,o}_proj` — attention projections
- `model.language_model.layers.*.mlp.{gate,up,down}_proj` — MLP projections
- `model.prediction_head.cond_proj` — diffusion conditioning projection
- `model.prediction_head.layers.*.ffn.{gate,up,down}_proj` — diffusion FFN
- `model.prediction_head.layers.*.adaLN_modulation.1` — adaptive layer norm modulation
