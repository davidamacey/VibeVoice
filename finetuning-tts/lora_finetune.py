#!/usr/bin/env python
"""
VibeVoice TTS LoRA Fine-tuning Script

Fine-tunes the 1.5B TTS model using the custom LoRANetwork from vibevoice/lora/.
Unlike the ASR script (which uses PEFT/HuggingFace Trainer), this script uses
VibeVoice's own LoRA implementation that targets both the Qwen2.5 backbone AND
the diffusion prediction head — the layers that control how speech sounds.

Dataset format (one JSON per sample, audio file alongside):
    {
        "audio_path": "sample_001.wav",
        "text": "Hello, this is a training sample."
    }

Usage:
    python finetuning-tts/lora_finetune.py \\
        --model_path /mnt/nas/models/vibevoice/VibeVoice-1.5B \\
        --data_dir ./tts_dataset \\
        --output_dir ./checkpoints/tts_lora \\
        --lora_r 16 \\
        --lora_alpha 32 \\
        --num_epochs 3 \\
        --batch_size 2 \\
        --lr 1e-4
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.lora.lora_network import LoRANetwork, create_network

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TTSLoRADataset(Dataset):
    """
    TTS dataset for LoRA fine-tuning.

    Each sample is a JSON file containing:
        {
            "audio_path": "relative/path/to/audio.wav",
            "text": "Transcription of the audio."
        }

    Audio is resampled to 24 kHz mono.
    """

    def __init__(self, data_dir: str, processor: VibeVoiceProcessor,
                 max_audio_seconds: float = 30.0):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_audio_seconds = max_audio_seconds
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} TTS samples from {data_dir}")

    def _load_samples(self) -> List[Dict]:
        samples = []
        for json_path in sorted(self.data_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                audio_path = self.data_dir / data["audio_path"]
                if not audio_path.exists():
                    logger.warning(f"Audio not found: {audio_path}")
                    continue
                samples.append({"audio_path": str(audio_path), "text": data["text"]})
            except Exception as e:
                logger.warning(f"Skipping {json_path}: {e}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        sample = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(sample["audio_path"])
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            waveform = waveform.squeeze(0)

            max_samples = int(self.max_audio_seconds * SAMPLE_RATE)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]

            # Use processor to build model inputs (text + voice sample as target)
            # For TTS training, the voice sample IS the target audio
            inputs = self.processor(
                text=sample["text"],
                voice_samples=[waveform.numpy()],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            inputs["audio"] = waveform
            return inputs
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            return None


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    keys = [k for k in batch[0].keys() if k != "audio"]
    result = {}

    for key in keys:
        tensors = [b[key] for b in batch if key in b]
        if not tensors:
            continue
        if isinstance(tensors[0], torch.Tensor):
            # Pad to longest in batch
            max_len = max(t.shape[-1] for t in tensors)
            padded = []
            for t in tensors:
                pad_size = max_len - t.shape[-1]
                if pad_size > 0 and t.dim() > 0:
                    t = torch.nn.functional.pad(t, (0, pad_size))
                padded.append(t)
            result[key] = torch.stack(padded)
        else:
            result[key] = tensors

    # Pad audio waveforms
    audios = [b["audio"] for b in batch]
    max_audio = max(a.shape[0] for a in audios)
    padded_audio = torch.zeros(len(audios), max_audio)
    for i, a in enumerate(audios):
        padded_audio[i, :a.shape[0]] = a
    result["audio"] = padded_audio

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def setup_model(model_path: str, device: str, dtype: torch.dtype):
    logger.info(f"Loading TTS model from {model_path}")
    processor = VibeVoiceProcessor.from_pretrained(model_path)

    try:
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            attn_implementation="sdpa",
        )

    if device != "cpu" and not hasattr(model, "hf_device_map"):
        model = model.to(device)

    return model, processor


def freeze_non_lora_params(model: nn.Module) -> None:
    """Freeze everything; LoRANetwork.apply_to() will inject trainable params."""
    for param in model.parameters():
        param.requires_grad = False
    # Always keep speech scaling/bias factors trainable as they're tiny
    for name, param in model.named_parameters():
        if "speech_scaling_factor" in name or "speech_bias_factor" in name:
            param.requires_grad = True


def save_lora_weights(network: LoRANetwork, output_dir: str, epoch: int) -> str:
    """Save LoRA weights as safetensors with vibevoice_lora- prefix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = network.state_dict()
    # Ensure keys follow the expected vibevoice_lora- prefix
    prefixed = {}
    for key, value in state_dict.items():
        if not key.startswith("vibevoice_lora-"):
            prefixed[f"vibevoice_lora-{key}"] = value.cpu().float()
        else:
            prefixed[key] = value.cpu().float()

    out_path = output_dir / f"tts_lora_epoch{epoch}.safetensors"
    save_file(prefixed, str(out_path), metadata={"epoch": str(epoch), "multiplier": "1.0"})
    logger.info(f"Saved LoRA weights to {out_path}")
    return str(out_path)


def train(
    model_path: str,
    data_dir: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 1,
    lr: float = 1e-4,
    grad_accum_steps: int = 4,
    max_audio_seconds: float = 20.0,
    save_every_n_epochs: int = 1,
    device: str = "cuda",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model, processor = setup_model(model_path, device, dtype)
    model.train()

    # Freeze base weights — LoRA will inject trainable adapters
    freeze_non_lora_params(model)

    # Build LoRA network targeting attention, MLP, and diffusion head
    lora_network = create_network(
        original_model=model,
        multiplier=1.0,
        network_dim=lora_r,
        network_alpha=lora_alpha,
        neuron_dropout=lora_dropout,
    )
    # Monkey-patch forward methods to include LoRA deltas
    lora_network.apply_to()
    lora_network.to(device)

    trainable_params, lr_descriptions = lora_network.prepare_optimizer_params(learning_rate=lr)
    n_trainable = sum(p.numel() for group in trainable_params for p in group["params"])
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    dataset = TTSLoRADataset(data_dir, processor, max_audio_seconds=max_audio_seconds)
    if len(dataset) == 0:
        logger.error("No training samples found. Check --data_dir.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    logger.info(f"Starting TTS LoRA training: {num_epochs} epochs, {len(dataset)} samples")
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            # Move tensors to device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != "audio"
            }

            try:
                with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                    outputs = model(**inputs)

                loss = outputs.loss
                if outputs.diffusion_loss is not None:
                    loss = loss + outputs.diffusion_loss

                if loss is None:
                    logger.warning(f"Step {global_step}: loss is None, skipping.")
                    continue

                (loss / grad_accum_steps).backward()

            except Exception as e:
                logger.warning(f"Step {global_step} forward/backward failed: {e}")
                optimizer.zero_grad()
                continue

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_network.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 20 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Flush remaining gradients at epoch end
        if global_step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(lora_network.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch} complete | avg_loss={avg_loss:.4f}")

        if epoch % save_every_n_epochs == 0:
            save_lora_weights(lora_network, output_dir, epoch)

    # Always save final
    final_path = save_lora_weights(lora_network, output_dir, epoch=num_epochs)
    logger.info(f"Training complete. Final LoRA saved to {final_path}")

    # Save a symlink / copy as 'tts_lora_final.safetensors' for convenience
    final_link = Path(output_dir) / "tts_lora_final.safetensors"
    if final_link.exists():
        final_link.unlink()
    final_link.symlink_to(Path(final_path).name)
    logger.info(f"Symlink: {final_link} -> {Path(final_path).name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VibeVoice TTS LoRA Fine-tuning")
    parser.add_argument("--model_path", required=True,
                        help="Path to VibeVoice-1.5B checkpoint (HF format or NAS path)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with *.json samples (each referencing an audio file)")
    parser.add_argument("--output_dir", default="./checkpoints/tts_lora",
                        help="Where to save LoRA checkpoints")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_audio_seconds", type=float, default=20.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        max_audio_seconds=args.max_audio_seconds,
        save_every_n_epochs=args.save_every_n_epochs,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
