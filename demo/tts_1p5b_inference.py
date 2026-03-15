"""
VibeVoice 1.5B TTS Inference Demo

Demonstrates zero-shot TTS and voice cloning with the 1.5B model backed up on NAS.

Usage:
    # Basic TTS (no voice cloning)
    python demo/tts_1p5b_inference.py --text "Hello, how are you?" --output output.wav

    # Voice cloning from a reference audio file
    python demo/tts_1p5b_inference.py \\
        --text "Hello, how are you?" \\
        --voice /path/to/reference.wav \\
        --output output.wav

    # With layer offloading for low-VRAM GPUs
    python demo/tts_1p5b_inference.py \\
        --text "Hello, how are you?" \\
        --output output.wav \\
        --offload --gpu-layers 12
"""

import argparse
import os
import sys
import torch
import torchaudio

# VibeVoice models are backed up at /mnt/nas/models/vibevoice/VibeVoice-1.5B
DEFAULT_MODEL_PATH = "/mnt/nas/models/vibevoice/VibeVoice-1.5B"
SAMPLE_RATE = 24000


def load_model(model_path: str, device: str = "cuda", offload: bool = False, gpu_layers: int = 12):
    from vibevoice.modular import VibeVoiceForConditionalGenerationInference, OffloadConfig

    print(f"Loading model from {model_path} ...")
    offload_config = None
    if offload:
        offload_config = OffloadConfig(
            enabled=True,
            num_layers_on_gpu=gpu_layers,
            offload_prediction_head=False,
        )

    model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
        model_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model


def load_processor(model_path: str):
    from vibevoice.processor import VibeVoiceProcessor
    return VibeVoiceProcessor.from_pretrained(model_path)


def run_inference(
    model,
    processor,
    text: str,
    voice_path: str = None,
    cfg_scale: float = 3.0,
    ddpm_steps: int = 20,
    device: str = "cuda",
):
    model.set_ddpm_inference_steps(ddpm_steps)

    voice_samples = None
    if voice_path is not None:
        waveform, sr = torchaudio.load(voice_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        voice_samples = [waveform.squeeze(0).numpy()]
        print(f"Voice cloning from: {voice_path}")

    inputs = processor(
        text=text,
        voice_samples=voice_samples,
        return_tensors="pt",
    ).to(device)

    print(f"Generating speech for: {repr(text)}")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            cfg_scale=cfg_scale,
            return_speech=True,
        )

    if output.speech_outputs and output.speech_outputs[0] is not None:
        return output.speech_outputs[0]
    else:
        print("Warning: No audio was generated.")
        return None


def main():
    parser = argparse.ArgumentParser(description="VibeVoice 1.5B TTS Inference")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default=None, help="Path to reference voice audio for cloning")
    parser.add_argument("--output", default="output.wav", help="Output WAV file path")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda / cpu)")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--steps", type=int, default=20, help="DDPM inference steps")
    parser.add_argument("--offload", action="store_true", help="Enable CPU layer offloading")
    parser.add_argument("--gpu-layers", type=int, default=12, help="Layers to keep on GPU when offloading")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model path not found: {args.model}")
        print("Make sure the 1.5B model is available. It was backed up to /mnt/nas/models/vibevoice/VibeVoice-1.5B")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    model = load_model(args.model, device=device, offload=args.offload, gpu_layers=args.gpu_layers)
    processor = load_processor(args.model)

    audio = run_inference(
        model, processor,
        text=args.text,
        voice_path=args.voice,
        cfg_scale=args.cfg_scale,
        ddpm_steps=args.steps,
        device=device,
    )

    if audio is not None:
        audio_cpu = audio.cpu().unsqueeze(0).float()
        torchaudio.save(args.output, audio_cpu, SAMPLE_RATE)
        duration = audio_cpu.shape[-1] / SAMPLE_RATE
        print(f"Saved {duration:.2f}s audio to {args.output}")
    else:
        print("No audio generated.")


if __name__ == "__main__":
    main()
