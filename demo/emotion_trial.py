"""
Emotion Trial — does injecting emotion cues actually change VibeVoice output?

Generates 7 audio files across 3 conditions (neutral, sad, angry), each with
three variants:
  A) plain text, no cue
  B) same text + system-prompt emotion injection
  C) same text + inline stage-direction prefix

Output directory: test_outputs/emotion_trial/
"""

import os, sys, torch
import numpy as np
import scipy.io.wavfile as wavfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = "/mnt/nas/models/vibevoice/VibeVoice-1.5B"
OUT_DIR     = "test_outputs/emotion_trial"
SAMPLE_RATE = 24000
DEVICE      = "cuda:0"
CFG_SCALE   = 3.0
DDPM_STEPS  = 20

# ---------------------------------------------------------------------------
# TEXT CONTENT
# Kept consistent per emotion so we isolate the cue variable.
# ---------------------------------------------------------------------------

NEUTRAL_TEXT = (
    "The weather today is partly cloudy with a chance of rain in the afternoon. "
    "Temperatures will be in the mid-sixties, dropping to the low fifties overnight."
)

SAD_TEXT = (
    "I can't believe they're gone. "
    "Everything reminds me of them. "
    "The house feels so empty now. "
    "I just don't know how I'm supposed to go on without them."
)

ANGRY_TEXT = (
    "This is completely unacceptable! "
    "I told you exactly what needed to be done and you ignored every single thing I said! "
    "This is the last time I put up with this!"
)

# System prompt templates
DEFAULT_SYS = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
SAD_SYS = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker. "
    "The speaker is deeply sad, grieving, and on the verge of tears — "
    "voice should be soft, slow, and heavy with sorrow.\n"
)
ANGRY_SYS = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker. "
    "The speaker is furious and enraged — "
    "voice should be loud, sharp, tense, and full of anger.\n"
)

# Style lines for the new processor style= parameter
SAD_STYLE   = "deeply sad and grief-stricken, voice barely above a whisper, slow and heavy with sorrow"
ANGRY_STYLE = "furious and enraged, voice loud, sharp, and tense with anger"

# ---------------------------------------------------------------------------
# Trial definitions: (filename_stem, text, system_prompt, style)
# ---------------------------------------------------------------------------
TRIALS = [
    # --- NEUTRAL baseline ---
    ("01_neutral_plain",
     NEUTRAL_TEXT, DEFAULT_SYS, None),

    # --- SAD variants ---
    ("02_sad_plain",
     SAD_TEXT,     DEFAULT_SYS, None),
    ("03_sad_sysprompt",
     SAD_TEXT,     SAD_SYS,     None),
    ("04_sad_style",
     SAD_TEXT,     DEFAULT_SYS, SAD_STYLE),

    # --- ANGRY variants ---
    ("05_angry_plain",
     ANGRY_TEXT,   DEFAULT_SYS, None),
    ("06_angry_sysprompt",
     ANGRY_TEXT,   ANGRY_SYS,   None),
    ("07_angry_style",
     ANGRY_TEXT,   DEFAULT_SYS, ANGRY_STYLE),
]


def build_script(text):
    """Wrap text in Speaker 0: format (required by VibeVoiceProcessor)."""
    return f"Speaker 0: {text}"


def run_trial(model, processor, stem, text, sys_prompt, style):
    from vibevoice.utils.rand_init import get_generator
    # Reset to the same seed before every trial so diffusion noise is identical.
    # Any audible difference between variants is then purely from the conditioning.
    get_generator(seeds=42, force_set=True)

    script = build_script(text)
    processor.system_prompt = sys_prompt

    print(f"\n{'='*60}")
    print(f"  {stem}")
    print(f"  sys_prompt excerpt: {sys_prompt[50:90].strip()!r}...")
    print(f"  style: {style!r}")
    print(f"  text[:60]: {text[:60]!r}")
    print(f"{'='*60}")

    inputs = processor(text=script, style=style, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            cfg_scale=CFG_SCALE,
            return_speech=True,
        )

    if not output.speech_outputs or output.speech_outputs[0] is None:
        print(f"  WARNING: no audio produced for {stem}")
        return None

    audio = output.speech_outputs[0].cpu().float().numpy().squeeze()
    # Clamp and convert to int16 for broadest compatibility
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    out_path = os.path.join(OUT_DIR, f"{stem}.wav")
    wavfile.write(out_path, SAMPLE_RATE, audio_int16)
    duration = len(audio) / SAMPLE_RATE
    print(f"  -> saved {duration:.2f}s  {out_path}")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_PATH} ...")
    from vibevoice.modular import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor import VibeVoiceProcessor

    model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
        MODEL_PATH, device=DEVICE, torch_dtype=torch.bfloat16
    )
    model.eval()
    model.set_ddpm_inference_steps(DDPM_STEPS)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)

    print(f"\nRunning {len(TRIALS)} trials -> {OUT_DIR}/\n")
    results = []
    for stem, text, sys_prompt, style in TRIALS:
        path = run_trial(model, processor, stem, text, sys_prompt, style)
        results.append((stem, path))

    print("\n\n--- RESULTS ---")
    for stem, path in results:
        status = "OK" if path else "FAILED"
        print(f"  [{status}] {stem}  {path or ''}")
    print(f"\nAll files in {OUT_DIR}/")
    print("Listen to them and compare — especially 02 vs 03 vs 04, and 05 vs 06 vs 07.")


if __name__ == "__main__":
    main()
