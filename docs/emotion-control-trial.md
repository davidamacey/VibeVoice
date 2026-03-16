# Emotion & Style Control Trial

**Date:** 2026-03-16
**Model tested:** VibeVoice-TTS-1.5B
**Output files:** `test_outputs/emotion_trial/`
**Trial script:** `demo/emotion_trial.py`

---

## What We Tested

Whether emotion and speaking style can be injected into VibeVoice speech generation without retraining, using only prompt-level conditioning. Three approaches were evaluated across two emotions (sad, angry) against a neutral baseline.

### Approaches

| Approach | Description | Works? |
|----------|-------------|--------|
| **Punctuation / natural text** | Commas, ellipsis, `—`, `!` naturally influence pacing | Yes — organic, always on |
| **System prompt injection** | Override `processor.system_prompt` with emotion description before generating | Yes — measurable effect |
| **Inline stage direction** | Prepend `(voice trembling, ...)` to the speaker text | **No** — model reads it aloud as literal speech |
| **`Style:` line (new)** | Insert a `Style: ...` line between voice block and `Text input:` section | Yes — not spoken, conditions LLM |

---

## Key Finding: Inline Prefix Doesn't Work

Prepending a stage direction like `(voice trembling, barely holding back tears)` inside the speaker text field causes the model to **speak those words aloud**. The processor passes all speaker text directly to the speech generation token sequence, so there is no way for the model to distinguish "stage direction" from "words to say" at that position.

**Fix implemented:** Added `style=` parameter to `VibeVoiceProcessor.__call__()` and `_process_single()`. When provided, a ` Style: <text>\n` line is inserted into the prompt between the voice-sample block and the `Text input:` section. This is read by the LLM backbone as conditioning context but is never inside a speaker turn, so it is never spoken.

```python
# Usage
processor(
    text="Speaker 0: I can't believe they're gone...",
    style="deeply sad and grief-stricken, voice slow and heavy with sorrow",
    return_tensors="pt"
)
```

---

## Results

### Sad Emotion
- **Plain vs system-prompt vs Style:** All three sounded noticeably different.
- The conditioning had clear audible effect — pacing, tone, and delivery shifted.
- Likely reason: sad-appropriate words ("gone", "empty", "don't know how") are prosodically neutral; the text alone doesn't force a delivery style, leaving room for conditioning to work.

### Angry Emotion
- **Plain vs system-prompt vs Style:** Very similar across all three.
- Likely reason: the angry text itself (`"completely unacceptable!"`, `"you ignored every single thing I said!"`) already drives the model's LM hidden states into a high-energy register. The conditioning has little additional room to push further.
- **Untested follow-up:** Apply angry style to the neutral weather text — if that sounds angry, it would confirm the conditioning works but the angry text content was already saturating the effect.

### Variability / Seed Control
Initial comparisons were confounded because `get_generator()` in `vibevoice/utils/rand_init.py` creates a global generator once at seed 42 and then **advances its state** on each call. Trials ran sequentially meant each started from a different noise position.

**Fix:** Call `get_generator(seeds=42, force_set=True)` before each trial to reset to the same starting diffusion noise. With a fixed seed, any audible difference between variants is purely from prompt conditioning, not noise luck.

---

## Prompt Structure (for reference)

The full prompt structure that reaches the LLM:

```
[system_prompt]
 Voice input:                         ← optional, if voice_samples provided
  Speaker 0: <audio_tokens>
 Style: <style text>                  ← NEW — optional, injected here
 Text input:
  Speaker 0: <text to speak>
 Speech output:
<speech_start>
```

---

## What Does NOT Work

- **SSML tags** (`<break>`, `<prosody>`) — not parsed, passed as literal text tokens
- **Emotion embedding vectors** — no style/emotion embedding layer in the architecture
- **Pitch / speed / energy sliders** — diffusion head does not expose these parameters
- **Inline stage directions** in speaker text — get spoken aloud
- **LoRA for emotion** — LoRA in this repo targets ASR fine-tuning only; TTS LoRA would need emotion-labeled training data

---

## Things Worth Revisiting

1. **Higher CFG scale for angry** — `cfg_scale` (default 3.0) is the diffusion guidance multiplier. Values of 5–7 push conditioning harder and might reveal more angry effect on already-aggressive text.

2. **Neutral text + angry style** — The definitive test: apply `ANGRY_STYLE` to the weather forecast `NEUTRAL_TEXT`. If it sounds angry, the conditioning works but saturates on angry text. Trial files would be `08_neutral_angry_style.wav` and `09_neutral_angry_sysprompt.wav`.

3. **Combined system prompt + Style line** — Both levers active simultaneously. Not tested; may compound the effect.

4. **VibeVoice-Large** — The 7B Qwen2 backbone model has more capacity to follow nuanced instructions. Style conditioning may be significantly stronger there.

5. **Multiple seeds** — Testing each variant across 3–5 different seeds would give a more statistically robust picture of whether the effect is consistent or seed-dependent.

---

## Files

| File | Description |
|------|-------------|
| `demo/emotion_trial.py` | Trial runner — loads model once, iterates TRIALS list, fixed seed per trial |
| `vibevoice/processor/vibevoice_processor.py` | `style=` parameter added to `__call__` and `_process_single` |
| `vibevoice/utils/rand_init.py` | `get_generator(seeds, force_set)` — use `force_set=True` for reproducible trials |
| `test_outputs/emotion_trial/` | 7 output WAV files from final seeded run |
