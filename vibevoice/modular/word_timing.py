"""Word-level timing utilities for VibeVoice TTS generation.

Extracts word timestamps from the windowed text-to-speech generation process.
No model or torch dependencies — only requires a tokenizer with decode().
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WordTimestamp:
    """A single word with its timing in the generated audio."""
    word: str
    start_time: float  # seconds
    end_time: float    # seconds
    token_ids: List[int] = field(default_factory=list)


def distribute_time_in_window(
    token_ids: List[int],
    start_sample: int,
    end_sample: int,
    tokenizer: Any,
    sample_rate: int = 24000,
) -> List[Dict[str, Any]]:
    """Distribute a window's audio duration across its tokens proportionally by character length.

    Args:
        token_ids: Token IDs in this text window.
        start_sample: Audio sample index where this window starts.
        end_sample: Audio sample index where this window ends.
        tokenizer: Tokenizer with a decode(ids) method.
        sample_rate: Audio sample rate in Hz.

    Returns:
        List of dicts with keys: token_id, text, start_sample, end_sample.
    """
    if not token_ids:
        return []

    texts = [tokenizer.decode([tid]) for tid in token_ids]
    char_lengths = [max(len(t), 1) for t in texts]
    total_chars = sum(char_lengths)
    total_samples = end_sample - start_sample

    result = []
    cursor = start_sample
    for i, (tid, text) in enumerate(zip(token_ids, texts)):
        if i == len(token_ids) - 1:
            # Last token gets remaining samples to avoid rounding gaps
            token_end = end_sample
        else:
            token_duration = int(total_samples * char_lengths[i] / total_chars)
            token_end = cursor + token_duration
        result.append({
            "token_id": tid,
            "text": text,
            "start_sample": cursor,
            "end_sample": token_end,
        })
        cursor = token_end

    return result


def merge_subword_tokens_to_words(
    token_timestamps: List[Dict[str, Any]],
    sample_rate: int = 24000,
) -> List[WordTimestamp]:
    """Merge subword token timestamps into word-level WordTimestamp objects.

    Uses BPE space-prefix boundary detection: Qwen2 tokenizer prepends a space
    character to word-initial tokens. Tokens without a leading space are
    continuations of the previous word (including punctuation).

    Args:
        token_timestamps: Flat list of per-token dicts from distribute_time_in_window.
        sample_rate: Audio sample rate in Hz.

    Returns:
        List of WordTimestamp objects.
    """
    if not token_timestamps:
        return []

    words: List[WordTimestamp] = []
    current_text = ""
    current_ids: List[int] = []
    current_start = 0
    current_end = 0

    for ts in token_timestamps:
        text = ts["text"]
        is_word_start = text.startswith(" ") or text.startswith("\u0120")

        if is_word_start and current_text:
            # Flush previous word
            word_str = current_text.strip()
            if word_str:
                words.append(WordTimestamp(
                    word=word_str,
                    start_time=current_start / sample_rate,
                    end_time=current_end / sample_rate,
                    token_ids=current_ids,
                ))
            current_text = text
            current_ids = [ts["token_id"]]
            current_start = ts["start_sample"]
            current_end = ts["end_sample"]
        else:
            if not current_text:
                current_start = ts["start_sample"]
            current_text += text
            current_ids.append(ts["token_id"])
            current_end = ts["end_sample"]

    # Flush last word
    if current_text:
        word_str = current_text.strip()
        if word_str:
            words.append(WordTimestamp(
                word=word_str,
                start_time=current_start / sample_rate,
                end_time=current_end / sample_rate,
                token_ids=current_ids,
            ))

    return words


def build_word_timestamps(
    window_token_map: List[Dict[str, Any]],
    tokenizer: Any,
    sample_rate: int = 24000,
) -> List[WordTimestamp]:
    """Build word-level timestamps from the per-window token map collected during generation.

    Args:
        window_token_map: List of dicts, each with keys:
            - token_ids: List[int] of text token IDs in that window
            - start_sample: int, audio sample index at window start
            - end_sample: int, audio sample index at window end
        tokenizer: Tokenizer with decode(ids) method.
        sample_rate: Audio sample rate in Hz.

    Returns:
        List of WordTimestamp objects covering the full utterance.
    """
    all_token_timestamps: List[Dict[str, Any]] = []
    for window in window_token_map:
        token_ts = distribute_time_in_window(
            token_ids=window["token_ids"],
            start_sample=window["start_sample"],
            end_sample=window["end_sample"],
            tokenizer=tokenizer,
            sample_rate=sample_rate,
        )
        all_token_timestamps.extend(token_ts)

    return merge_subword_tokens_to_words(all_token_timestamps, sample_rate=sample_rate)


def timestamps_to_json(word_timestamps: List[WordTimestamp]) -> List[Dict[str, Any]]:
    """Convert WordTimestamp objects to JSON-serializable list of dicts."""
    return [
        {
            "word": wt.word,
            "start_time": round(wt.start_time, 4),
            "end_time": round(wt.end_time, 4),
        }
        for wt in word_timestamps
    ]


def timestamps_to_srt(
    word_timestamps: List[WordTimestamp],
    words_per_cue: int = 6,
) -> str:
    """Generate SRT subtitle format string from word timestamps.

    Args:
        word_timestamps: List of WordTimestamp objects.
        words_per_cue: Number of words per subtitle cue.

    Returns:
        SRT formatted string.
    """
    if not word_timestamps:
        return ""

    def _fmt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    cues = []
    for i in range(0, len(word_timestamps), words_per_cue):
        group = word_timestamps[i:i + words_per_cue]
        start = group[0].start_time
        end = group[-1].end_time
        text = " ".join(wt.word for wt in group)
        cue_num = len(cues) + 1
        cues.append(f"{cue_num}\n{_fmt_time(start)} --> {_fmt_time(end)}\n{text}")

    return "\n\n".join(cues) + "\n"
