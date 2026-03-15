"""Unit tests for vibevoice.modular.word_timing utilities."""

import json
import pytest

from vibevoice.modular.word_timing import (
    WordTimestamp,
    distribute_time_in_window,
    merge_subword_tokens_to_words,
    build_word_timestamps,
    timestamps_to_json,
    timestamps_to_srt,
)


# ---------------------------------------------------------------------------
# Mock tokenizer that simulates Qwen2 BPE decode behaviour:
#   - Word-initial tokens are decoded with a leading space (e.g. " Hello")
#   - Continuation tokens (punctuation, subword) have no leading space
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock with per-ID decode mapping."""

    def __init__(self, vocab: dict):
        self._vocab = vocab  # {token_id: decoded_str}

    def decode(self, ids):
        return "".join(self._vocab.get(tid, "?") for tid in ids)


TOKENIZER = MockTokenizer({
    100: " Hello",
    101: " world",
    102: ",",
    103: " this",
    104: " is",
    105: " a",
    106: " test",
    107: ".",
    108: " un",
    109: "believ",
    110: "able",
    111: " The",
    112: " quick",
    113: " brown",
    114: " fox",
    115: " jumps",
})

SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# A1. distribute_time_in_window — basic
# ---------------------------------------------------------------------------

class TestDistributeTimeInWindow:
    def test_basic(self):
        token_ids = [100, 101, 102]  # " Hello", " world", ","
        result = distribute_time_in_window(
            token_ids, start_sample=0, end_sample=19200,
            tokenizer=TOKENIZER, sample_rate=SAMPLE_RATE,
        )
        assert len(result) == 3
        # First starts at 0, last ends at 19200
        assert result[0]["start_sample"] == 0
        assert result[-1]["end_sample"] == 19200
        # All durations positive
        for r in result:
            assert r["end_sample"] > r["start_sample"]
        # Continuity — no gaps
        for i in range(len(result) - 1):
            assert result[i]["end_sample"] == result[i + 1]["start_sample"]

    # A2. single token
    def test_single_token(self):
        result = distribute_time_in_window(
            [100], start_sample=0, end_sample=19200,
            tokenizer=TOKENIZER, sample_rate=SAMPLE_RATE,
        )
        assert len(result) == 1
        assert result[0]["start_sample"] == 0
        assert result[0]["end_sample"] == 19200

    # A3. empty window
    def test_empty_window(self):
        result = distribute_time_in_window(
            [], start_sample=0, end_sample=19200,
            tokenizer=TOKENIZER, sample_rate=SAMPLE_RATE,
        )
        assert result == []


# ---------------------------------------------------------------------------
# A4. merge_subword_tokens_to_words — basic
# ---------------------------------------------------------------------------

class TestMergeSubwordTokens:
    def test_basic(self):
        # Tokens: " Hello", " world", ",", " this"
        token_timestamps = [
            {"token_id": 100, "text": " Hello", "start_sample": 0, "end_sample": 4800},
            {"token_id": 101, "text": " world", "start_sample": 4800, "end_sample": 9600},
            {"token_id": 102, "text": ",", "start_sample": 9600, "end_sample": 11200},
            {"token_id": 103, "text": " this", "start_sample": 11200, "end_sample": 16000},
        ]
        words = merge_subword_tokens_to_words(token_timestamps, sample_rate=SAMPLE_RATE)

        assert len(words) == 3
        assert words[0].word == "Hello"
        assert words[1].word == "world,"  # comma merges with preceding word
        assert words[2].word == "this"
        # Timing checks
        assert words[0].start_time == pytest.approx(0.0)
        assert words[0].end_time == pytest.approx(4800 / SAMPLE_RATE)
        assert words[1].start_time == pytest.approx(4800 / SAMPLE_RATE)
        assert words[1].end_time == pytest.approx(11200 / SAMPLE_RATE)

    # A5. cross-window word continuation
    def test_cross_window(self):
        # " un" in window 1, "believ" and "able" continue in window 2
        token_timestamps = [
            {"token_id": 108, "text": " un", "start_sample": 0, "end_sample": 6000},
            {"token_id": 109, "text": "believ", "start_sample": 6000, "end_sample": 14000},
            {"token_id": 110, "text": "able", "start_sample": 14000, "end_sample": 19200},
        ]
        words = merge_subword_tokens_to_words(token_timestamps, sample_rate=SAMPLE_RATE)

        assert len(words) == 1
        assert words[0].word == "unbelievable"
        assert words[0].start_time == pytest.approx(0.0)
        assert words[0].end_time == pytest.approx(19200 / SAMPLE_RATE)
        assert words[0].token_ids == [108, 109, 110]

    def test_empty_input(self):
        words = merge_subword_tokens_to_words([], sample_rate=SAMPLE_RATE)
        assert words == []


# ---------------------------------------------------------------------------
# A6. build_word_timestamps — full pipeline
# ---------------------------------------------------------------------------

class TestBuildWordTimestamps:
    def test_full_pipeline(self):
        window_token_map = [
            {"token_ids": [100, 101, 102, 103, 104], "start_sample": 0, "end_sample": 19200},
            {"token_ids": [105, 106, 107], "start_sample": 19200, "end_sample": 38400},
            {"token_ids": [111, 112, 113, 114, 115], "start_sample": 38400, "end_sample": 57600},
        ]
        words = build_word_timestamps(window_token_map, TOKENIZER, sample_rate=SAMPLE_RATE)

        # Check monotonicity
        for i in range(len(words) - 1):
            assert words[i].start_time <= words[i + 1].start_time
            assert words[i].end_time <= words[i + 1].start_time + 1e-9  # no overlaps

        # No gaps — each word ends where the next begins
        for i in range(len(words) - 1):
            assert words[i].end_time == pytest.approx(words[i + 1].start_time, abs=1e-9)

        # Total span matches total audio
        assert words[0].start_time == pytest.approx(0.0)
        assert words[-1].end_time == pytest.approx(57600 / SAMPLE_RATE)


# ---------------------------------------------------------------------------
# A7. timestamps_to_json
# ---------------------------------------------------------------------------

class TestTimestampsToJson:
    def test_basic(self):
        wts = [
            WordTimestamp(word="Hello", start_time=0.0, end_time=0.2, token_ids=[100]),
            WordTimestamp(word="world", start_time=0.2, end_time=0.5, token_ids=[101]),
        ]
        result = timestamps_to_json(wts)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["word"] == "Hello"
        assert "start_time" in result[0]
        assert "end_time" in result[0]
        # Verify JSON serializable
        json.dumps(result)


# ---------------------------------------------------------------------------
# A8. timestamps_to_srt
# ---------------------------------------------------------------------------

class TestTimestampsToSrt:
    def test_basic(self):
        wts = [
            WordTimestamp(word="Hello", start_time=0.0, end_time=0.2),
            WordTimestamp(word="world", start_time=0.2, end_time=0.5),
            WordTimestamp(word="this", start_time=0.5, end_time=0.7),
            WordTimestamp(word="is", start_time=0.7, end_time=0.9),
            WordTimestamp(word="a", start_time=0.9, end_time=1.0),
            WordTimestamp(word="test", start_time=1.0, end_time=1.3),
        ]
        srt = timestamps_to_srt(wts, words_per_cue=3)

        lines = srt.strip().split("\n")
        # Should have 2 cues
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "Hello world this" in lines[2]
        # Second cue
        assert "2" in srt
        assert "is a test" in srt

    def test_time_format(self):
        wts = [
            WordTimestamp(word="Long", start_time=3661.5, end_time=3662.0),
        ]
        srt = timestamps_to_srt(wts, words_per_cue=6)
        # 3661.5s = 1h 1m 1s 500ms => 01:01:01,500
        assert "01:01:01,500" in srt

    def test_empty(self):
        assert timestamps_to_srt([]) == ""
