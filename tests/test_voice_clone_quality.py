"""
Voice cloning quality tests.

These tests generate real TTS audio with the 1.5B model, then use that audio
as a voice reference to clone the speaker, and measure whether the cloned output
is measurably more similar to the reference than a no-reference baseline.

Metrics (ensemble of):
  - ECAPA-TDNN 192-dim speaker embeddings (speechbrain)
  - Librosa MFCC + delta + spectral contrast + F0 (always available)

All tests require the NAS model and run on GPU 0.
"""

import os
import numpy as np
import pytest
import torch

NAS_MODEL_PATH = "/mnt/nas/models/vibevoice/VibeVoice-1.5B"
NAS_AVAILABLE = os.path.isdir(NAS_MODEL_PATH)
nas_only = pytest.mark.skipif(not NAS_AVAILABLE, reason="NAS model not mounted")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
SR = 24_000


# ---------------------------------------------------------------------------
# Shared fixture — loads model, processor, and similarity engine once
# ---------------------------------------------------------------------------

@nas_only
class TestVoiceCloneQuality:

    @pytest.fixture(scope="class")
    def resources(self):
        """Load model, processor, and similarity engine once for the class."""
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.eval import SpeakerSimilarity

        processor = VibeVoiceProcessor.from_pretrained(NAS_MODEL_PATH)
        model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
            NAS_MODEL_PATH, device=DEVICE, torch_dtype=DTYPE,
        )
        model.eval()
        model.set_ddpm_inference_steps(10)

        # ECAPA-TDNN on the same GPU
        sim = SpeakerSimilarity(device=DEVICE)

        return {"model": model, "processor": processor, "sim": sim}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate(self, resources, text: str, voice_samples=None) -> np.ndarray:
        """Run TTS inference and return float32 numpy array at 24 kHz."""
        model = resources["model"]
        processor = resources["processor"]

        kwargs = dict(text=text, return_tensors="pt")
        if voice_samples is not None:
            kwargs["voice_samples"] = voice_samples
        inputs = processor(**kwargs).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=3.0,
                return_speech=True,
            )

        audio = output.speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Similarity engine sanity checks (no model needed)
    # ------------------------------------------------------------------

    def test_similarity_engine_self_consistency(self, resources):
        """Same audio vs itself must score 1.0 on all metrics."""
        sim = resources["sim"]
        wav = np.random.default_rng(0).standard_normal(SR * 3).astype(np.float32) * 0.1
        scores = sim.compute(wav, wav, sr=SR)

        assert abs(scores["librosa"] - 1.0) < 1e-4, (
            f"librosa self-similarity should be 1.0, got {scores['librosa']:.6f}"
        )
        if "ecapa_tdnn" in scores:
            assert abs(scores["ecapa_tdnn"] - 1.0) < 1e-3, (
                f"ECAPA self-similarity should be ~1.0, got {scores['ecapa_tdnn']:.6f}"
            )

    def test_similarity_engine_discriminates(self, resources):
        """Two independent signals must score lower than same-signal comparison."""
        sim = resources["sim"]
        rng = np.random.default_rng(42)
        wav_a = rng.standard_normal(SR * 3).astype(np.float32) * 0.1
        wav_b = rng.standard_normal(SR * 3).astype(np.float32) * 0.1

        same = sim.compute(wav_a, wav_a, sr=SR)
        diff = sim.compute(wav_a, wav_b, sr=SR)

        assert same["ensemble"] > diff["ensemble"], (
            f"Same-signal similarity ({same['ensemble']:.4f}) should exceed "
            f"different-signal similarity ({diff['ensemble']:.4f})"
        )

    def test_ecapa_available(self, resources):
        """ECAPA-TDNN must have loaded successfully."""
        sim = resources["sim"]
        assert sim.ecapa_available, (
            "ECAPA-TDNN speaker encoder failed to load. "
            "Check speechbrain installation and HF connectivity."
        )

    # ------------------------------------------------------------------
    # Voice cloning quality tests — real model audio
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def voice_clone_set(self, resources):
        """
        Generate three audio clips once for all quality tests:
          - ref_audio:      reference voice (model TTS, no voice sample)
          - cloned_audio:   new text generated using ref_audio as voice reference
          - baseline_audio: same new text generated WITHOUT voice reference
        """
        # Reference voice — a longer sentence gives better speaker characterization
        ref_text = (
            "Speaker 0: Good morning. This is a reference voice sample "
            "for testing the VibeVoice voice cloning system."
        )
        clone_text = (
            "Speaker 0: The quick brown fox jumps over the lazy dog "
            "near the riverbank on a sunny afternoon."
        )

        ref_audio = self._generate(resources, ref_text)
        cloned_audio = self._generate(
            resources, clone_text, voice_samples=[ref_audio]
        )
        baseline_audio = self._generate(resources, clone_text)

        return {
            "ref_audio": ref_audio,
            "cloned_audio": cloned_audio,
            "baseline_audio": baseline_audio,
        }

    def test_reference_audio_generated(self, voice_clone_set):
        """Reference audio must be non-empty speech."""
        ref = voice_clone_set["ref_audio"]
        assert ref is not None
        assert len(ref) >= SR * 2, (
            f"Reference audio too short: {len(ref)/SR:.1f}s (expected >= 2s)"
        )

    def test_cloned_audio_generated(self, voice_clone_set):
        """Cloned audio must be non-empty."""
        cloned = voice_clone_set["cloned_audio"]
        assert cloned is not None
        assert len(cloned) > 0

    def test_baseline_audio_generated(self, voice_clone_set):
        """Baseline audio must be non-empty."""
        baseline = voice_clone_set["baseline_audio"]
        assert baseline is not None
        assert len(baseline) > 0

    def test_cloned_differs_from_baseline(self, voice_clone_set):
        """Cloned and baseline outputs must not be identical waveforms."""
        cloned = voice_clone_set["cloned_audio"]
        baseline = voice_clone_set["baseline_audio"]
        min_len = min(len(cloned), len(baseline))
        diff = np.abs(cloned[:min_len] - baseline[:min_len]).mean()
        assert diff > 1e-4, (
            "Cloned and baseline audio are identical — "
            "voice conditioning does not appear to be applied."
        )

    def test_ecapa_cloning_beats_baseline(self, resources, voice_clone_set):
        """
        ECAPA-TDNN speaker similarity: reference vs cloned must exceed
        reference vs baseline, proving the voice is actually being transferred.
        """
        sim = resources["sim"]
        if not sim.ecapa_available:
            pytest.skip("ECAPA-TDNN not available")

        ref = voice_clone_set["ref_audio"]
        cloned = voice_clone_set["cloned_audio"]
        baseline = voice_clone_set["baseline_audio"]

        sim_cloned = sim.ecapa_similarity(ref, cloned, sr=SR)
        sim_baseline = sim.ecapa_similarity(ref, baseline, sr=SR)

        print(f"\n  ECAPA: cloned={sim_cloned:.4f}  baseline={sim_baseline:.4f}  "
              f"delta={sim_cloned - sim_baseline:+.4f}")

        assert sim_cloned > sim_baseline, (
            f"ECAPA similarity: ref vs cloned ({sim_cloned:.4f}) should exceed "
            f"ref vs baseline ({sim_baseline:.4f})"
        )

    def test_librosa_cloning_beats_baseline(self, resources, voice_clone_set):
        """
        Librosa multi-feature speaker similarity: reference vs cloned must
        exceed reference vs baseline.
        """
        sim = resources["sim"]
        ref = voice_clone_set["ref_audio"]
        cloned = voice_clone_set["cloned_audio"]
        baseline = voice_clone_set["baseline_audio"]

        sim_cloned = sim.librosa_similarity(ref, cloned, sr=SR)
        sim_baseline = sim.librosa_similarity(ref, baseline, sr=SR)

        print(f"\n  Librosa: cloned={sim_cloned:.4f}  baseline={sim_baseline:.4f}  "
              f"delta={sim_cloned - sim_baseline:+.4f}")

        assert sim_cloned > sim_baseline, (
            f"Librosa similarity: ref vs cloned ({sim_cloned:.4f}) should exceed "
            f"ref vs baseline ({sim_baseline:.4f})"
        )

    def test_ensemble_quality_report(self, resources, voice_clone_set):
        """
        Full quality report via sim.report() — ensemble score must show that
        cloned output is closer to the reference than the baseline.
        """
        sim = resources["sim"]
        report = sim.report(
            ref_audio=voice_clone_set["ref_audio"],
            cloned_audio=voice_clone_set["cloned_audio"],
            baseline_audio=voice_clone_set["baseline_audio"],
            sr=SR,
        )

        print(f"\n  {report['summary']}")
        print(f"  ref vs cloned:   {report['reference_vs_cloned']}")
        print(f"  ref vs baseline: {report['reference_vs_baseline']}")

        assert report["passed"], (
            f"Voice clone quality test FAILED.\n{report['summary']}\n"
            "The cloned audio is not measurably more similar to the reference "
            "than the baseline. Check voice cloning pipeline."
        )
