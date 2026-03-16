"""
Speaker Similarity Engine for voice cloning quality evaluation.

Three complementary metrics:

  1. WeSpeaker (ONNX, via wespeakerruntime)
     256-dim ResNet34-LM speaker embeddings trained on VoxCeleb.
     ~25 MB ONNX model, CPU-friendly, no GPU required.

  2. ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
     192-dim speaker embeddings from a model trained for speaker verification.
     Operates at 16 kHz. Downloads ~80 MB on first use.

  3. Librosa multi-feature vector
     MFCC + delta + spectral contrast + F0 statistics combined into a
     single cosine similarity. No model download required; always available.

  Ensemble score = 0.7 * primary_model + 0.3 * librosa (or librosa-only
  if no neural model is available). Primary model prefers wespeaker > ECAPA.

Usage::

    from vibevoice.eval import SpeakerSimilarity

    sim = SpeakerSimilarity(backend="wespeaker")
    scores = sim.compute(wav_reference, wav_cloned, sr=24000)
    print(scores)
    # {'wespeaker': 0.87, 'librosa': 0.73, 'ensemble': 0.83}
"""

import logging
import os
import tempfile
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SPEECHBRAIN_MODEL_DIR = os.environ.get("ECAPA_MODEL_DIR", "/tmp/speechbrain_models")
_ECAPA_HF_ID = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_SR = 16_000        # speechbrain model expects 16 kHz
_VIBEVOICE_SR = 24_000    # VibeVoice model output sample rate
_WESPEAKER_LANG = os.environ.get("WESPEAKER_LANG", "en")
_WESPEAKER_MODEL_PATH = os.environ.get("WESPEAKER_MODEL_PATH", "")


# ---------------------------------------------------------------------------
# Librosa feature extraction
# ---------------------------------------------------------------------------

def _librosa_speaker_vector(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a discriminative speaker feature vector using librosa.

    Features (concatenated):
        - MFCC mean + std (40 coefficients → 80 dims)
        - MFCC delta mean + std (40 → 80)
        - Spectral contrast mean + std (6 bands → 14 dims)
        - Fundamental frequency (F0) mean + std (2 dims)

    Total: 176-dim vector.
    """
    import librosa

    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=0)

    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_bands=6)

    try:
        f0, _, _ = librosa.pyin(wav, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        f0_stats = np.array([f0_valid.mean(), f0_valid.std()]) if len(f0_valid) > 0 else np.zeros(2)
    except Exception:
        f0_stats = np.zeros(2)

    return np.concatenate([
        mfcc.mean(1), mfcc.std(1),
        mfcc_delta.mean(1), mfcc_delta.std(1),
        contrast.mean(1), contrast.std(1),
        f0_stats,
    ])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Speaker Similarity class
# ---------------------------------------------------------------------------

class SpeakerSimilarity:
    """
    Ensemble speaker similarity calculator.

    Args:
        device: torch device string for the ECAPA-TDNN model (``"cuda:0"``,
            ``"cpu"``, etc.)
        model_dir: local directory to cache the speechbrain model weights.
        backend: preferred neural backend — ``"wespeaker"`` (ONNX, default),
            ``"ecapa"`` (speechbrain), or ``"auto"`` (try wespeaker then ECAPA).
    """

    def __init__(
        self,
        device: str = "cpu",
        model_dir: str = _SPEECHBRAIN_MODEL_DIR,
        backend: str = "wespeaker",
    ):
        self.device = device
        self.model_dir = model_dir
        self._ecapa = None
        self._wespeaker = None

        if backend in ("wespeaker", "auto"):
            self._load_wespeaker()
        if backend in ("ecapa", "auto") or (backend == "wespeaker" and self._wespeaker is None):
            self._load_ecapa()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_wespeaker(self) -> None:
        try:
            import wespeakerruntime as wespeaker
            if _WESPEAKER_MODEL_PATH and os.path.isfile(_WESPEAKER_MODEL_PATH):
                self._wespeaker = wespeaker.Speaker(onnx_path=_WESPEAKER_MODEL_PATH)
                logger.info("WeSpeaker ONNX loaded from %s.", _WESPEAKER_MODEL_PATH)
            else:
                self._wespeaker = wespeaker.Speaker(lang=_WESPEAKER_LANG)
                logger.info("WeSpeaker ONNX loaded (lang=%s).", _WESPEAKER_LANG)
        except Exception as exc:
            logger.warning(
                "WeSpeaker unavailable (%s). Will try ECAPA-TDNN fallback.", exc
            )

    def _load_ecapa(self) -> None:
        try:
            # speechbrain <= 1.0.3 still calls torchaudio.list_audio_backends() which
            # was removed in torchaudio 2.0.  Patch it back in if missing.
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: []

            try:
                from speechbrain.inference.classifiers import EncoderClassifier  # speechbrain >= 1.0
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier  # speechbrain 0.5.x
            self._ecapa = EncoderClassifier.from_hparams(
                source=_ECAPA_HF_ID,
                run_opts={"device": self.device},
                savedir=self.model_dir,
            )
            logger.info("ECAPA-TDNN speaker encoder loaded.")
        except Exception as exc:
            logger.warning(
                f"ECAPA-TDNN unavailable ({exc}). "
                "Speaker similarity will use librosa features only."
            )

    @property
    def wespeaker_available(self) -> bool:
        return self._wespeaker is not None

    @property
    def ecapa_available(self) -> bool:
        return self._ecapa is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wespeaker_embedding(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        if self._wespeaker is None:
            return None
        import soundfile as sf
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, wav, sr)
        try:
            emb = self._wespeaker.extract_embedding(tmp_path)
            return emb.squeeze()
        finally:
            os.unlink(tmp_path)

    def _resample_to_ecapa(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        """Resample to 16 kHz mono and return as (1, T) tensor."""
        import librosa
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        if sr != _ECAPA_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=_ECAPA_SR)
        return torch.from_numpy(wav).unsqueeze(0)

    def _ecapa_embedding(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        if self._ecapa is None:
            return None
        t = self._resample_to_ecapa(wav, sr)
        with torch.no_grad():
            emb = self._ecapa.encode_batch(t)
        return emb.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wespeaker_similarity(self, wav1: np.ndarray, wav2: np.ndarray, sr: int = _VIBEVOICE_SR) -> Optional[float]:
        """
        Cosine similarity of WeSpeaker ONNX speaker embeddings.
        Returns ``None`` if WeSpeaker is unavailable.
        """
        e1 = self._wespeaker_embedding(wav1, sr)
        e2 = self._wespeaker_embedding(wav2, sr)
        if e1 is None or e2 is None:
            return None
        return _cosine(e1, e2)

    def ecapa_similarity(self, wav1: np.ndarray, wav2: np.ndarray, sr: int = _VIBEVOICE_SR) -> Optional[float]:
        """
        Cosine similarity of ECAPA-TDNN speaker embeddings.
        Returns ``None`` if the ECAPA model is unavailable.
        """
        e1 = self._ecapa_embedding(wav1, sr)
        e2 = self._ecapa_embedding(wav2, sr)
        if e1 is None or e2 is None:
            return None
        return _cosine(e1, e2)

    def librosa_similarity(self, wav1: np.ndarray, wav2: np.ndarray, sr: int = _VIBEVOICE_SR) -> float:
        """
        Cosine similarity of the librosa multi-feature speaker vector.
        Always available (no model download needed).
        """
        v1 = _librosa_speaker_vector(wav1, sr)
        v2 = _librosa_speaker_vector(wav2, sr)
        return _cosine(v1, v2)

    def compute(
        self,
        wav1: np.ndarray,
        wav2: np.ndarray,
        sr: int = _VIBEVOICE_SR,
    ) -> Dict[str, float]:
        """
        Compute all available speaker similarity metrics.

        Args:
            wav1: First audio waveform (float32, mono, any length).
            wav2: Second audio waveform (float32, mono, any length).
            sr: Sample rate of the input waveforms (default: 24 000 Hz).

        Returns:
            Dictionary with keys:
                - ``ecapa_tdnn``: ECAPA-TDNN cosine similarity (only if available)
                - ``librosa``: librosa multi-feature cosine similarity
                - ``ensemble``: weighted combination of available metrics
        """
        results: Dict[str, float] = {}

        # Prefer wespeaker, fall back to ECAPA
        primary_key = None
        ws = self.wespeaker_similarity(wav1, wav2, sr)
        if ws is not None:
            results["wespeaker"] = ws
            primary_key = "wespeaker"

        ecapa = self.ecapa_similarity(wav1, wav2, sr)
        if ecapa is not None:
            results["ecapa_tdnn"] = ecapa
            if primary_key is None:
                primary_key = "ecapa_tdnn"

        results["librosa"] = self.librosa_similarity(wav1, wav2, sr)

        if primary_key is not None:
            results["ensemble"] = 0.7 * results[primary_key] + 0.3 * results["librosa"]
        else:
            results["ensemble"] = results["librosa"]

        return results

    def report(
        self,
        ref_audio: np.ndarray,
        cloned_audio: np.ndarray,
        baseline_audio: np.ndarray,
        sr: int = _VIBEVOICE_SR,
    ) -> Dict:
        """
        Full quality report comparing cloned audio against a reference and a baseline.

        Args:
            ref_audio: Reference voice sample used for cloning.
            cloned_audio: Audio generated with the reference voice applied.
            baseline_audio: Audio generated for the same text WITHOUT voice reference.
            sr: Sample rate of all waveforms.

        Returns:
            Dictionary with similarity scores and a ``passed`` boolean indicating
            whether cloned audio is measurably closer to the reference than baseline.
        """
        sim_cloned = self.compute(ref_audio, cloned_audio, sr)
        sim_baseline = self.compute(ref_audio, baseline_audio, sr)

        passed = sim_cloned["ensemble"] > sim_baseline["ensemble"]

        return {
            "reference_vs_cloned": sim_cloned,
            "reference_vs_baseline": sim_baseline,
            "delta": {k: sim_cloned.get(k, 0) - sim_baseline.get(k, 0)
                      for k in sim_cloned},
            "passed": passed,
            "summary": (
                f"Cloned similarity: {sim_cloned['ensemble']:.4f}  |  "
                f"Baseline similarity: {sim_baseline['ensemble']:.4f}  |  "
                f"Delta: {sim_cloned['ensemble'] - sim_baseline['ensemble']:+.4f}  |  "
                f"{'PASS' if passed else 'FAIL'}"
            ),
        }
