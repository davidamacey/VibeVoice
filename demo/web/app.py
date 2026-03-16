import datetime
import builtins
import asyncio
import gc
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

import copy

SAMPLE_RATE_1P5B = 24_000

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        # Prefer NAS-resident voices when the model path is set; fall back to repo copy
        model_voices = Path(self.model_path) / "voices"
        repo_voices = BASE.parent / "voices" / "streaming_model"
        if model_voices.exists():
            voices_dir = model_voices
        elif repo_voices.exists():
            voices_dir = repo_voices
        else:
            raise RuntimeError(f"Voices directory not found at {model_voices} or {repo_voices}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-Carter_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
        results: Optional[list] = None,
    ) -> None:
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                return_word_timestamps=True,
            )
            if results is not None:
                results.append(outputs)
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        gen_results: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
                "results": gen_results,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]
            # Emit word timing after generation completes
            if gen_results and gen_results[0].word_timestamps:
                from vibevoice.modular.word_timing import timestamps_to_json
                word_ts = gen_results[0].word_timestamps[0]
                if word_ts:
                    emit("word_timing", words=timestamps_to_json(word_ts))

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


# ---------------------------------------------------------------------------
# 1.5B TTS service (non-streaming, REST-based)
# ---------------------------------------------------------------------------

class TTSService1p5B:
    """Wraps VibeVoiceForConditionalGenerationInference for REST-based TTS."""

    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 20):
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.model = None
        self.processor = None
        self._lock = threading.Lock()

    def load(self) -> None:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        print(f"[1.5B startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        load_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"[1.5B startup] Loading model with dtype={load_dtype}, device={self.device}")
        try:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=self.device,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=self.device,
                attn_implementation="sdpa",
            )
        self.model.eval()
        self.model.set_ddpm_inference_steps(self.inference_steps)
        print("[1.5B startup] Model ready.")

    def generate(
        self,
        text: str,
        voice_audio: Optional[np.ndarray] = None,
        voice_samples: Optional[List[np.ndarray]] = None,
        cfg_scale: float = 3.0,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Synthesize speech. Returns float32 numpy array at 24 kHz.

        voice_audio: single reference for Speaker 0 (convenience wrapper).
        voice_samples: per-speaker list [spk0, spk1, ...] (takes precedence).
        seed: RNG seed for reproducible generation (None = random).
        """
        if steps is not None:
            self.model.set_ddpm_inference_steps(steps)

        # Build voice_samples list: explicit list takes precedence over single audio
        if voice_samples is None:
            voice_samples = [voice_audio] if voice_audio is not None else None

        if seed is not None:
            torch.manual_seed(seed)

        with self._lock:
            inputs = self.processor(
                text=text.strip(),
                voice_samples=voice_samples,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    return_speech=True,
                    show_progress_bar=False,
                )

        if output.speech_outputs and output.speech_outputs[0] is not None:
            audio = output.speech_outputs[0].cpu().float().numpy()
            return audio
        return np.zeros(0, dtype=np.float32)

    @staticmethod
    def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE_1P5B) -> bytes:
        import wave as _wave
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# ASR service (non-streaming, REST-based)
# ---------------------------------------------------------------------------

class ASRService:
    """Wraps VibeVoiceASRForConditionalGeneration for REST-based transcription."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self._lock = threading.Lock()

    def load(self) -> None:
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        tokenizer_path = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen2.5-7B")
        print(f"[ASR startup] Loading processor from {self.model_path}, tokenizer={tokenizer_path}")
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_path,
            language_model_pretrained_name=tokenizer_path,
        )

        load_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"[ASR startup] Loading model with dtype={load_dtype}, device={self.device}")
        # Note: don't pass dtype= to from_pretrained — the ASR config already stores
        # "dtype" as a string, and passing a torch.dtype object overwrites it and
        # breaks HF's JSON config logging. Cast after load instead.
        try:
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except Exception:
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
        self.model = self.model.to(load_dtype).to(self.device)
        self.model.eval()
        print("[ASR startup] ASR model ready.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 24000) -> dict:
        """Transcribe audio array. Returns dict with text and segments."""
        ASR_SAMPLE_RATE = 24000
        if sample_rate != ASR_SAMPLE_RATE:
            waveform = torch.from_numpy(audio).unsqueeze(0).float()
            waveform = torchaudio.functional.resample(waveform, sample_rate, ASR_SAMPLE_RATE)
            audio = waveform.squeeze(0).numpy()

        with self._lock:
            inputs = self.processor(
                audio=audio,
                sampling_rate=ASR_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True,
            )
            load_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            casted = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device=self.device, dtype=load_dtype) if v.is_floating_point() else v.to(self.device)
                casted[k] = v
            inputs = casted
            gen_cfg = {
                "max_new_tokens": 512,
                "pad_token_id": self.processor.pad_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "do_sample": False,
            }
            device_type = "cuda" if self.device.startswith("cuda") else "cpu"
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
                output_ids = self.model.generate(**inputs, **gen_cfg)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        eos_pos = (generated_ids == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            generated_ids = generated_ids[: eos_pos[0] + 1]

        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        try:
            segments = self.processor.post_process_transcription(raw_text)
        except Exception:
            segments = []

        return {"text": raw_text, "segments": segments}


# ---------------------------------------------------------------------------
# Hot-swap model manager — keeps one heavy model in VRAM at a time
# ---------------------------------------------------------------------------

class HotSwapManager:
    """Load one heavy model (TTS 1.5B / Large / ASR) on demand.

    Only one model is kept in VRAM at a time.  Requesting a different model
    automatically unloads the current one first via del + gc + cuda cache clear.
    The 0.5B streaming model is managed separately and is always loaded.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._configs: Dict[str, dict] = {}      # model_id -> {type, path, steps}
        self._current_id: Optional[str] = None
        self._current_service: Optional[Any] = None
        self._lock = threading.Lock()

    def register(self, model_id: str, model_type: str, path: str, steps: int = 20) -> None:
        """Register a model config without loading it."""
        self._configs[model_id] = {"type": model_type, "path": path, "steps": steps}

    def is_registered(self, model_id: str) -> bool:
        return model_id in self._configs

    def get_or_load(self, model_id: str) -> Any:
        """Return the service for model_id, loading (and unloading the current) as needed.
        Blocking — call via asyncio.to_thread from async handlers."""
        with self._lock:
            if self._current_id == model_id and self._current_service is not None:
                return self._current_service
            self._unload_current()
            self._current_service = self._load(model_id)
            self._current_id = model_id
            return self._current_service

    def _unload_current(self) -> None:
        if self._current_service is None:
            return
        print(f"[hot-swap] Unloading {self._current_id}")
        svc = self._current_service
        self._current_service = None
        self._current_id = None
        if hasattr(svc, "model") and svc.model is not None:
            del svc.model
            svc.model = None
        if hasattr(svc, "processor") and svc.processor is not None:
            del svc.processor
            svc.processor = None
        del svc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[hot-swap] VRAM freed.")

    def _load(self, model_id: str) -> Any:
        cfg = self._configs.get(model_id)
        if cfg is None:
            raise ValueError(f"Model {model_id!r} not registered")
        print(f"[hot-swap] Loading {model_id} from {cfg['path']}")
        if cfg["type"] == "tts":
            svc: Any = TTSService1p5B(
                model_path=cfg["path"], device=self.device, inference_steps=cfg["steps"]
            )
        else:
            svc = ASRService(model_path=cfg["path"], device=self.device)
        svc.load()
        print(f"[hot-swap] {model_id} ready.")
        return svc

    def status(self) -> Dict[str, bool]:
        """Return {model_id: is_currently_loaded} for all registered models."""
        return {mid: (self._current_id == mid) for mid in self._configs}

    def currently_loaded(self) -> Optional[str]:
        return self._current_id


def _load_audio_bytes(audio_bytes: bytes, filename: str = "") -> tuple:
    """Load audio from raw bytes. Returns (waveform np.float32 1D, sample_rate int).

    Tries soundfile first (WAV/FLAC/OGG/MP3), then falls back to ffmpeg
    for video containers (MP4, MKV, WebM, etc.) or unsupported codecs.
    """
    import soundfile as sf

    # Try soundfile (works for wav, flac, ogg, mp3 with libsndfile >= 1.1)
    try:
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        pass

    # Fallback: ffmpeg can handle MP4/MKV/WebM/MP3/etc.
    suffix = Path(filename).suffix if filename else ".bin"
    if not suffix:
        suffix = ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path,
             "-ac", "1", "-ar", "24000", "-f", "wav", tmp_out_path],
            capture_output=True, check=True,
        )
        data, sr = sf.read(tmp_out_path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    finally:
        os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)


def _split_into_word_chunks(text: str, max_words: int) -> List[str]:
    """Split text at sentence boundaries so no chunk exceeds max_words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Split on sentence-ending punctuation, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_words + s_words > max_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = s_words
        else:
            current.append(sentence)
            current_words += s_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def _speed_shift_audio(audio: np.ndarray, speed: float) -> np.ndarray:
    """Time-stretch audio via linear interpolation (pitch-preserving for small deltas).
    Used on reference voice audio before model conditioning."""
    if abs(speed - 1.0) < 0.01 or audio.size == 0:
        return audio
    target_len = max(1, int(len(audio) / speed))
    src_indices = np.arange(len(audio))
    tgt_indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(tgt_indices, src_indices, audio).astype(np.float32)


def _generate_with_pauses(
    service: "TTSService1p5B",
    text: str,
    voice_audio: Optional[np.ndarray],
    voice_samples: Optional[List[Optional[np.ndarray]]],
    cfg_scale: float,
    steps: Optional[int],
    seed: Optional[int],
    speed: float,
    max_words: int,
) -> np.ndarray:
    """Split text on [pause:Xs]/[pause:Xms] markers, auto-chunk long segments,
    generate each piece, and stitch with silence.

    Speed is applied to the reference voice audio before model conditioning
    (pitch-preserving via interpolation).  When no voice reference is provided,
    speed falls back to output resampling.
    """
    # Apply speed to reference audio so the model conditions on the adjusted tempo
    shifted_voice_audio = _speed_shift_audio(voice_audio, speed) if voice_audio is not None else None
    shifted_voice_samples: Optional[List[Optional[np.ndarray]]] = None
    if voice_samples is not None:
        shifted_voice_samples = [
            (_speed_shift_audio(v, speed) if v is not None else None)
            for v in voice_samples
        ]

    # Split on [pause:Xs] / [pause:Xms] — re.split with 2 capture groups gives
    # (text, dur_str, unit) triples
    parts = re.split(r'\[pause:(\d+(?:\.\d+)?)(s|ms)?\]', text)
    audio_chunks: List[np.ndarray] = []

    for idx in range(0, len(parts), 3):
        chunk_text = parts[idx].strip()
        if chunk_text:
            # Ensure at least one Speaker prefix is present
            if not re.search(r'Speaker\s*\d+\s*:', chunk_text):
                chunk_text = f"Speaker 0: {chunk_text}"

            # Auto-chunk long segments at sentence boundaries
            sub_chunks = _split_into_word_chunks(chunk_text, max_words)
            for sub in sub_chunks:
                chunk_audio = service.generate(
                    text=sub,
                    voice_audio=shifted_voice_audio,
                    voice_samples=shifted_voice_samples,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    seed=seed,
                )
                if chunk_audio.size > 0:
                    audio_chunks.append(chunk_audio)

        # Insert silence after this segment if a pause marker follows
        if idx + 2 < len(parts):
            try:
                duration = float(parts[idx + 1])
                unit = parts[idx + 2] or "s"
                if unit == "ms":
                    duration /= 1000.0
                silence_samples = max(0, int(duration * SAMPLE_RATE_1P5B))
                audio_chunks.append(np.zeros(silence_samples, dtype=np.float32))
            except (ValueError, TypeError):
                pass

    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(audio_chunks)


def _apply_speed_output(audio: np.ndarray, speed: float) -> np.ndarray:
    """Fallback output resample when no voice reference is provided.
    Changes pitch proportionally — use only when reference-based speed isn't available."""
    if abs(speed - 1.0) < 0.01 or audio.size == 0:
        return audio
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    new_sr = max(1, int(round(SAMPLE_RATE_1P5B / speed)))
    waveform = torchaudio.functional.resample(waveform, SAMPLE_RATE_1P5B, new_sr)
    return waveform.squeeze(0).numpy()


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")

    service = StreamingTTSService(
        model_path=model_path,
        device=device,
    )
    service.load()
    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()

    # Register heavy models in the hot-swap manager (lazy — nothing loaded yet)
    hot_swap = HotSwapManager(device=device)

    model_path_1p5b = os.environ.get("MODEL_PATH_1P5B")
    if model_path_1p5b:
        steps_1p5b = int(os.environ.get("MODEL_1P5B_STEPS", "20"))
        hot_swap.register("1.5b", "tts", model_path_1p5b, steps_1p5b)
        print(f"[startup] 1.5B model registered (path={model_path_1p5b})")

    model_path_large = os.environ.get("MODEL_PATH_LARGE")
    if model_path_large:
        steps_large = int(os.environ.get("MODEL_LARGE_STEPS", "20"))
        hot_swap.register("large", "tts", model_path_large, steps_large)
        print(f"[startup] Large model registered (path={model_path_large})")

    model_path_asr = os.environ.get("MODEL_PATH_ASR")
    if model_path_asr:
        hot_swap.register("asr", "asr", model_path_asr)
        print(f"[startup] ASR model registered (path={model_path_asr})")

    app.state.hot_swap = hot_swap
    print("[startup] Hot-swap manager ready. Heavy models load on first request.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)

@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")
    do_sample_param = ws.query_params.get("do_sample", "false")
    temperature_param = ws.query_params.get("temperature")
    top_p_param = ws.query_params.get("top_p")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
    except ValueError:
        cfg_scale = 1.5
    if cfg_scale <= 0:
        cfg_scale = 1.5
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None
    do_sample = do_sample_param.lower() in ("1", "true", "yes")
    try:
        temperature = float(temperature_param) if temperature_param is not None else 0.9
    except ValueError:
        temperature = 0.9
    try:
        top_p = float(top_p_param) if top_p_param is not None else 0.9
    except ValueError:
        top_p = 0.9

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
        )

        stop_signal = threading.Event()

        iterator = streaming_tts(
            text,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice_key=voice_param,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            log_callback=enqueue_log,
            stop_event=stop_signal,
        )
        sentinel = object()
        first_ws_send_logged = False

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                await flush_logs()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        except Exception as e:
            print(f"Error in websocket stream: {e}")
            traceback.print_exc()
            enqueue_log("backend_error", message=str(e))
            stop_signal.set()
        finally:
            stop_signal.set()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
        "models": get_models()["models"],
    }


@app.get("/models")
def get_models():
    """List available models (registered + loaded status)."""
    hot_swap: HotSwapManager = app.state.hot_swap
    hs_status = hot_swap.status()
    models = [
        {"id": "0.5b-streaming", "name": "VibeVoice Realtime 0.5B (streaming)",
         "streaming": True, "loaded": True},
    ]
    if "1.5b" in hs_status:
        models.append({"id": "1.5b", "name": "VibeVoice TTS 1.5B (voice cloning)",
                        "streaming": False, "loaded": hs_status["1.5b"]})
    if "large" in hs_status:
        models.append({"id": "large", "name": "VibeVoice Large (voice cloning, 7B)",
                        "streaming": False, "loaded": hs_status["large"]})
    if "asr" in hs_status:
        models.append({"id": "asr", "name": "VibeVoice ASR 7B (transcription)",
                        "streaming": False, "asr": True, "loaded": hs_status["asr"]})
    return {"models": models}


async def _load_voice_file(upload: UploadFile) -> np.ndarray:
    """Read an uploaded audio file and return float32 mono array at SAMPLE_RATE_1P5B."""
    audio_bytes = await upload.read()
    data, sr = _load_audio_bytes(audio_bytes, upload.filename or "")
    if sr != SAMPLE_RATE_1P5B:
        waveform = torch.from_numpy(data).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE_1P5B)
        data = waveform.squeeze(0).numpy()
    return data.astype(np.float32)


@app.post("/generate")
async def generate_1p5b(
    text: str = Form(...),
    cfg_scale: float = Form(3.0),
    steps: Optional[int] = Form(None),
    speed: float = Form(1.0),
    seed: Optional[int] = Form(None),
    max_words: int = Form(200),
    model_id: str = Form("1.5b"),
    voice_0: Optional[UploadFile] = File(None),
    voice_1: Optional[UploadFile] = File(None),
    voice_2: Optional[UploadFile] = File(None),
    voice_3: Optional[UploadFile] = File(None),
):
    """Generate speech with the 1.5B or Large model (non-streaming, supports voice cloning).

    Supports [pause:Xs] / [pause:Xms] markers in text for programmatic silence.
    Natural pauses via punctuation (..., ,, —) are handled by the model automatically.

    Form fields:
        text: Text to synthesize. Use 'Speaker N: text' for multi-speaker.
              Insert [pause:0.5s] or [pause:500ms] for timed silences.
        cfg_scale: CFG guidance scale (default 3.0).
        steps: DDPM inference steps override (default: model default).
        speed: Speed multiplier 0.5–2.0 (default 1.0). Applied to reference audio for
               pitch-preserving conditioning; falls back to output resample with no reference.
        seed: RNG seed for reproducible output (default: random).
        max_words: Auto-chunk segments longer than this word count (default 200).
        model_id: '1.5b' or 'large' (default '1.5b').
        voice_0..voice_3: Optional audio files for per-speaker voice cloning.
    """
    # Select service via hot-swap manager (loads on demand, unloads previous)
    hot_swap: HotSwapManager = app.state.hot_swap
    if not hot_swap.is_registered(model_id):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is not configured. Check MODEL_PATH_1P5B / MODEL_PATH_LARGE environment variables.",
        )
    try:
        service: TTSService1p5B = await asyncio.to_thread(hot_swap.get_or_load, model_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_id}': {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    speed = max(0.5, min(2.0, speed))
    max_words = max(50, min(500, max_words))

    # Load per-speaker voice files
    voice_uploads = [voice_0, voice_1, voice_2, voice_3]
    voice_samples: List[Optional[np.ndarray]] = []
    has_any_voice = False
    for upload in voice_uploads:
        if upload is not None:
            try:
                arr = await _load_voice_file(upload)
                voice_samples.append(arr)
                has_any_voice = True
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not decode voice file: {e}")
        else:
            voice_samples.append(None)

    # Normalize: if no voices provided, pass None; otherwise pass the list
    resolved_voice_samples: Optional[List[Optional[np.ndarray]]] = voice_samples if has_any_voice else None
    resolved_voice_audio: Optional[np.ndarray] = voice_samples[0] if has_any_voice else None

    try:
        audio = await asyncio.to_thread(
            _generate_with_pauses,
            service,
            text,
            resolved_voice_audio,
            resolved_voice_samples,
            cfg_scale,
            steps,
            seed,
            speed,
            max_words,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # If no voice reference was given, speed couldn't be applied at conditioning time —
    # fall back to output resampling
    if not has_any_voice:
        audio = _apply_speed_output(audio, speed)

    wav_bytes = TTSService1p5B.audio_to_wav_bytes(audio)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    )


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe an audio file with the ASR model.

    Returns JSON with transcribed text and word-level segments.

    Form fields:
        audio: Audio file (WAV/MP3/FLAC/etc.)
    """
    hot_swap: HotSwapManager = app.state.hot_swap
    if not hot_swap.is_registered("asr"):
        raise HTTPException(
            status_code=503,
            detail="ASR model not configured. Set MODEL_PATH_ASR environment variable to enable.",
        )
    try:
        asr_service: ASRService = await asyncio.to_thread(hot_swap.get_or_load, "asr")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load ASR model: {e}")

    try:
        audio_bytes = await audio.read()
        audio_array, sr = _load_audio_bytes(audio_bytes, audio.filename or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio file: {e}")

    try:
        result = await asyncio.to_thread(asr_service.transcribe, audio_array, sr)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return result

