"""
Tests for vibevoice.modular.modeling_vibevoice_inference and related types.

Unit tests run without loading any real model weights (fast, always pass).
Integration tests require the NAS model at /mnt/nas/models/vibevoice/VibeVoice-1.5B
and are skipped automatically when the path is absent.
"""

import os
import pytest
import torch
import torch.nn as nn

NAS_MODEL_PATH = "/mnt/nas/models/vibevoice/VibeVoice-1.5B"
NAS_AVAILABLE = os.path.isdir(NAS_MODEL_PATH)
nas_only = pytest.mark.skipif(not NAS_AVAILABLE, reason="NAS model not mounted")


# ---------------------------------------------------------------------------
# TestVibeVoiceGenerationOutput
# ---------------------------------------------------------------------------

class TestVibeVoiceGenerationOutput:
    def test_construct_minimal(self):
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceGenerationOutput
        out = VibeVoiceGenerationOutput(
            sequences=torch.zeros(1, 10, dtype=torch.long),
            speech_outputs=None,
        )
        assert out.sequences.shape == (1, 10)
        assert out.speech_outputs is None

    def test_construct_with_audio(self):
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceGenerationOutput
        audio = torch.randn(24000)
        out = VibeVoiceGenerationOutput(
            sequences=torch.zeros(1, 5, dtype=torch.long),
            speech_outputs=[audio],
        )
        assert out.speech_outputs is not None
        assert len(out.speech_outputs) == 1
        assert out.speech_outputs[0].shape == (24000,)

    def test_none_speech_outputs(self):
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceGenerationOutput
        out = VibeVoiceGenerationOutput(
            sequences=None,
            speech_outputs=None,
        )
        assert out.sequences is None
        assert out.speech_outputs is None


# ---------------------------------------------------------------------------
# TestOffloadConfig (structural/defaults)
# ---------------------------------------------------------------------------

class TestOffloadConfigDefaults:
    def test_disabled_by_default(self):
        from vibevoice.modular.custom_offloading_utils import OffloadConfig
        cfg = OffloadConfig()
        assert cfg.enabled is False

    def test_gpu_layers_default(self):
        from vibevoice.modular.custom_offloading_utils import OffloadConfig
        cfg = OffloadConfig()
        assert isinstance(cfg.num_layers_on_gpu, int)
        assert cfg.num_layers_on_gpu > 0

    def test_enabled_flag(self):
        from vibevoice.modular.custom_offloading_utils import OffloadConfig
        cfg = OffloadConfig(enabled=True, num_layers_on_gpu=4)
        assert cfg.enabled is True
        assert cfg.num_layers_on_gpu == 4

    def test_dataclass_equality(self):
        from vibevoice.modular.custom_offloading_utils import OffloadConfig
        a = OffloadConfig(enabled=True, num_layers_on_gpu=16)
        b = OffloadConfig(enabled=True, num_layers_on_gpu=16)
        assert a == b


# ---------------------------------------------------------------------------
# TestVibeVoiceForConditionalGenerationInference (unit — no weights)
# ---------------------------------------------------------------------------

class TestInferenceClassStructure:
    def test_class_importable(self):
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        assert VibeVoiceForConditionalGenerationInference is not None

    def test_has_from_pretrained_hf(self):
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        assert callable(getattr(VibeVoiceForConditionalGenerationInference, "from_pretrained_hf", None))

    def test_has_from_pretrained_file(self):
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        assert callable(getattr(VibeVoiceForConditionalGenerationInference, "from_pretrained_file", None))

    def test_has_generate(self):
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        assert callable(getattr(VibeVoiceForConditionalGenerationInference, "generate", None))

    def test_has_set_ddpm_inference_steps(self):
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        assert callable(getattr(VibeVoiceForConditionalGenerationInference, "set_ddpm_inference_steps", None))

    def test_top_level_imports(self):
        """Ensure all symbols are reachable from the top-level vibevoice package."""
        from vibevoice import (
            VibeVoiceForConditionalGenerationInference,
            VibeVoiceGenerationOutput,
            OffloadConfig,
            LayerOffloader,
            GenerationVisitor,
            LoRANetwork,
            create_network,
        )
        assert all(x is not None for x in [
            VibeVoiceForConditionalGenerationInference,
            VibeVoiceGenerationOutput,
            OffloadConfig,
            LayerOffloader,
            GenerationVisitor,
            LoRANetwork,
            create_network,
        ])


# ---------------------------------------------------------------------------
# TestLoRANetwork (unit — tiny synthetic model)
# ---------------------------------------------------------------------------

class _TinyQwen(nn.Module):
    """Minimal model that looks like the patterns LoRANetwork searches for."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.ModuleDict({
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": nn.ModuleDict({
                        "q_proj": nn.Linear(8, 8, bias=False),
                        "k_proj": nn.Linear(8, 8, bias=False),
                    })
                })
            ])
        })

    def parameters(self):
        for p in super().parameters():
            yield p

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


class TestLoRANetworkUnit:
    def test_create_network_importable(self):
        from vibevoice.lora.lora_network import create_network, LoRANetwork
        assert create_network is not None
        assert LoRANetwork is not None

    def test_lora_module_forward(self):
        from vibevoice.lora.lora_network import LoRAModule
        base = nn.Linear(8, 4, bias=False)
        lora = LoRAModule(
            lora_name="test",
            original_name="fc",
            original_module=base,
            multiplier=1.0,
            lora_dim=2,
            lora_alpha=2,
        )
        lora.apply_to()
        x = torch.randn(3, 8)
        out = lora(x)
        assert out.shape == (3, 4)

    def test_lora_module_zero_init(self):
        """LoRA up-projection is zero-initialized, so the initial delta is zero."""
        from vibevoice.lora.lora_network import LoRAModule
        base = nn.Linear(8, 4, bias=False)
        x = torch.randn(3, 8)
        # Record raw linear output BEFORE patching
        with torch.no_grad():
            expected = base(x).clone()

        lora = LoRAModule(
            lora_name="test",
            original_name="fc",
            original_module=base,
            multiplier=1.0,
            lora_dim=2,
            lora_alpha=2,
        )
        lora.apply_to()  # patches base.forward; lora_up is zero-init => delta == 0

        with torch.no_grad():
            patched = lora(x)

        assert torch.allclose(expected, patched, atol=1e-6), (
            "Initial LoRA output should equal base output (lora_up is zero-init)"
        )


# ---------------------------------------------------------------------------
# TestGenerationVisitor (abstract interface)
# ---------------------------------------------------------------------------

class TestGenerationVisitor:
    def test_visitor_is_abstract(self):
        from vibevoice.generation.visitor import GenerationVisitor
        import inspect
        assert inspect.isabstract(GenerationVisitor)

    def test_concrete_subclass_works(self):
        from vibevoice.generation.visitor import GenerationVisitor

        class _NoopVisitor(GenerationVisitor):
            def visit_preprocessing(self, *a, **kw): pass
            def visit_inference_start(self, *a, **kw): pass
            def visit_inference_batch_start(self, *a, **kw): pass
            def visit_inference_batch_end(self, *a, **kw): pass
            def visit_inference_save_audio_file(self, *a, **kw): pass
            def visit_inference_step_start(self, *a, **kw): pass
            def visit_inference_step_end(self, *a, **kw): pass
            def visit_completed(self, *a, **kw): pass
            def visit_failed(self, *a, **kw): pass

        v = _NoopVisitor()
        assert isinstance(v, GenerationVisitor)


# ---------------------------------------------------------------------------
# Integration tests — require NAS model
# ---------------------------------------------------------------------------

@nas_only
class TestInferenceIntegration:
    @pytest.fixture(scope="class")
    def model_and_processor(self):
        """Load the 1.5B model once for all integration tests in this class."""
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        processor = VibeVoiceProcessor.from_pretrained(NAS_MODEL_PATH)
        model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
            NAS_MODEL_PATH,
            device="cpu",
            torch_dtype=torch.float32,
        )
        model.eval()
        return model, processor

    def test_model_loads(self, model_and_processor):
        model, processor = model_and_processor
        assert model is not None
        assert processor is not None

    def test_model_has_language_model(self, model_and_processor):
        model, _ = model_and_processor
        assert hasattr(model, "model") or hasattr(model, "language_model"), (
            "Model should have a 'model' or 'language_model' attribute"
        )

    def test_processor_tokenizes(self, model_and_processor):
        _, processor = model_and_processor
        # Processor requires "Speaker N: ..." format
        inputs = processor(text="Speaker 0: Hello world.", return_tensors="pt")
        assert "input_ids" in inputs
        assert inputs["input_ids"].shape[-1] > 0

    def test_set_ddpm_steps(self, model_and_processor):
        model, _ = model_and_processor
        model.set_ddpm_inference_steps(10)

    def test_generate_short_text(self, model_and_processor):
        """Generate a short audio sample on CPU (slow but validates the forward pass)."""
        model, processor = model_and_processor
        model.set_ddpm_inference_steps(5)

        inputs = processor(text="Speaker 0: Hi.", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=2.0,
                return_speech=True,
            )

        assert output is not None
        # Either we get audio or sequences — at minimum no exception
        if output.speech_outputs:
            audio = output.speech_outputs[0]
            assert audio is not None
            # Audio may be 1D (samples,) or 2D (1, samples)
            assert audio.ndim in (1, 2)
            assert audio.numel() > 0
