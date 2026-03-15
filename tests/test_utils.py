"""Unit tests for vibevoice.utils.*"""

import json
import struct
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path: str, tensors: dict, metadata: dict = None):
    """Write a minimal safetensors file for testing (F32 tensors only)."""
    import numpy as np

    # Build tensor data bytes and offsets
    data_parts = []
    header = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for key, tensor in tensors.items():
        arr = tensor.numpy().astype("float32")
        raw = arr.tobytes()
        header[key] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)

    header_json = json.dumps(header).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


# ---------------------------------------------------------------------------
# TestGetLogger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_logger(self):
        from vibevoice.utils.logger import get_logger
        import logging
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)

    def test_same_name_same_instance(self):
        from vibevoice.utils.logger import get_logger
        a = get_logger("same_name")
        b = get_logger("same_name")
        assert a is b

    def test_different_names(self):
        from vibevoice.utils.logger import get_logger
        a = get_logger("module_a")
        b = get_logger("module_b")
        assert a.name != b.name


# ---------------------------------------------------------------------------
# TestGetGenerator
# ---------------------------------------------------------------------------

class TestGetGenerator:
    def test_returns_generator(self):
        from vibevoice.utils.rand_init import get_generator
        gen = get_generator(seeds=0, force_set=True)
        assert isinstance(gen, torch.Generator)

    def test_seeded_determinism(self):
        from vibevoice.utils.rand_init import get_generator
        gen1 = get_generator(seeds=123, force_set=True)
        t1 = torch.randn(4, generator=gen1)
        gen2 = get_generator(seeds=123, force_set=True)
        t2 = torch.randn(4, generator=gen2)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        from vibevoice.utils.rand_init import get_generator
        gen1 = get_generator(seeds=1, force_set=True)
        t1 = torch.randn(16, generator=gen1)
        gen2 = get_generator(seeds=2, force_set=True)
        t2 = torch.randn(16, generator=gen2)
        assert not torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# TestMemoryEfficientSafeOpen
# ---------------------------------------------------------------------------

class TestMemoryEfficientSafeOpen:
    def test_read_keys(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        tensors = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, tensors)

        with MemoryEfficientSafeOpen(path) as safe:
            keys = safe.keys()

        assert set(keys) == {"weight", "bias"}

    def test_get_tensor_shape(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        w = torch.randn(3, 5)
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, {"w": w})

        with MemoryEfficientSafeOpen(path) as safe:
            loaded = safe.get_tensor("w")

        assert loaded.shape == (3, 5)

    def test_get_tensor_values(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, {"w": w})

        with MemoryEfficientSafeOpen(path) as safe:
            loaded = safe.get_tensor("w")

        assert torch.allclose(loaded, w)

    def test_metadata_round_trip(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, {"x": torch.zeros(2)},
                           metadata={"epoch": "3", "multiplier": "1.0"})

        with MemoryEfficientSafeOpen(path) as safe:
            meta = safe.metadata()

        assert meta["epoch"] == "3"
        assert meta["multiplier"] == "1.0"

    def test_missing_key_raises(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, {"x": torch.zeros(2)})

        with pytest.raises(KeyError):
            with MemoryEfficientSafeOpen(path) as safe:
                safe.get_tensor("nonexistent")

    def test_get_dtype_map_f32(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, {"x": torch.randn(4)})

        with MemoryEfficientSafeOpen(path) as safe:
            t = safe.get_tensor("x")

        assert t.dtype == torch.float32

    def test_dtype_cast_on_load(self, tmp_path):
        from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen
        path = str(tmp_path / "test.safetensors")
        _write_safetensors(path, {"x": torch.randn(4)})

        with MemoryEfficientSafeOpen(path) as safe:
            t = safe.get_tensor("x", dtype=torch.float16)

        assert t.dtype == torch.float16


# ---------------------------------------------------------------------------
# TestMultipleSafetensorLoader
# ---------------------------------------------------------------------------

class TestMultipleSafetensorLoader:
    def test_load_single_shard(self, tmp_path):
        from vibevoice.utils.safetensors_util import MultipleSafetensorLoader

        # Write a single shard
        shard = {"model.weight": torch.randn(8, 8)}
        _write_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), shard)

        # Write the index file
        index = {
            "metadata": {},
            "weight_map": {
                "model.weight": "model-00001-of-00001.safetensors"
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))

        loader = MultipleSafetensorLoader(str(index_path))
        result = loader.load_dict()

        assert "model.weight" in result
        assert result["model.weight"].shape == (8, 8)

    def test_load_two_shards(self, tmp_path):
        from vibevoice.utils.safetensors_util import MultipleSafetensorLoader

        shard1 = {"layer0.weight": torch.randn(4, 4)}
        shard2 = {"layer1.weight": torch.randn(4, 4)}
        _write_safetensors(str(tmp_path / "shard-01.safetensors"), shard1)
        _write_safetensors(str(tmp_path / "shard-02.safetensors"), shard2)

        index = {
            "metadata": {},
            "weight_map": {
                "layer0.weight": "shard-01.safetensors",
                "layer1.weight": "shard-02.safetensors",
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))

        loader = MultipleSafetensorLoader(str(index_path))
        result = loader.load_dict()

        assert set(result.keys()) == {"layer0.weight", "layer1.weight"}


# ---------------------------------------------------------------------------
# TestMergeLoraWeights
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)


class TestMergeLoraWeights:
    def _write_lora(self, tmp_path, layer_prefix: str, r: int, in_dim: int, out_dim: int,
                    alpha: float, up_val: float = 0.1, down_val: float = 0.1):
        """Write a synthetic LoRA safetensors file."""
        up = torch.full((out_dim, r), up_val)
        down = torch.full((r, in_dim), down_val)
        alpha_t = torch.tensor(alpha)

        tensors = {
            f"vibevoice_lora-{layer_prefix}.alpha": alpha_t,
            f"vibevoice_lora-{layer_prefix}.lora_up.weight": up,
            f"vibevoice_lora-{layer_prefix}.lora_down.weight": down,
        }
        path = str(tmp_path / "lora.safetensors")
        _write_safetensors(path, tensors, metadata={"multiplier": "1.0", "epoch": "1"})
        return path, up, down, alpha_t

    def test_delta_applied(self, tmp_path):
        from vibevoice.utils.model_utils import merge_lora_weights

        model = _TinyModel()
        original_weight = model.fc.weight.data.clone()

        r = 2
        path, up, down, alpha = self._write_lora(
            tmp_path, layer_prefix="fc", r=r, in_dim=4, out_dim=4,
            alpha=2.0, up_val=0.1, down_val=0.2,
        )

        merge_lora_weights(model, path, lora_weight=1.0)

        scale = alpha.item() / r
        expected_delta = (up.float() @ down.float()) * scale
        expected = original_weight.float() + expected_delta

        assert torch.allclose(model.fc.weight.data.float(), expected, atol=1e-5), (
            f"Merged weight does not match expected.\n"
            f"diff max: {(model.fc.weight.data.float() - expected).abs().max()}"
        )

    def test_lora_weight_scale(self, tmp_path):
        from vibevoice.utils.model_utils import merge_lora_weights

        model = _TinyModel()
        original = model.fc.weight.data.clone()

        r = 2
        path, up, down, alpha = self._write_lora(
            tmp_path, layer_prefix="fc", r=r, in_dim=4, out_dim=4,
            alpha=2.0, up_val=0.1, down_val=0.2,
        )

        scale = alpha.item() / r
        full_delta = (up.float() @ down.float()) * scale

        merge_lora_weights(model, path, lora_weight=0.5)

        expected = original.float() + 0.5 * full_delta
        assert torch.allclose(model.fc.weight.data.float(), expected, atol=1e-5)

    def test_missing_file_returns_original(self, tmp_path):
        from vibevoice.utils.model_utils import merge_lora_weights

        model = _TinyModel()
        original = model.fc.weight.data.clone()

        returned = merge_lora_weights(model, str(tmp_path / "nonexistent.safetensors"))

        assert returned is model
        assert torch.equal(model.fc.weight.data, original)

    def test_incomplete_lora_keys_skipped(self, tmp_path):
        """File with only lora_down (no lora_up/alpha) should not crash or modify model."""
        from vibevoice.utils.model_utils import merge_lora_weights

        down = torch.full((2, 4), 0.1)
        tensors = {"vibevoice_lora-fc.lora_down.weight": down}
        path = str(tmp_path / "incomplete.safetensors")
        _write_safetensors(path, tensors)

        model = _TinyModel()
        original = model.fc.weight.data.clone()

        returned = merge_lora_weights(model, path)
        assert returned is model
        assert torch.equal(model.fc.weight.data, original)


# ---------------------------------------------------------------------------
# TestOffloadConfig
# ---------------------------------------------------------------------------

class TestOffloadConfig:
    def test_defaults(self):
        from vibevoice.modular.custom_offloading_utils import OffloadConfig
        cfg = OffloadConfig()
        assert cfg.enabled is False
        assert cfg.num_layers_on_gpu == 8
        assert cfg.offload_prediction_head is False
        assert cfg.pin_memory is True
        assert cfg.prefetch_next_layer is True
        assert cfg.async_transfer is True

    def test_override(self):
        from vibevoice.modular.custom_offloading_utils import OffloadConfig
        cfg = OffloadConfig(enabled=True, num_layers_on_gpu=12, offload_prediction_head=True)
        assert cfg.enabled is True
        assert cfg.num_layers_on_gpu == 12
        assert cfg.offload_prediction_head is True
