import numpy as np
import torch
import json
import struct
from typing import Dict, Any, Union, Optional


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files using memory mapping for large tensors."""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor with memory-efficient strategies.

        For large tensors (>10MB) to CUDA, uses memory mapping to avoid intermediate copies.
        GPU transfers use pinned memory + non-blocking for efficiency — call
        torch.cuda.synchronize() before using the result if non-blocking matters.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        non_blocking = device is not None and device.type == "cuda"
        tensor_offset = self.header_size + 8 + offset_start

        if num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu":
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)
            del mm
            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)
            del byte_tensor
            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        self.file.seek(tensor_offset)
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array
        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor
        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


class MultipleSafetensorLoader:
    """Loads sharded HuggingFace safetensors from a model.safetensors.index.json file."""

    def __init__(self, model_index_file: str):
        with open(model_index_file, "r") as f:
            self.index = json.load(f)
        import pathlib
        self.path = pathlib.Path(model_index_file).parent

    def load_dict(self) -> Dict:
        import pathlib
        all_tensors = {}
        files = sorted(set(self.index.get('weight_map', {}).values()))
        for file in files:
            file_path = self.path / pathlib.Path(file)
            print(f"Loading shard: {file_path}")
            with MemoryEfficientSafeOpen(str(file_path)) as safe:
                for key in safe.keys():
                    all_tensors[key] = safe.get_tensor(key)
        return all_tensors
