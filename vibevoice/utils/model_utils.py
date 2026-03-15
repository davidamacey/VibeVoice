import hashlib
import safetensors.torch
import torch

from io import BytesIO
from torch import nn
from vibevoice.utils.logger import get_logger

logger = get_logger(__name__)


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()
    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate model hashes for sd-webui-additional-networks indexing."""
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}
    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)
    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def merge_lora_weights(model: nn.Module, lora_path: str, lora_weight: float = 1.0) -> nn.Module:
    """Merge LoRA weights directly into model weights (bakes in the LoRA delta).

    Args:
        model: The base model
        lora_path: Path to a LoRA safetensors file with 'vibevoice_lora-' prefixed keys
        lora_weight: Scale multiplier for the LoRA delta (default: 1.0)

    Returns:
        The model with LoRA deltas merged into its base weights
    """
    import os
    from vibevoice.utils.safetensors_util import MemoryEfficientSafeOpen

    if not os.path.exists(lora_path):
        logger.warning(f"LoRA file not found: {lora_path}. Returning original model.")
        return model

    try:
        logger.info(f"Loading LoRA weights from {lora_path}")
        with MemoryEfficientSafeOpen(lora_path) as safe:
            lora_sd = {key: safe.get_tensor(key) for key in safe.keys()}
            metadata = safe.metadata()

            if metadata and "multiplier" in metadata:
                try:
                    if lora_weight is None:
                        lora_weight = float(metadata["multiplier"])
                        logger.info(f"Using multiplier from metadata: {lora_weight}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid multiplier in metadata: {metadata['multiplier']}, using default: {lora_weight}")
    except Exception as e:
        logger.warning(f"Failed to load LoRA weights from {lora_path}, returning original model.", exc_info=e)
        return model

    lora_structure = {}
    for key in lora_sd.keys():
        if not key.startswith("vibevoice_lora-"):
            continue
        key_without_prefix = key[len("vibevoice_lora-"):]

        if key_without_prefix.endswith(".alpha"):
            original_key = key_without_prefix[:-len(".alpha")].replace("-", ".") + ".weight"
            lora_structure.setdefault(original_key, {})["alpha"] = lora_sd[key]
        elif key_without_prefix.endswith(".lora_down.weight"):
            original_key = key_without_prefix[:-len(".lora_down.weight")].replace("-", ".") + ".weight"
            lora_structure.setdefault(original_key, {})["lora_down.weight"] = lora_sd[key]
        elif key_without_prefix.endswith(".lora_up.weight"):
            original_key = key_without_prefix[:-len(".lora_up.weight")].replace("-", ".") + ".weight"
            lora_structure.setdefault(original_key, {})["lora_up.weight"] = lora_sd[key]

    keys_to_remove = [
        k for k, v in lora_structure.items()
        if not all(c in v for c in ["alpha", "lora_down.weight", "lora_up.weight"])
    ]
    for key in keys_to_remove:
        logger.warning(f"Incomplete LoRA components for key '{key}', skipping.")
        del lora_structure[key]

    logger.info(f"Found {len(lora_structure)} valid LoRA layers to merge")

    if not lora_structure:
        logger.warning(f"No valid LoRA layers found in {lora_path}. Returning original model.")
        return model

    merged_count = 0
    missing_count = 0
    scale = 1.0

    for original_key, lora_components in lora_structure.items():
        down_weight = lora_components["lora_down.weight"]
        up_weight = lora_components["lora_up.weight"]
        alpha = lora_components["alpha"]

        dim = down_weight.size()[0]
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        scale = alpha / dim

        try:
            param = model
            for attr in original_key.split("."):
                param = getattr(param, attr)

            if not isinstance(param, nn.Parameter):
                logger.warning(f"Key {original_key} does not correspond to a model parameter")
                missing_count += 1
                continue

            device = param.device
            dtype = param.dtype
            up_weight = up_weight.to(device=device, dtype=dtype)
            down_weight = down_weight.to(device=device, dtype=dtype)
            lora_delta = lora_weight * (up_weight @ down_weight) * scale
            param.data = param.data + lora_delta
            merged_count += 1

        except AttributeError:
            logger.warning(f"Could not find parameter for key: {original_key}")
            missing_count += 1
            continue

    logger.info(f"Successfully merged {merged_count} LoRA layers, weight={lora_weight}, scale={scale}.")
    if missing_count > 0:
        logger.warning(f"Failed to merge {missing_count} LoRA layers (keys not found in model)")

    return model
