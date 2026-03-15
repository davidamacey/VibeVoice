import torch
import math
import re
import os

from typing import List, Optional, Union, Type, Dict
from torch import nn
from vibevoice.utils.logger import get_logger

logger = get_logger(__name__)


class LoRAModule(nn.Module):
    """LoRA Module for fine-tuning VibeVoice TTS layers.

    Based on kohya-ss/musubi-tuner.

    Args:
        lora_name: Unique name for this LoRA module
        original_name: Dotted path of the original module in the model
        lora_dim: LoRA rank (r)
        lora_alpha: LoRA alpha (scale = alpha / lora_dim)
        dropout: Dropout applied to lora_down output during training
        multiplier: Global scale multiplier
        original_module: The nn.Linear module being adapted
        rank_dropout: Rank-level dropout rate
        module_dropout: Probability of skipping the entire LoRA path
    """
    def __init__(self,
                 lora_name: str,
                 original_name: str,
                 lora_dim: int = 16,
                 lora_alpha: int = 1,
                 dropout: float = 0.0,
                 multiplier: float = 1.0,
                 original_module: nn.Linear = None,
                 rank_dropout: int = 0.0,
                 module_dropout: Optional[float] = None):
        super().__init__()

        self.lora_name = lora_name
        self.original_name = original_name
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        in_dim = original_module.in_features
        out_dim = original_module.out_features
        self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)
        alpha = self.lora_dim if lora_alpha is None or lora_alpha == 0 else lora_alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))
        self.org_module = original_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx)
        return org_forwarded + lx * self.multiplier * scale


class LoRANetwork(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.verbose = verbose
        self.prefix = "vibevoice_lora"

        self.loraplus_lr_ratio = None
        self.fine_tuning_layers = self._includes_layers()

        logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
        logger.info(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, "
                    f"module dropout: p={self.module_dropout}")

        def create_modules(pfx: str, root_module: torch.nn.Module,
                           default_dim: Optional[int] = None) -> List[LoRAModule]:
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ == "Linear":
                    original_name = name
                    lora_name = f"{pfx}.{original_name}".replace(".", "-")

                    matched = any(pattern.match(original_name) for pattern in self.fine_tuning_layers)
                    if not matched:
                        continue

                    dim = default_dim if default_dim is not None else self.lora_dim
                    if dim is None or dim == 0:
                        continue

                    lora = LoRAModule(lora_name, original_name,
                                      original_module=module,
                                      multiplier=self.multiplier,
                                      lora_dim=dim,
                                      lora_alpha=self.alpha,
                                      dropout=dropout,
                                      rank_dropout=rank_dropout,
                                      module_dropout=module_dropout)
                    loras.append(lora)
            return loras

        self.lora_layers: List[LoRAModule] = create_modules(self.prefix, model)

        if not self.lora_layers:
            raise RuntimeError("No LoRA modules were created. Please check the configuration.")

        logger.info(f"create LoRA for VibeVoice: {len(self.lora_layers)} modules.")
        if verbose:
            for lora in self.lora_layers:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        names = set()
        for lora in self.lora_layers:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def _includes_layers(self) -> List[re.Pattern]:
        """Target layers for LoRA fine-tuning.

        Attention + MLP layers in the Qwen2.5 backbone and
        key projection/FFN layers in the diffusion prediction head.
        """
        patterns = [
            r"^model.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"^model.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$",
            r"^model.prediction_head\.cond_proj$",
            r"^model.prediction_head\.layers\.\d+\.ffn\.(gate_proj|up_proj|down_proj)$",
            r"^model.prediction_head\.layers\.\d+\.adaLN_modulation\.1$",
        ]
        return [re.compile(p) for p in patterns]

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.lora_layers:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.lora_layers:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        return self.load_state_dict(weights_sd, False)

    def apply_to(self):
        if not self.lora_layers:
            raise RuntimeError("No LoRA modules found")
        for lora in self.lora_layers:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, weights_sd, dtype=None, device=None, non_blocking=False):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for lora in self.lora_layers:
                sd_for_lora = {
                    key[len(lora.lora_name) + 1:]: weights_sd[key]
                    for key in weights_sd.keys()
                    if key.startswith(lora.lora_name)
                }
                if not sd_for_lora:
                    logger.info(f"no weight for {lora.lora_name}")
                    continue
                futures.append(executor.submit(lora.merge_to, sd_for_lora, dtype, device, non_blocking))
        for future in futures:
            future.result()
        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, learning_rate: float = 1e-4, **kwargs):
        self.requires_grad_(True)
        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if not param_data["params"]:
                    continue
                if lr is not None:
                    param_data["lr"] = lr * loraplus_ratio if key == "plus" else lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.lora_layers:
            params, descriptions = assemble_params(self.lora_layers, learning_rate, self.loraplus_lr_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["vibevoice" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file: str, dtype: torch.dtype, metadata: Dict[str, str]):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from vibevoice.utils import model_utils

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


def create_network(original_model: nn.Module,
                   multiplier: float,
                   network_dim: Optional[int],
                   network_alpha: Optional[float],
                   neuron_dropout: Optional[float] = None,
                   **kwargs) -> LoRANetwork:
    """Architecture-independent LoRA network creation helper."""
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    network = LoRANetwork(original_model,
                          multiplier=multiplier,
                          lora_dim=network_dim,
                          alpha=network_alpha,
                          dropout=neuron_dropout,
                          rank_dropout=rank_dropout,
                          module_dropout=module_dropout,
                          verbose=verbose)

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network
