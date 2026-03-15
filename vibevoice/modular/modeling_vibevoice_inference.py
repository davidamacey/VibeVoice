"""
VibeVoice inference model for TTS generation.

This module provides ``VibeVoiceForConditionalGenerationInference``, a HuggingFace-compatible
inference class that supports:

- Zero-shot voice cloning via ``speech_tensors`` / ``speech_masks``
- Classifier-free guidance (CFG) during diffusion decoding
- Streaming audio output via ``audio_streamer``
- Float8 E4M3FN quantized weights (via ``AutoCast`` wrappers)
- Layer-wise CPU<->GPU offloading for low-VRAM inference
- TTS LoRA weight merging
- Deterministic generation via seeded noise
- Generation visitor callbacks for progress tracking

Sources:
  - Community inference: vibevoice-community/VibeVoice
  - Enhanced features:   zhao-kun/VibeVoiceFusion
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceTokenizerEncoderOutput
from vibevoice.modular.modeling_vibevoice import VibeVoiceModel, VibeVoicePreTrainedModel
from vibevoice.modular.streamer import AudioStreamer, AsyncAudioStreamer
from vibevoice.generation.visitor import GenerationVisitor
from vibevoice.utils.rand_init import get_generator
from vibevoice.utils.logger import get_logger

logger = get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """Output type for VibeVoice generation.

    Args:
        sequences: Generated token id sequences of shape ``(batch, seq_len)``.
        speech_outputs: List of decoded audio waveforms (one per batch sample), or ``None``.
        reach_max_step_sample: Boolean mask indicating which samples hit the generation limit.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None


# ---------------------------------------------------------------------------
# Logits processor
# ---------------------------------------------------------------------------

class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only the valid VibeVoice control tokens."""

    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.valid_token_ids] = 0
        return scores + mask


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class VibeVoiceForConditionalGenerationInference(VibeVoicePreTrainedModel, GenerationMixin):
    """VibeVoice TTS inference model.

    Extends ``VibeVoicePreTrainedModel`` with a custom ``generate()`` loop
    that interleaves LM token prediction with diffusion-based speech synthesis.

    Loading:
        Standard HuggingFace checkpoint::

            model = VibeVoiceForConditionalGenerationInference.from_pretrained("path/to/checkpoint")

        Sharded safetensors (e.g., 1.5B model from NAS)::

            model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf("/mnt/nas/models/vibevoice/VibeVoice-1.5B")

        Single safetensors file (optionally with float8/LoRA)::

            model = VibeVoiceForConditionalGenerationInference.from_pretrained_file(
                "path/to/model.safetensors",
                config=config,
                lora_model_path="path/to/lora.safetensors",
                offload_config=OffloadConfig(enabled=True, num_layers_on_gpu=12),
            )
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config: VibeVoiceConfig):
        super().__init__(config)

        self.model = VibeVoiceModel(config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)

        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Layer offloader (set by from_pretrained_file if offload_config is provided)
        self.offloader = None

        self.post_init()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    # ------------------------------------------------------------------
    # Weight tie / embedding helpers
    # ------------------------------------------------------------------

    def tie_weights(self):
        if not getattr(self.config, 'tie_word_embeddings', False):
            return
        if hasattr(self, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        self.model.set_speech_tokenizers(acoustic_tokenizer, semantic_tokenizer)

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def __del__(self):
        if hasattr(self, 'offloader') and self.offloader is not None:
            try:
                self.offloader.cleanup()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Speech input processing
    # ------------------------------------------------------------------

    def _process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
        """Encode speech through acoustic tokenizer and project into LM embedding space."""
        with torch.no_grad():
            if speech_type == "audio":
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = (
                    (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device))
                    * self.model.speech_scaling_factor.to(acoustic_latents.device)
                )
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]
                return acoustic_features, acoustic_connected

            elif speech_type == "pt":
                encoder_output = VibeVoiceTokenizerEncoderOutput(
                    mean=speech_tensors, std=self.acoustic_tokenizer.config.fix_std
                )
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = (
                    (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device))
                    * self.model.speech_scaling_factor.to(acoustic_latents.device)
                )
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]
                return acoustic_features, acoustic_connected

            else:
                raise NotImplementedError(f"Speech type '{speech_type}' not implemented")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if speech_tensors is not None and speech_masks is not None:
            acoustic_features, speech_embeds = self._process_speech_inputs(
                speech_tensors.to(self.dtype), speech_masks
            )
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this inference class.")

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _build_generate_config_model_kwargs(
        self, generation_config, inputs, tokenizer, return_processors=False, **kwargs
    ):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config,
            True,
            speech_start_id=tokenizer.speech_start_id,
            speech_end_id=tokenizer.speech_end_id,
            speech_diffusion_id=tokenizer.speech_diffusion_id,
            **kwargs,
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = self.device

        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)

        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )
            stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=StoppingCriteriaList()
            )
            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria
        else:
            return generation_config, model_kwargs, input_ids

    # ------------------------------------------------------------------
    # Main generate() loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        is_prefill: bool = True,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        tqdm_class: Optional[type] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        """Generate token sequences and optionally decode to speech.

        Args:
            inputs: Input tensor (typically from processor).
            generation_config: HuggingFace GenerationConfig.
            audio_streamer: Optional streamer for real-time audio output.
            speech_tensors: Voice cloning reference audio (encoded waveform).
            speech_masks: Boolean mask for valid frames in ``speech_tensors``.
            speech_input_mask: Positions in the input sequence for speech embeddings.
            is_prefill: Whether to inject speech embeddings on the first step.
            return_speech: If True, decode acoustic latents and return waveforms.
            cfg_scale: Classifier-free guidance scale for diffusion (1.0 = no CFG).
            stop_check_fn: Optional callable returning True to abort generation.
            tqdm_class: Optional custom tqdm class for the progress bar.
            **kwargs: Passed to the underlying model (must include ``input_ids``).

        Returns:
            ``VibeVoiceGenerationOutput`` with ``sequences``, ``speech_outputs``,
            and ``reach_max_step_sample``.
        """
        generation_visitor: GenerationVisitor = kwargs.pop("generation_visitor", None)
        tokenizer = kwargs.pop("tokenizer", None)
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        max_length_times = kwargs.pop("max_length_times", 2)

        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = (
                self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]
            )

        generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = \
            self._build_generate_config_model_kwargs(
                generation_config, inputs, tokenizer, return_processors=True, **kwargs
            )

        negative_kwargs = {
            'input_ids': torch.full(
                (kwargs['input_ids'].shape[0], 1),
                tokenizer.speech_start_id,
                dtype=torch.long,
                device=kwargs['input_ids'].device,
            ),
            'attention_mask': torch.ones(
                (kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device
            ),
            'max_new_tokens': kwargs.get('max_new_tokens', 100),
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = \
            self._build_generate_config_model_kwargs(None, None, tokenizer, return_processors=False, **negative_kwargs)

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()

        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        inputs_embeds = None
        verbose = kwargs.get("verbose", False)

        audio_chunks = [[] for _ in range(batch_size)]
        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)

        valid_tokens = [
            generation_config.speech_start_id,
            generation_config.speech_end_id,
            generation_config.speech_diffusion_id,
            generation_config.eos_token_id,
        ]
        if hasattr(generation_config, 'bos_token_id') and generation_config.bos_token_id is not None:
            valid_tokens.append(generation_config.bos_token_id)

        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(token_constraint_processor)

        max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
        max_step_per_sample = torch.min(
            generation_config.max_length - initial_length_per_sample,
            (max_length_times * initial_length_per_sample).long()
        )
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if kwargs.get("show_progress_bar", True):
            tqdm_fn = tqdm_class if tqdm_class is not None else tqdm
            progress_bar = tqdm_fn(range(max_steps), desc="Generating", leave=False)
        else:
            progress_bar = range(max_steps)

        for step in progress_bar:
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                if audio_streamer is not None:
                    audio_streamer.end()
                break

            if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
                if any(audio_streamer.finished_flags):
                    if verbose:
                        print(f"Audio generation stopped externally at step {step + 1}")
                    break

            if finished_tags.all():
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description("Generation complete")
                break

            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopping.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break

            if hasattr(progress_bar, 'set_description'):
                active_samples = (~finished_tags).sum().item()
                progress_bar.set_description(f"Generating (active: {active_samples}/{batch_size})")

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                prefill_inputs = {}
                if speech_tensors is not None:
                    prefill_inputs["speech_tensors"] = speech_tensors.to(device=device)
                if speech_masks is not None:
                    prefill_inputs["speech_masks"] = speech_masks.to(device)
                if speech_input_mask is not None:
                    prefill_inputs["speech_input_mask"] = speech_input_mask.to(device)
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}

            if generation_visitor is not None:
                generation_visitor.visit_inference_step_start(current_step=step, total_steps=max_steps)

            outputs = self(
                **model_inputs, **prefill_inputs,
                logits_to_keep=1, return_dict=True,
                output_attentions=False, output_hidden_states=False,
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if generation_config.do_sample:
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_tokens[finished_tags] = generation_config.eos_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if not kwargs.get('refresh_negative', True):
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None
                negative_outputs = self(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True,
                    output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

            # EOS handling
            if (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    finished_tags[new_eos_indices] = True
                    if verbose:
                        print(f"Samples {new_eos_indices.tolist()} reached EOS at step {step + 1}.", flush=True)
                    if audio_streamer is not None:
                        audio_streamer.end(new_eos_indices)

            # Max-per-sample length handling
            max_length_reached = step >= max_step_per_sample
            new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
            if new_max_length_indices.numel() > 0:
                finished_tags[new_max_length_indices] = True
                reach_max_step_sample[new_max_length_indices] = True
                if verbose:
                    print(f"Samples {new_max_length_indices.tolist()} reached max length at step {step + 1}.", flush=True)
                if audio_streamer is not None:
                    audio_streamer.end(new_max_length_indices)

            # Speech-segment end: clear tokenizer caches
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)

            # Speech-segment start: reset negative KV cache for CFG
            diffusion_start_indices = torch.arange(batch_size, device=device)[
                ~finished_tags & (next_tokens == generation_config.speech_start_id)
            ]
            if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                for sample_idx in diffusion_start_indices.tolist():
                    negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                    negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                for layer_idx, (k_cache, v_cache) in enumerate(
                    zip(negative_model_kwargs['past_key_values'].key_cache,
                        negative_model_kwargs['past_key_values'].value_cache)
                ):
                    for sample_idx in diffusion_start_indices.tolist():
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = generation_config.speech_start_id

            # Prepare next embeddings
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)

            # Diffusion decoding for speech tokens
            diffusion_indices = torch.arange(batch_size, device=device)[
                ~finished_tags & (next_tokens == generation_config.speech_diffusion_id)
            ]

            if diffusion_indices.numel() > 0:
                if kwargs.get('refresh_negative', True):
                    negative_model_inputs = self.prepare_inputs_for_generation(
                        negative_input_ids, **negative_model_kwargs
                    )
                    if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                        negative_model_inputs['inputs_embeds'] = inputs_embeds
                        negative_model_inputs['input_ids'] = None
                    negative_outputs = self(
                        **negative_model_inputs, logits_to_keep=0, return_dict=True,
                        output_attentions=False, output_hidden_states=False,
                    )
                    negative_model_kwargs = self._update_model_kwargs_for_generation(
                        negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                    )
                    negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # Fix KV cache for non-diffusion samples
                non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                    start_indices = correct_cnt[non_diffusion_indices]
                    seq_len = negative_model_kwargs['attention_mask'].shape[1]

                    for i, (sample_idx, start_idx) in enumerate(
                        zip(non_diffusion_indices.tolist(), start_indices.tolist())
                    ):
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx + 1:] = \
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                    for layer_idx, (k_cache, v_cache) in enumerate(
                        zip(negative_model_kwargs['past_key_values'].key_cache,
                            negative_model_kwargs['past_key_values'].value_cache)
                    ):
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < k_cache.shape[2] - 1:
                                k_cache[sample_idx, :, start_idx + 1:, :] = \
                                    k_cache[sample_idx, :, start_idx:-1, :].clone()
                                v_cache[sample_idx, :, start_idx + 1:, :] = \
                                    v_cache[sample_idx, :, start_idx:-1, :].clone()

                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx + 1:] = \
                                negative_input_ids[sample_idx, start_idx:-1].clone()

                    correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                speech_latent = self.sample_speech_tokens(
                    positive_condition, negative_condition, cfg_scale=cfg_scale
                ).unsqueeze(1)

                scaled_latent = (
                    speech_latent / self.model.speech_scaling_factor.to(speech_latent.device)
                    - self.model.speech_bias_factor.to(speech_latent.device)
                )
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,
                    sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                    use_cache=True,
                    debug=False,
                )

                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                if audio_streamer is not None:
                    audio_streamer.put(audio_chunk, diffusion_indices)

                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False,
                ).mean

                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                next_inputs_embeds[diffusion_indices] = diffusion_embeds

                if generation_visitor is not None:
                    generation_visitor.visit_inference_step_end(current_step=step, total_steps=max_steps)

            inputs_embeds = next_inputs_embeds

        if audio_streamer is not None:
            audio_streamer.end()

        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                final_audio_outputs.append(torch.cat(sample_chunks, dim=-1))
            else:
                final_audio_outputs.append(None)

        return VibeVoiceGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )

    # ------------------------------------------------------------------
    # Diffusion sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale: float = 3.0):
        """Run DDPM diffusion loop with classifier-free guidance.

        Args:
            condition: Positive LM hidden state for conditioning, shape ``(n, hidden)``.
            neg_condition: Negative LM hidden state, shape ``(n, hidden)``.
            cfg_scale: ``uncond + cfg_scale * (cond - uncond)``.  1.0 = no guidance.
        """
        self._ensure_prediction_head_on_gpu()
        try:
            self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
            condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
            temp = torch.randn(condition.shape[0], self.config.acoustic_vae_dim, generator=get_generator())
            speech = temp.to(condition)
            for t in self.model.noise_scheduler.timesteps:
                half = speech[: len(speech) // 2]
                combined = torch.cat([half, half], dim=0)
                eps = self.model.prediction_head(
                    combined, t.repeat(combined.shape[0]).to(combined), condition=condition
                )
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
            return speech[: len(speech) // 2]
        finally:
            self._move_prediction_head_to_cpu()

    # ------------------------------------------------------------------
    # Offloading helpers
    # ------------------------------------------------------------------

    def _ensure_prediction_head_on_gpu(self):
        if self.offloader and self.offloader.config.offload_prediction_head:
            if self.model.prediction_head.device.type == 'cpu':
                self.model.prediction_head.to(self.device)

    def _move_prediction_head_to_cpu(self):
        if self.offloader and self.offloader.config.offload_prediction_head:
            if self.model.prediction_head.device.type != 'cpu':
                self.model.prediction_head.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Loading classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_hf(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "VibeVoiceForConditionalGenerationInference":
        """Load from a HuggingFace-format checkpoint (sharded or single safetensors).

        This is the recommended way to load the 1.5B model::

            model = VibeVoiceForConditionalGenerationInference.from_pretrained_hf(
                "/mnt/nas/models/vibevoice/VibeVoice-1.5B",
                device="cuda",
            )

        Args:
            model_name_or_path: Local directory or HuggingFace repo ID.
            device: Target device string.
            torch_dtype: Dtype for model weights (default: bfloat16).
        """
        return cls.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            **kwargs,
        )

    @classmethod
    def from_pretrained_file(
        cls,
        model_path: str,
        config: VibeVoiceConfig,
        device: str = "cuda",
        offload_config=None,
        lora_model_path: Optional[str] = None,
        lora_weight: float = 1.0,
    ) -> "VibeVoiceForConditionalGenerationInference":
        """Load from a single safetensors file or sharded directory (VibeVoiceFusion style).

        Supports float8 quantized weights, layer offloading, and LoRA merging::

            from vibevoice.modular.custom_offloading_utils import OffloadConfig

            model = VibeVoiceForConditionalGenerationInference.from_pretrained_file(
                "/mnt/nas/models/vibevoice/VibeVoice-1.5B",
                config=config,
                device="cuda",
                offload_config=OffloadConfig(enabled=True, num_layers_on_gpu=12),
                lora_model_path="path/to/tts_lora.safetensors",
            )

        Args:
            model_path: Path to a single ``.safetensors`` file OR a directory containing
                        ``model.safetensors.index.json`` (sharded HuggingFace format).
            config: ``VibeVoiceConfig`` instance.
            device: Target CUDA device (e.g. ``"cuda"`` or ``"cuda:0"``).
            offload_config: Optional ``OffloadConfig`` for layer-wise CPU offloading.
            lora_model_path: Optional path to a LoRA ``.safetensors`` file to merge in.
            lora_weight: LoRA delta multiplier.
        """
        from accelerate import init_empty_weights
        from vibevoice.utils.safetensors_util import MultipleSafetensorLoader, MemoryEfficientSafeOpen
        from vibevoice.modular.custom_offloading_utils import LayerOffloader
        from vibevoice.utils.model_utils import merge_lora_weights

        with init_empty_weights():
            model = cls(config)

        state_dict = {}
        if os.path.isdir(model_path):
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            print(f"Loading sharded model from {model_path}")
            state_dict = MultipleSafetensorLoader(index_path).load_dict()
        else:
            print(f"Loading model from single file {model_path}")
            with MemoryEfficientSafeOpen(model_path) as safe:
                for key in safe.keys():
                    state_dict[key] = safe.get_tensor(key)

        model.load_state_dict(state_dict, strict=False, assign=True)
        print("Model weights loaded.")

        if lora_model_path:
            model = merge_lora_weights(model, lora_model_path, lora_weight)

        if offload_config is not None and offload_config.enabled:
            print(f"Setting up layer offloading: {offload_config.num_layers_on_gpu} layers on GPU")
            model.lm_head.to(device)

            if model.model.language_model.embed_tokens.weight.dtype == torch.float8_e4m3fn:
                print("Converting embedding layer from float8 to bfloat16...")
                model.model.language_model.embed_tokens.cpu()
                model.model.language_model.embed_tokens.weight.data = \
                    model.model.language_model.embed_tokens.weight.data.to(torch.bfloat16)

            model.model.language_model.embed_tokens.to(device)
            model.model.language_model.norm.to(device)
            model.model.acoustic_tokenizer.to(device)
            model.model.semantic_tokenizer.to(device)
            model.model.acoustic_connector.to(device)
            model.model.semantic_connector.to(device)

            if offload_config.offload_prediction_head:
                model.model.prediction_head.cpu()
                print("Prediction head offloaded to CPU (on-demand transfer enabled)")
            else:
                model.model.prediction_head.to(device)

            model.offloader = LayerOffloader(
                language_model=model.model.language_model,
                config=offload_config,
                device=torch.device(device),
                logger=logger,
            )
            print(f"Layer offloading enabled: {len(model.offloader.offloaded_layers)} layers on CPU")
        else:
            model.to(device)
            print(f"Model moved to {device}")

        return model

    # ------------------------------------------------------------------
    # Generation infrastructure (adapted from VibeVoiceFusion)
    # ------------------------------------------------------------------

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig],
        use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        modified_values = {}
        default_generation_config = GenerationConfig()
        for key, default_value in default_generation_config.__dict__.items():
            if key.startswith("_") or key == "transformers_version":
                continue
            custom_gen_config_value = getattr(generation_config, key)
            model_gen_config_value = getattr(self.generation_config, key, default_value)
            if custom_gen_config_value == default_value and model_gen_config_value != default_value:
                modified_values[key] = model_gen_config_value
                setattr(generation_config, key, model_gen_config_value)

        model_kwargs = generation_config.update(**kwargs)
        return generation_config, model_kwargs

    def _prepare_special_tokens(
        self,
        generation_config: GenerationConfig,
        kwargs_has_attention_mask: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        def _tensor_or_none(token, device=None):
            if token is None:
                return token
            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)

        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        input_name = self.main_input_name
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(f"`inputs` and `{input_name}` cannot both be provided.")
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg
        return inputs, input_name, model_kwargs

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        if generation_config.use_cache is False:
            return
        model_kwargs["past_key_values"] = DynamicCache()

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        for possible_cache_name in ['past_key_values', 'cache_params', 'state', 'mems', 'past_buckets_states']:
            if possible_cache_name in outputs:
                cache_name = "past_key_values" if possible_cache_name in ("past_buckets_states", "mems") else possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1,
                dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        **kwargs,
    ) -> StoppingCriteriaList:
        from transformers.generation.stopping_criteria import MaxLengthCriteria, EosTokenCriteria
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            ))
        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        return criteria

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        model_inputs["cache_position"] = cache_position
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["inputs_embeds"] = None

        if (attention_mask is not None and kwargs.get("position_ids") is None
                and "position_ids" in set(inspect.signature(self.forward).parameters.keys())):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids

        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs["input_ids"].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        model_inputs["attention_mask"] = attention_mask
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        model_inputs.pop("labels", None)
        return model_inputs

    def _cache_dependant_input_preparation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        if inputs_embeds is not None and input_ids.shape[1] == 0:
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0]:]
        elif inputs_embeds is not None or (cache_position[-1] >= input_ids.shape[1]):
            input_ids = input_ids[:, -cache_position.shape[0]:]
        elif input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]
        return inputs_embeds, input_ids


# Register with HuggingFace AutoModel
from transformers.models.auto import AutoModelForCausalLM
AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGenerationInference)

__all__ = [
    "VibeVoiceForConditionalGenerationInference",
    "VibeVoiceGenerationOutput",
    "VibeVoiceCausalLMOutputWithPast",
]
