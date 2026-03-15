from abc import ABC, abstractmethod
from typing import List


class GenerationVisitor(ABC):

    @abstractmethod
    def visit_preprocessing(self, timestamp: float = None):
        pass

    @abstractmethod
    def visit_inference_start(self, scripts: List[str] = None,
                              unique_speaker_names: List[str] = None,
                              voice_sample: List[str] = None,
                              max_speaker_id: int = None):
        pass

    @abstractmethod
    def visit_inference_batch_start(self, batch_index: int, seeds: int):
        pass

    @abstractmethod
    def visit_inference_batch_end(self, batch_index: int):
        pass

    @abstractmethod
    def visit_inference_save_audio_file(self, output_audio_path: str = None,
                                        generation_time: float = None,
                                        prefilling_tokens: int = None,
                                        total_tokens: int = None,
                                        generated_tokens: int = None,
                                        audio_duration_seconds: float = None,
                                        real_time_factor: float = None,
                                        **kwargs):
        pass

    @abstractmethod
    def visit_inference_step_start(self, current_step: int, total_steps: int):
        pass

    @abstractmethod
    def visit_inference_step_end(self, current_step: int, total_steps: int):
        pass

    @abstractmethod
    def visit_completed(self, message: str = None):
        pass

    @abstractmethod
    def visit_failed(self, message: str, failure_type: str):
        pass
