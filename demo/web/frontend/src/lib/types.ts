export interface ModelInfo {
	id: string;
	name: string;
	loaded: boolean;
}

export interface LogEntry {
	timestamp: string;
	date: Date;
	message: string;
	seq: number;
}

export interface StreamingParams {
	text: string;
	cfg: number;
	steps: number;
	voice: string;
	doSample: boolean;
	temperature: number;
	topP: number;
}

export interface WordTiming {
	word: string;
	start_time: number;
	end_time: number;
}

export interface SpeakerTurn {
	speakerId: number;
	text: string;
}

export interface AsrSegment {
	text: string;
	start_time: number;
	end_time: number;
	speaker_id?: number;
}

export interface ConfigResponse {
	voices: string[];
	default_voice: string;
	models: ModelInfo[];
}

export interface AsrResponse {
	text: string;
	segments?: AsrSegment[];
}

export interface SimilarityScores {
	wespeaker?: number;
	ecapa_tdnn?: number;
	librosa: number;
	ensemble: number;
}

export interface GenerateProgressEvent {
	type: 'loading' | 'loaded' | 'progress' | 'step' | 'similarity' | 'complete' | 'error';
	model_id?: string;
	chunk?: number;
	total_chunks?: number;
	step?: number;
	total_steps?: number;
	audio?: string;
	similarity?: SimilarityScores;
	message?: string;
}

export interface TranscribeProgressEvent {
	type: 'loading' | 'loaded' | 'progress' | 'complete' | 'error';
	tokens_generated?: number;
	max_tokens?: number;
	result?: AsrResponse;
	audio_duration?: number;
	message?: string;
}
