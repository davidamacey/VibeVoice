import type {
	ConfigResponse,
	AsrResponse,
	GenerateProgressEvent,
	TranscribeProgressEvent,
	SimilarityScores
} from './types';

export async function fetchConfig(): Promise<ConfigResponse> {
	const response = await fetch('/config');
	if (!response.ok) {
		throw new Error(`Failed to fetch config: ${response.status}`);
	}
	return response.json();
}

/**
 * Parse a Server-Sent Events stream, calling onEvent for each parsed event.
 * Returns when the stream ends.
 */
async function readSSEStream<T>(
	response: Response,
	onEvent: (event: T) => void
): Promise<void> {
	const reader = response.body!.getReader();
	const decoder = new TextDecoder();
	let buffer = '';

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		buffer += decoder.decode(value, { stream: true });
		const lines = buffer.split('\n');
		buffer = lines.pop() || '';

		for (const line of lines) {
			if (!line.startsWith('data: ')) continue;
			try {
				onEvent(JSON.parse(line.slice(6)));
			} catch {
				// skip malformed lines
			}
		}
	}
}

export interface GenerateResult {
	audioBlob: Blob;
	similarity?: SimilarityScores;
}

export async function postGenerate(
	formData: FormData,
	onProgress?: (event: GenerateProgressEvent) => void
): Promise<GenerateResult> {
	const response = await fetch('/generate', { method: 'POST', body: formData });
	if (!response.ok) {
		const errText = await response.text();
		throw new Error(`HTTP ${response.status}: ${errText}`);
	}

	let audioBlob: Blob | null = null;
	let similarity: SimilarityScores | undefined;
	let errorMsg: string | null = null;

	await readSSEStream<GenerateProgressEvent>(response, (event) => {
		if (event.type === 'error') {
			errorMsg = event.message || 'Generation failed';
		} else if (event.type === 'similarity' && event.similarity) {
			similarity = event.similarity;
		} else if (event.type === 'complete' && event.audio) {
			const binary = atob(event.audio);
			const bytes = new Uint8Array(binary.length);
			for (let i = 0; i < binary.length; i++) {
				bytes[i] = binary.charCodeAt(i);
			}
			audioBlob = new Blob([bytes], { type: 'audio/wav' });
		}
		onProgress?.(event);
	});

	if (errorMsg) throw new Error(errorMsg);
	if (!audioBlob) throw new Error('No audio received from server');
	return { audioBlob, similarity };
}

export async function postTranscribe(
	formData: FormData,
	onProgress?: (event: TranscribeProgressEvent) => void
): Promise<AsrResponse> {
	const response = await fetch('/transcribe', { method: 'POST', body: formData });
	if (!response.ok) {
		const errText = await response.text();
		throw new Error(`HTTP ${response.status}: ${errText}`);
	}

	let result: AsrResponse | null = null;
	let errorMsg: string | null = null;

	await readSSEStream<TranscribeProgressEvent>(response, (event) => {
		if (event.type === 'error') {
			errorMsg = event.message || 'Transcription failed';
		} else if (event.type === 'complete' && event.result) {
			result = event.result;
		}
		onProgress?.(event);
	});

	if (errorMsg) throw new Error(errorMsg);
	if (!result) throw new Error('No transcription result received');
	return result;
}
