import { SAMPLE_RATE, BUFFER_SIZE } from '$lib/constants';
import type { WordTiming } from '$lib/types';
import type { AppState } from './app.svelte';
import { createWavBlob, triggerDownload } from '$lib/utils/wav';
import { formatSeconds } from '$lib/utils/format';

export class AudioState {
	// Reactive state
	isPlaying = $state(false);
	isRecordingComplete = $state(false);
	recordedSamples = $state(0);
	modelGenerated = $state(0);
	playbackElapsed = $state(0);
	wordTimings = $state<WordTiming[] | null>(null);
	chunksReceived = $state(0);
	connectionStatus = $state<'idle' | 'connecting' | 'connected' | 'closed' | 'error'>('idle');

	// Non-reactive (performance-critical)
	audioCtx: AudioContext | null = null;
	scriptNode: ScriptProcessorNode | null = null;
	socket: WebSocket | null = null;
	buffer = new Float32Array(0);
	recordedChunks: ArrayBuffer[] = [];
	private hasStartedPlayback = false;
	private silentFrameCount = 0;
	private firstBrowserChunkLogged = false;
	private playbackStartedLogged = false;
	private playbackSamples = 0;
	private playbackTimer: ReturnType<typeof setInterval> | null = null;
	private downloadUrl: string | null = null;
	private app: AppState;
	private generationStartTime = 0;
	private prebufferSec = 0.1;

	constructor(app: AppState) {
		this.app = app;
	}

	get canSave(): boolean {
		return this.recordedSamples > 0 && this.isRecordingComplete;
	}

	get modelGeneratedFormatted(): string {
		return formatSeconds(this.modelGenerated);
	}

	get playbackElapsedFormatted(): string {
		return formatSeconds(this.playbackElapsed);
	}

	get rtf(): string {
		if (this.generationStartTime <= 0 || this.modelGenerated <= 0) return '-';
		const wallClock = (performance.now() - this.generationStartTime) / 1000;
		if (wallClock < 0.1) return '-';
		return (this.modelGenerated / wallClock).toFixed(2);
	}

	setModelGenerated(value: number) {
		const numeric = Number(value);
		if (!Number.isFinite(numeric)) return;
		this.modelGenerated = Math.max(0, numeric);
	}

	private setPlaybackElapsed(value: number) {
		this.playbackElapsed = Math.min(this.modelGenerated, Math.max(0, value));
	}

	private appendAudio(chunk: Float32Array) {
		const merged = new Float32Array(this.buffer.length + chunk.length);
		merged.set(this.buffer, 0);
		merged.set(chunk, this.buffer.length);
		this.buffer = merged;
	}

	private pullAudio(frameCount: number): Float32Array {
		const available = this.buffer.length;
		if (available === 0) return new Float32Array(frameCount);
		if (available <= frameCount) {
			const chunk = this.buffer;
			this.buffer = new Float32Array(0);
			if (chunk.length < frameCount) {
				const padded = new Float32Array(frameCount);
				padded.set(chunk, 0);
				return padded;
			}
			return chunk;
		}
		const chunk = this.buffer.subarray(0, frameCount);
		this.buffer = this.buffer.subarray(frameCount);
		return chunk;
	}

	private closeSocket() {
		if (
			this.socket &&
			(this.socket.readyState === WebSocket.OPEN ||
				this.socket.readyState === WebSocket.CONNECTING)
		) {
			this.socket.close();
		}
		this.socket = null;
	}

	private teardownAudio() {
		if (this.scriptNode) {
			try {
				this.scriptNode.disconnect();
			} catch (_) {}
			this.scriptNode.onaudioprocess = null;
		}
		if (this.audioCtx) {
			try {
				this.audioCtx.close();
			} catch (_) {}
		}
		this.audioCtx = null;
		this.scriptNode = null;
	}

	private resetPlaybackFlags(resetSamples = true) {
		this.buffer = new Float32Array(0);
		if (resetSamples) {
			this.playbackSamples = 0;
			this.setPlaybackElapsed(0);
		}
		this.hasStartedPlayback = false;
		this.silentFrameCount = 0;
		this.firstBrowserChunkLogged = false;
		this.playbackStartedLogged = false;
	}

	private stopPlaybackTimer() {
		if (this.playbackTimer) {
			clearInterval(this.playbackTimer);
			this.playbackTimer = null;
		}
	}

	private startPlaybackTimer() {
		this.stopPlaybackTimer();
		this.playbackTimer = setInterval(() => {
			this.setPlaybackElapsed(this.playbackSamples / SAMPLE_RATE);
		}, 250);
	}

	private clearRecordedChunks() {
		this.recordedChunks = [];
		this.recordedSamples = 0;
		this.isRecordingComplete = false;
		this.revokeDownloadUrl();
	}

	private revokeDownloadUrl() {
		if (this.downloadUrl) {
			URL.revokeObjectURL(this.downloadUrl);
			this.downloadUrl = null;
		}
	}

	private resetState(resetSamples = true) {
		this.closeSocket();
		this.teardownAudio();
		this.resetPlaybackFlags(resetSamples);
		this.isPlaying = false;
		this.stopPlaybackTimer();
	}

	private createAudioChain() {
		this.teardownAudio();
		this.resetPlaybackFlags();

		const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
		this.audioCtx = new AudioCtxClass({ sampleRate: SAMPLE_RATE });

		// Browsers may start AudioContext in suspended state — resume immediately
		if (this.audioCtx.state === 'suspended') {
			this.audioCtx.resume();
		}

		this.scriptNode = this.audioCtx.createScriptProcessor(BUFFER_SIZE, 0, 1);

		const minBufferSamples = Math.floor(this.audioCtx.sampleRate * this.prebufferSec);

		this.scriptNode.onaudioprocess = (event) => {
			const output = event.outputBuffer.getChannelData(0);
			const needPrebuffer = !this.hasStartedPlayback;
			const socketClosed =
				!this.socket ||
				this.socket.readyState === WebSocket.CLOSED ||
				this.socket.readyState === WebSocket.CLOSING;

			if (needPrebuffer) {
				if (this.buffer.length >= minBufferSamples || socketClosed) {
					this.hasStartedPlayback = true;
					if (!this.playbackStartedLogged) {
						this.playbackStartedLogged = true;
						this.app.appendLog('[Frontend] Browser started to play audio');
						this.startPlaybackTimer();
					}
				} else {
					output.fill(0);
					return;
				}
			}

			const chunk = this.pullAudio(output.length);
			output.set(chunk);

			if (this.hasStartedPlayback) {
				this.playbackSamples += output.length;
			}

			if (socketClosed && this.buffer.length === 0 && chunk.every((s) => s === 0)) {
				this.silentFrameCount += 1;
				if (this.silentFrameCount >= 4) {
					this.stop();
				}
			} else {
				this.silentFrameCount = 0;
			}
		};

		this.scriptNode.connect(this.audioCtx.destination);
	}

	private handleLogMessage(raw: string) {
		let payload: any;
		try {
			payload = JSON.parse(raw);
		} catch {
			this.app.appendLog(`[Error] Failed to parse log message: ${raw}`);
			return;
		}
		if (!payload || payload.type !== 'log') {
			this.app.appendLog(`[Log] ${raw}`);
			return;
		}

		const { event, data = {}, timestamp } = payload;
		switch (event) {
			case 'backend_request_received':
				this.app.appendLog('[Backend]  Received request', timestamp);
				break;
			case 'backend_first_chunk_sent':
				this.app.appendLog('[Backend]  Sent first audio chunk', timestamp);
				break;
			case 'model_progress':
				if (typeof data.generated_sec !== 'undefined') {
					const generated = Number(data.generated_sec);
					if (Number.isFinite(generated)) {
						this.setModelGenerated(generated);
					}
				}
				return;
			case 'word_timing': {
				const words = data.words || [];
				if (words.length > 0) {
					this.wordTimings = words;
					this.app.appendLog(
						`[Backend]  Word timestamps: ${words.length} words`,
						timestamp
					);
				}
				break;
			}
			case 'generation_error':
				this.app.appendLog(
					`[Error] Generation error: ${data.message || 'Unknown error'}`,
					timestamp
				);
				break;
			case 'backend_error':
				this.app.appendLog(
					`[Error] Backend error: ${data.message || 'Unknown error'}`,
					timestamp
				);
				break;
			case 'client_disconnected':
				this.app.appendLog('[Frontend] Client disconnected', timestamp);
				break;
			case 'backend_stream_complete':
				this.app.appendLog('[Backend]  Backend finished', timestamp);
				this.isRecordingComplete = true;
				break;
			default:
				this.app.appendLog(`[Log] Event ${event}`, timestamp);
				break;
		}
	}

	start(params: {
		text: string;
		cfg: number;
		steps: number;
		voice: string;
		doSample: boolean;
		temperature: number;
		topP: number;
	}) {
		if (this.isPlaying) return;

		this.app.clearLogs();
		this.wordTimings = null;

		const cfgDisplay = Number.isFinite(params.cfg) ? params.cfg.toFixed(3) : 'default';
		const stepsDisplay = Number.isFinite(params.steps) ? params.steps : 'default';
		this.app.appendLog(
			`[Frontend] Start button clicked, CFG=${cfgDisplay}, Steps=${stepsDisplay}, Speaker=${params.voice || 'default'}`
		);
		this.setModelGenerated(0);
		this.setPlaybackElapsed(0);
		this.generationStartTime = performance.now();

		// Adaptive prebuffer: more inference steps = slower generation = need more buffer
		// steps <= 5: 0.1s, steps 10: 0.3s, steps 15: 0.6s, steps 20: 1.0s
		this.prebufferSec = Math.min(1.0, Math.max(0.1, (params.steps - 5) * 0.06 + 0.1));

		this.resetState(true);
		this.clearRecordedChunks();
		this.chunksReceived = 0;
		this.connectionStatus = 'connecting';
		this.isPlaying = true;
		this.createAudioChain();

		const urlParams = new URLSearchParams();
		urlParams.set('text', params.text);
		if (!Number.isNaN(params.cfg)) urlParams.set('cfg', params.cfg.toFixed(3));
		if (!Number.isNaN(params.steps)) urlParams.set('steps', params.steps.toString());
		if (params.voice) urlParams.set('voice', params.voice);
		urlParams.set('do_sample', params.doSample ? 'true' : 'false');
		if (params.doSample) {
			urlParams.set('temperature', params.temperature.toFixed(2));
			urlParams.set('top_p', params.topP.toFixed(2));
		}

		const wsUrl = `${location.origin.replace(/^http/, 'ws')}/stream?${urlParams.toString()}`;
		this.socket = new WebSocket(wsUrl);
		this.socket.binaryType = 'arraybuffer';

		this.socket.onopen = () => {
			this.connectionStatus = 'connected';
		};

		this.socket.onmessage = (event) => {
			if (typeof event.data === 'string') {
				this.handleLogMessage(event.data);
				return;
			}

			if (!(event.data instanceof ArrayBuffer)) return;

			const rawBuffer = event.data.slice(0);
			const view = new DataView(rawBuffer);
			const floatChunk = new Float32Array(view.byteLength / 2);
			for (let i = 0; i < floatChunk.length; i++) {
				floatChunk[i] = view.getInt16(i * 2, true) / 32768;
			}
			this.appendAudio(floatChunk);
			this.recordedChunks.push(rawBuffer);
			this.recordedSamples += floatChunk.length;
			this.chunksReceived++;

			if (!this.firstBrowserChunkLogged) {
				this.firstBrowserChunkLogged = true;
				this.app.appendLog('[Frontend] Received first audio chunk');
			}
		};

		this.socket.onerror = (err: any) => {
			console.error('WebSocket error', err);
			this.connectionStatus = 'error';
			this.app.appendLog(`[Error] WebSocket error: ${err?.message || err}`);
			this.stop();
		};

		this.socket.onclose = () => {
			this.socket = null;
			this.connectionStatus = 'closed';
			if (this.recordedSamples > 0) {
				this.isRecordingComplete = true;
			}
		};
	}

	stop() {
		if (!this.isPlaying) {
			this.resetState(false);
			return;
		}
		this.resetState(false);
		this.setPlaybackElapsed(
			Math.min(this.playbackElapsed, this.modelGenerated)
		);
		this.app.appendLog('[Frontend] Playback stopped');
		if (this.recordedSamples > 0) {
			this.isRecordingComplete = true;
		}
	}

	saveRecording() {
		if (!this.recordedSamples) {
			this.app.appendLog('[Frontend] Save requested but no audio received yet');
			return;
		}
		const wavBlob = createWavBlob(this.recordedChunks, this.recordedSamples);
		if (!wavBlob) {
			this.app.appendLog('[Error] Failed to assemble WAV data for download');
			return;
		}
		const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
		triggerDownload(wavBlob, `vibevoice_realtime_audio_${timestamp}.wav`);
		this.app.appendLog('[Frontend] Audio download triggered');
	}

	cleanup() {
		this.resetState();
		this.revokeDownloadUrl();
	}
}
