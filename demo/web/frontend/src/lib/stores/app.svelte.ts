import type { ModelInfo, LogEntry, ConfigResponse } from '$lib/types';
import { formatLocalTimestamp, parseTimestamp } from '$lib/utils/format';
import { MAX_LOG_ENTRIES } from '$lib/constants';
import { fetchConfig } from '$lib/api';

export class AppState {
	currentModel = $state('0.5b-streaming');
	models = $state<ModelInfo[]>([]);
	voices = $state<string[]>([]);
	defaultVoice = $state('');
	logEntries = $state<LogEntry[]>([]);
	modelStatuses = $state<Record<string, boolean>>({});
	voicesLoading = $state(true);
	isModelLoading = $state(false);

	private logSequence = 0;

	get is0p5b() {
		return this.currentModel === '0.5b-streaming';
	}

	get is1p5bLike() {
		return this.currentModel === '1.5b' || this.currentModel === 'large';
	}

	get isASR() {
		return this.currentModel === 'asr';
	}

	get currentModelLoaded(): boolean | undefined {
		if (this.currentModel === '0.5b-streaming') return true;
		return this.modelStatuses[this.currentModel];
	}

	switchModel(modelId: string) {
		this.currentModel = modelId;
	}

	appendLog(message: string, timestamp?: string) {
		const finalTimestamp = timestamp || formatLocalTimestamp();
		const entry: LogEntry = {
			timestamp: finalTimestamp,
			date: parseTimestamp(finalTimestamp),
			message,
			seq: ++this.logSequence
		};
		this.logEntries = [...this.logEntries, entry]
			.sort((a, b) => {
				const diff = a.date.getTime() - b.date.getTime();
				return diff !== 0 ? diff : a.seq - b.seq;
			})
			.slice(-MAX_LOG_ENTRIES);
	}

	clearLogs() {
		this.logEntries = [];
	}

	async loadConfig() {
		try {
			this.voicesLoading = true;
			const data: ConfigResponse = await fetchConfig();

			if (Array.isArray(data.models)) {
				this.models = data.models;
				const statuses: Record<string, boolean> = {};
				data.models.forEach((m) => {
					statuses[m.id] = m.loaded !== false;
				});
				this.modelStatuses = statuses;
			}

			this.voices = Array.isArray(data.voices) ? data.voices : [];
			this.defaultVoice = data.default_voice || '';
			this.voicesLoading = false;

			if (this.voices.length > 0) {
				this.appendLog(`[Frontend] Loaded ${this.voices.length} voice presets`);
			} else {
				this.appendLog('[Error] No voice presets available');
			}
		} catch (err) {
			console.error('Failed to load config', err);
			this.voices = [];
			this.voicesLoading = false;
			this.appendLog('[Error] Failed to load voice presets');
		}
	}

	setModelLoaded(modelId: string, loaded: boolean) {
		this.modelStatuses = { ...this.modelStatuses, [modelId]: loaded };
	}

	async refreshModelStatus() {
		try {
			const response = await fetch('/models');
			if (!response.ok) return;
			const data = await response.json();
			if (Array.isArray(data.models)) {
				const statuses: Record<string, boolean> = {};
				data.models.forEach((m: ModelInfo) => {
					statuses[m.id] = m.loaded !== false;
				});
				this.modelStatuses = statuses;
				// Also update the models array loaded flags
				this.models = this.models.map((m) => ({
					...m,
					loaded: statuses[m.id] ?? m.loaded
				}));
			}
		} catch {
			// Silent fail — status poll is best-effort
		}
	}
}
