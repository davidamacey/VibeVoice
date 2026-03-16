import { getContext, setContext } from 'svelte';
import type { AppState } from './stores/app.svelte';
import type { AudioState } from './stores/audio.svelte';

const APP_KEY = Symbol('app');
const AUDIO_KEY = Symbol('audio');

export function setAppState(app: AppState) {
	setContext(APP_KEY, app);
}

export function getAppState(): AppState {
	return getContext<AppState>(APP_KEY);
}

export function setAudioState(audio: AudioState) {
	setContext(AUDIO_KEY, audio);
}

export function getAudioState(): AudioState {
	return getContext<AudioState>(AUDIO_KEY);
}
