<script lang="ts">
	import '../app.css';
	import { AppState } from '$lib/stores/app.svelte';
	import { AudioState } from '$lib/stores/audio.svelte';
	import { setAppState, setAudioState } from '$lib/context';
	import { onMount } from 'svelte';

	let { children } = $props();

	const app = new AppState();
	const audio = new AudioState(app);

	setAppState(app);
	setAudioState(audio);

	onMount(() => {
		app.loadConfig();
		return () => audio.cleanup();
	});
</script>

<div class="app-shell">
	{@render children()}
</div>
