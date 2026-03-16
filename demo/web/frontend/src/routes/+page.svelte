<script lang="ts">
	import { getAppState, getAudioState } from '$lib/context';
	import Header from '$lib/components/Header.svelte';
	import ModelSelector from '$lib/components/ModelSelector.svelte';
	import StreamingPanel from '$lib/components/StreamingPanel.svelte';
	import GeneratePanel from '$lib/components/GeneratePanel.svelte';
	import AsrPanel from '$lib/components/AsrPanel.svelte';

	const app = getAppState();
	const audio = getAudioState();
</script>

<Header />
<ModelSelector {app} />

{#if app.is0p5b}
	<StreamingPanel {app} {audio} />
{:else if app.is1p5bLike}
	<GeneratePanel {app} />
{:else if app.isASR}
	<AsrPanel {app} />
{/if}
