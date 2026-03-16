<script lang="ts">
	import type { AppState } from '$lib/stores/app.svelte';
	import ModelBadge from './shared/ModelBadge.svelte';

	let { app }: { app: AppState } = $props();

	function handleChange(e: Event) {
		const select = e.target as HTMLSelectElement;
		app.switchModel(select.value);
	}
</script>

<section class="panel">
	<div class="inline-field" style="flex-wrap:wrap; gap:10px; align-items:center; flex-direction:row">
		<span class="field-label">Model</span>
		<select class="select-control" value={app.currentModel} onchange={handleChange}>
			{#if app.models.length === 0}
				<option value="0.5b-streaming">VibeVoice Realtime 0.5B (streaming)</option>
			{:else}
				{#each app.models as model}
					<option value={model.id}>
						{model.name}{#if model.id !== '0.5b-streaming'}{model.loaded ? ' \u2713' : ' (not loaded)'}{/if}
					</option>
				{/each}
			{/if}
		</select>
		<ModelBadge
			loaded={app.currentModelLoaded}
			isLoading={app.isModelLoading}
			isStreaming={app.is0p5b}
		/>
	</div>
</section>
