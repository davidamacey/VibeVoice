<script lang="ts">
	import AboutModal from './shared/AboutModal.svelte';

	let showAbout = $state(false);
</script>

<div style="display:flex; align-items:center; justify-content:space-between">
	<div></div>
	<h1>VibeVoice Demo</h1>
	<div style="display:flex; gap:8px; align-items:center">
		<a href="/logs" class="secondary-btn" style="text-decoration:none; font-size:12px; padding:5px 12px">Logs</a>
		<button type="button" class="secondary-btn" style="font-size:12px; padding:5px 12px" onclick={() => showAbout = true}>About</button>
	</div>
</div>

{#if showAbout}
	<AboutModal onclose={() => showAbout = false} />
{/if}
