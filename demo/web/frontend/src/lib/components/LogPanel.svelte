<script lang="ts">
	import type { AppState } from '$lib/stores/app.svelte';

	let { app }: { app: AppState } = $props();

	let logEl: HTMLPreElement;

	$effect(() => {
		// Track logEntries changes to auto-scroll
		app.logEntries;
		if (logEl) {
			logEl.scrollTop = logEl.scrollHeight;
		}
	});

	function formatLog(): string {
		return app.logEntries
			.map((item) => `[${item.timestamp}] ${item.message}`)
			.join('\n');
	}
</script>

<section class="panel">
	<span class="field-label">Runtime Logs</span>
	<pre class="log-output" bind:this={logEl}>{formatLog()}</pre>
</section>
