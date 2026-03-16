<script lang="ts">
	let { onclose }: { onclose: () => void } = $props();

	function handleBackdrop(e: MouseEvent) {
		if (e.target === e.currentTarget) onclose();
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="modal-backdrop" onclick={handleBackdrop} role="presentation">
	<div class="modal-content">
		<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px">
			<h2 style="margin:0; font-size:20px">About VibeVoice Demo</h2>
			<button type="button" class="speaker-remove-btn" onclick={onclose} style="font-size:18px">&times;</button>
		</div>

		<div style="display:flex; flex-direction:column; gap:14px; font-size:14px; line-height:1.7; color:var(--text-primary)">
			<section>
				<h3 style="margin:0 0 4px; font-size:15px">Project</h3>
				<p style="margin:0; color:var(--text-muted)">
					This is a community fork of <strong>Microsoft's VibeVoice</strong> open-source voice AI project,
					providing a web-based demo for text-to-speech and speech recognition.
				</p>
			</section>

			<section>
				<h3 style="margin:0 0 4px; font-size:15px">Available Models</h3>
				<ul style="margin:4px 0 0; padding-left:20px; color:var(--text-muted)">
					<li><strong>VibeVoice Realtime 0.5B</strong> &mdash; streaming TTS with low latency (~200ms TTFA)</li>
					<li><strong>VibeVoice TTS 1.5B</strong> &mdash; higher quality TTS with multi-speaker and voice cloning</li>
					<li><strong>VibeVoice Large</strong> &mdash; 7B-backbone TTS, most stable for long-form and Chinese text</li>
					<li><strong>VibeVoice ASR 7B</strong> &mdash; speech recognition with word-level timestamps</li>
				</ul>
			</section>

			<section>
				<h3 style="margin:0 0 4px; font-size:15px">Voice Cloning</h3>
				<div style="background:#fef3c7; border:1px solid #fde68a; border-radius:8px; padding:10px 12px; color:#92400e; font-size:13px">
					<strong>Important:</strong> Voice cloning capabilities (1.5B/Large models) are intended for
					authorized use only &mdash; personal projects, accessibility, creative work, and research.
					Do not use voice cloning to impersonate others without consent, create deepfakes, or for
					any fraudulent or harmful purpose. Misuse may violate local laws.
				</div>
			</section>

			<section>
				<h3 style="margin:0 0 4px; font-size:15px">Technical Notes</h3>
				<ul style="margin:4px 0 0; padding-left:20px; color:var(--text-muted); font-size:13px">
					<li>Architecture: Qwen2.5 LLM backbone + continuous speech tokenizers (7.5 Hz) + diffusion head</li>
					<li>0.5B streaming uses pre-computed .pt voice files (prefilled KV-cache states)</li>
					<li>1.5B/Large support zero-shot voice cloning via raw audio references</li>
					<li>Speed control uses pitch-preserving resampling when voice references are provided</li>
				</ul>
			</section>
		</div>
	</div>
</div>

<style>
	.modal-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(31, 39, 66, 0.45);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
		padding: 20px;
	}

	.modal-content {
		background: var(--surface);
		border-radius: 16px;
		padding: 28px 32px;
		max-width: 560px;
		width: 100%;
		max-height: 80vh;
		overflow-y: auto;
		box-shadow: 0 20px 60px rgba(31, 39, 66, 0.2);
	}
</style>
