<script lang="ts">
  import { untrack } from 'svelte';
  import { STREAMING_INTERVAL_MS } from '$lib/constants';

  let { text, active }: { text: string; active: boolean } = $props();

  let displayedText = $state('');
  let tokens = $state<string[]>([]);
  let tokenIndex = $state(0);
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  function clearTimer() {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
  }

  function tick() {
    if (tokenIndex >= tokens.length) return;
    displayedText += tokens[tokenIndex];
    tokenIndex++;
    if (tokenIndex < tokens.length) {
      timeoutId = setTimeout(tick, STREAMING_INTERVAL_MS);
    }
  }

  $effect(() => {
    const isActive = active;
    const currentText = text;

    untrack(() => {
      if (isActive && currentText) {
        clearTimer();
        displayedText = '';
        tokens = currentText.trimEnd().match(/\S+\s*/g) || [];
        tokenIndex = 0;
        if (tokens.length > 0) {
          timeoutId = setTimeout(tick, 0);
        }
      } else if (!isActive) {
        clearTimer();
      }
    });

    return () => clearTimer();
  });
</script>

<div class="streaming-preview-container">
  <div class="streaming-preview-header">
    <span>Streaming Input Text</span>
  </div>
  <div class="streaming-preview" class:streaming-active={active && tokenIndex < tokens.length}>
    {#if displayedText}
      {displayedText}
    {:else if !active}
      <span style="color: var(--text-muted)">Text will appear here word-by-word during playback</span>
    {/if}
  </div>
</div>
