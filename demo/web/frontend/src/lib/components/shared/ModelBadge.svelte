<script lang="ts">
  let { loaded, isLoading = false, isStreaming = false }: {
    loaded?: boolean;
    isLoading?: boolean;
    isStreaming?: boolean;
  } = $props();
</script>

{#if isLoading}
  <span class="model-badge model-badge--loading">&#8987; loading model&hellip;</span>
{:else if isStreaming || loaded === true}
  <span class="model-badge model-badge--loaded">&#10003; loaded</span>
{:else if loaded === false}
  <span class="model-badge model-badge--unloaded">not loaded</span>
{/if}
