<script lang="ts">
  import type { WordTiming } from '$lib/types';
  import { formatTs } from '$lib/utils/format';

  let { wordTimings }: { wordTimings: WordTiming[] | null } = $props();

  const LINE_SIZE = 8;

  function getGroups(words: WordTiming[]) {
    const groups: { start: number; end: number; text: string }[] = [];
    for (let i = 0; i < words.length; i += LINE_SIZE) {
      const group = words.slice(i, i + LINE_SIZE);
      groups.push({
        start: group[0].start_time,
        end: group[group.length - 1].end_time,
        text: group.map(w => w.word || '').join(' ')
      });
    }
    return groups;
  }

  function downloadTimestamps() {
    if (!wordTimings) return;
    const blob = new Blob([JSON.stringify(wordTimings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'word_timestamps.json';
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

{#if wordTimings && wordTimings.length > 0}
  <section class="panel">
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px">
      <span class="field-label">Transcript</span>
      <button type="button" class="secondary-btn" style="font-size:12px; padding:4px 12px" onclick={downloadTimestamps}>Download Timestamps JSON</button>
    </div>
    <div class="transcript-display">
      {#each getGroups(wordTimings) as group}
        <div class="transcript-line">
          <span class="transcript-time">{formatTs(group.start)} &rarr; {formatTs(group.end)}</span>
          <span class="transcript-text">{group.text}</span>
        </div>
      {/each}
    </div>
  </section>
{/if}
