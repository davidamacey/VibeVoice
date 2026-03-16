<script lang="ts">
  import type { SimilarityScores } from '$lib/types';

  let { scores }: { scores: SimilarityScores } = $props();

  function qualityLabel(score: number): string {
    if (score >= 0.85) return 'Excellent';
    if (score >= 0.70) return 'Good';
    if (score >= 0.50) return 'Fair';
    return 'Low';
  }

  function scoreColor(score: number): string {
    if (score >= 0.85) return '#27ae60';
    if (score >= 0.70) return '#f39c12';
    if (score >= 0.50) return '#e67e22';
    return '#c0392b';
  }

  function pct(v: number): string {
    return (v * 100).toFixed(1) + '%';
  }
</script>

<div class="metrics" style="margin-top:8px; flex-direction:column; gap:2px">
  <span>
    Voice Match
    <strong style="color: {scoreColor(scores.ensemble)}">{pct(scores.ensemble)}</strong>
    <span class="quality-badge" style="color: {scoreColor(scores.ensemble)}">[{qualityLabel(scores.ensemble)}]</span>
  </span>
  <span class="metric-unit" style="font-size:11px">
    {#if scores.wespeaker != null}WeSpeaker: {pct(scores.wespeaker)}{/if}
    {#if scores.wespeaker != null && scores.librosa != null} &nbsp;|&nbsp; {/if}
    Librosa: {pct(scores.librosa)}
  </span>
</div>
