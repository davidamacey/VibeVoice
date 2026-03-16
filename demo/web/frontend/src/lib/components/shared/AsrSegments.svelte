<script lang="ts">
  import type { AsrSegment } from '$lib/types';

  let { segments }: { segments: AsrSegment[] } = $props();
</script>

{#if segments.length > 0}
  <div style="margin-top:8px">
    <span class="field-label" style="font-weight:500; font-size:13px">Segments</span>
    <div class="asr-segments">
      {#each segments as seg}
        <div class="asr-segment">
          <span class="asr-segment-time">
            {seg.start_time != null ? Number(seg.start_time).toFixed(2) : '?'}s &ndash; {seg.end_time != null ? Number(seg.end_time).toFixed(2) : '?'}s{seg.speaker_id != null ? ` [spk ${seg.speaker_id}]` : ''}
          </span>
          <span>{seg.text || ''}</span>
        </div>
      {/each}
    </div>
  </div>
{/if}
