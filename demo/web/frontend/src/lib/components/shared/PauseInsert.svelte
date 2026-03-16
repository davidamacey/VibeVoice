<script lang="ts">
  let { oninsert }: { oninsert: (tag: string) => void } = $props();

  let customSec = $state(0.5);

  const presets = ['0.3s', '0.5s', '1s', '2s'];

  function insertCustom() {
    const val = Number.isFinite(customSec) && customSec > 0 ? customSec : 0.5;
    oninsert(`[pause:${val}s]`);
  }
</script>

<div class="field" style="margin-top:4px">
  <span class="field-label" style="font-size:13px; font-weight:500">
    Insert Pause at Cursor
    <span class="tip"><span class="tip-icon">ⓘ</span>
      <span class="tip-box">Inserts a timed silence at your cursor position in the focused speaker text. The text is split at each [pause:Xs] marker: each segment is generated separately and silence is stitched between them.<br><br>Natural pauses from punctuation (<code>...</code>, <code>,</code>, <code>—</code>) still work on top of this.</span>
    </span>
  </span>
  <div class="pause-btn-row">
    <span style="font-size:12px; color:var(--text-muted)">Quick:</span>
    {#each presets as p}
      <button type="button" class="pause-btn" onclick={() => oninsert(`[pause:${p}]`)}>[pause:{p}]</button>
    {/each}
    <span style="font-size:12px; color:var(--text-muted); margin-left:6px">Custom (sec):</span>
    <input type="number" min="0.1" max="10" step="0.1" bind:value={customSec}
      style="width:62px; border:1px solid rgba(31,39,66,0.18); border-radius:6px; padding:3px 6px; font-size:12px">
    <button type="button" class="pause-btn" onclick={insertCustom}>[pause:custom]</button>
  </div>
</div>
