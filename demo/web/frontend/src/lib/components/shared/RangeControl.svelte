<script lang="ts">
  let { label, tipText = '', min, max, step, value = $bindable(), formatValue = (v: number) => String(v) }: {
    label: string;
    tipText?: string;
    min: number;
    max: number;
    step: number;
    value: number;
    formatValue?: (v: number) => string;
  } = $props();
</script>

<label class="range-control">
  <span class="tip">{label}
    {#if tipText}
      <span class="tip-icon">ⓘ</span>
      <span class="tip-box">{@html tipText}</span>
    {/if}
  </span>
  <input type="range" {min} {max} {step} bind:value />
  <span class="range-value">{formatValue(value)}</span>
</label>
