<script lang="ts">
  let { label, tipText = '', checked = $bindable() }: {
    label: string;
    tipText?: string;
    checked: boolean;
  } = $props();
</script>

<span class="toggle-row">
  <label class="toggle-switch">
    <input type="checkbox" bind:checked />
    <span class="toggle-track"></span>
  </label>
  <span class="tip">{label}
    {#if tipText}
      <span class="tip-icon">ⓘ</span>
      <span class="tip-box">{@html tipText}</span>
    {/if}
  </span>
</span>
