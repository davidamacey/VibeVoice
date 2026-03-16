<script lang="ts">
  let { speakerId, onfilechange }: {
    speakerId: number;
    onfilechange: (file: File | null) => void;
  } = $props();

  function handleChange(e: Event) {
    const input = e.target as HTMLInputElement;
    onfilechange(input.files?.[0] || null);
  }
</script>

<label class="inline-field" style="flex-direction:row; align-items:center; gap:8px; font-size:13px">
  <span style="min-width:80px; font-weight:500; color:var(--text-muted)">Speaker {speakerId}:</span>
  <input type="file" accept="audio/*" style="font-size:12px" onchange={handleChange}>
</label>
