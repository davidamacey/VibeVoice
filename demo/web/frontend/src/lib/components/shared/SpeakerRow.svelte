<script lang="ts">
  let { speakerId = $bindable(), text = $bindable(), canRemove, onremove, onfocus, index }: {
    speakerId: number;
    text: string;
    canRemove: boolean;
    onremove: () => void;
    onfocus: (el: HTMLTextAreaElement, index: number) => void;
    index: number;
  } = $props();

  let textareaEl: HTMLTextAreaElement;
</script>

<div class="speaker-row">
  <div class="speaker-label-col">
    <select class="speaker-select" bind:value={speakerId}>
      {#each [0, 1, 2, 3] as id}
        <option value={id}>Speaker {id}</option>
      {/each}
    </select>
  </div>
  <textarea
    class="speaker-textarea"
    rows="3"
    placeholder="Speaker {speakerId} dialogue..."
    bind:value={text}
    bind:this={textareaEl}
    onfocus={() => onfocus(textareaEl, index)}
  ></textarea>
  <button type="button" class="speaker-remove-btn" title="Remove this speaker turn" onclick={onremove} disabled={!canRemove}>
    &#10005;
  </button>
</div>
