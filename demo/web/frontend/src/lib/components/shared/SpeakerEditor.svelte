<script lang="ts">
  import type { SpeakerTurn } from '$lib/types';
  import SpeakerRow from './SpeakerRow.svelte';

  let { turns = $bindable(), onfocustextarea }: {
    turns: SpeakerTurn[];
    onfocustextarea: (el: HTMLTextAreaElement, index: number) => void;
  } = $props();

  function addTurn() {
    const lastId = turns.length > 0 ? turns[turns.length - 1].speakerId : -1;
    const nextId = (lastId + 1) % 4;
    turns = [...turns, { speakerId: nextId, text: '' }];
  }

  function removeTurn(index: number) {
    if (turns.length <= 1) return;
    turns = turns.filter((_, i) => i !== index);
  }
</script>

<div class="field">
  <span class="field-label">
    Script
    <span class="tip"><span class="tip-icon">ⓘ</span>
      <span class="tip-box">Each row is one speaker turn. Choose the speaker number and type that speaker's text. The model supports up to 4 distinct speakers (0-3) in one generation.<br><br>Use <strong>... , —</strong> for natural pauses (handled by the model). Use the <strong>[pause:Xs]</strong> buttons below for precise timed silences.</span>
    </span>
  </span>
  <div class="speaker-editor">
    {#each turns as turn, i}
      <SpeakerRow
        bind:speakerId={turns[i].speakerId}
        bind:text={turns[i].text}
        canRemove={turns.length > 1}
        onremove={() => removeTurn(i)}
        index={i}
        onfocus={onfocustextarea}
      />
    {/each}
  </div>
  <div style="display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; align-items:center">
    <button type="button" class="secondary-btn" style="font-size:13px; padding:6px 14px" onclick={addTurn}>+ Add Speaker Turn</button>
    <span class="helper-text" style="font-size:11px">Tip: type <code>...</code> or <code>,</code> for natural pauses. Use buttons below for precise gaps.</span>
  </div>
</div>
