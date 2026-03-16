<script lang="ts">
  import type { AppState } from '$lib/stores/app.svelte';
  import type { AudioState } from '$lib/stores/audio.svelte';
  import RangeControl from './shared/RangeControl.svelte';
  import ToggleSwitch from './shared/ToggleSwitch.svelte';
  import StreamingPreview from './shared/StreamingPreview.svelte';
  import TranscriptDisplay from './shared/TranscriptDisplay.svelte';
  import { formatSeconds } from '$lib/utils/format';

  let { app, audio }: { app: AppState; audio: AudioState } = $props();

  let promptText = $state('');
  let selectedVoice = $state('');
  let cfg = $state(1.5);
  let steps = $state(5);
  let doSample = $state(false);
  let temperature = $state(0.9);
  let topP = $state(0.9);

  $effect(() => {
    if (app.defaultVoice && !selectedVoice) {
      selectedVoice = app.defaultVoice;
    }
  });

  function handleStartStop() {
    if (audio.isPlaying) {
      audio.stop();
    } else {
      if (!promptText.trim()) {
        app.appendLog('[Frontend] Text is empty');
        return;
      }
      audio.start({
        text: promptText,
        cfg,
        steps,
        voice: selectedVoice,
        doSample,
        temperature,
        topP
      });
    }
  }

  function resetControls() {
    cfg = 1.5;
    steps = 5;
    doSample = false;
    temperature = 0.9;
    topP = 0.9;
    app.appendLog('[Frontend] Controls reset to defaults');
  }
</script>

<section class="panel">
  <label class="field">
    <span class="field-label">Text</span>
    <textarea
      class="text-input"
      rows="4"
      placeholder="Enter your text here and click Start to hear the streaming TTS output..."
      bind:value={promptText}
    ></textarea>
  </label>

  <StreamingPreview text={promptText} active={audio.isPlaying} />
</section>

<span class="helper-text">This demo requires the full text to be provided upfront. The model then receives the text via streaming input during synthesis.<br>
  For non-punctuation special characters, applying text normalization before processing often yields better results.</span>

<section class="panel control-panel">
  <div class="inline-field">
    <span class="field-label">Speaker</span>
    <select class="select-control" bind:value={selectedVoice} disabled={app.voicesLoading || app.voices.length === 0}>
      {#if app.voices.length === 0}
        <option value="">{app.voicesLoading ? 'Loading...' : 'No voices available'}</option>
      {:else}
        {#each app.voices as voice}
          <option value={voice}>{voice}</option>
        {/each}
      {/if}
    </select>
  </div>

  <div class="control-row">
    <RangeControl label="CFG" tipText="Classifier-Free Guidance — controls how closely the model follows your text. Higher = more faithful but can sound over-enunciated. Lower = more natural but may drift.<br><br>Default: 1.5 &nbsp;|&nbsp; Range: 1.3 – 3.0" min={1.3} max={3} step={0.05} bind:value={cfg} formatValue={(v) => v.toFixed(2)} />
    <RangeControl label="Inference Steps" tipText="Diffusion denoising steps for the acoustic head. More steps = slightly cleaner audio, but slower. Does not affect how fast the speaker talks — only audio texture/quality.<br><br>Default: 5 &nbsp;|&nbsp; Range: 5 – 20 &nbsp;|&nbsp; Sweet spot: 10–15<br><br>Higher steps automatically increase the pre-play buffer to prevent jittery playback." min={5} max={20} step={1} bind:value={steps} formatValue={(v) => String(v)} />
    <button type="button" class="secondary-btn" onclick={resetControls}>Reset Controls</button>
  </div>

  <div class="control-row">
    <ToggleSwitch label="Sampling" tipText="When off, generation is deterministic (same output every run). When on, the model samples from probability distributions, adding natural variation — useful if the output sounds too robotic.<br><br>Default: off" bind:checked={doSample} />
  </div>

  {#if doSample}
    <div class="control-row">
      <RangeControl label="Temperature" tipText="Controls randomness when Sampling is on. Higher = more varied and expressive but less predictable. Lower = more consistent but flatter.<br><br>Default: 0.9 &nbsp;|&nbsp; Range: 0.1 – 2.0" min={0.1} max={2.0} step={0.05} bind:value={temperature} formatValue={(v) => v.toFixed(2)} />
      <RangeControl label="Top-P" tipText="Nucleus sampling threshold. Only tokens whose cumulative probability reaches this value are considered. Lower = more focused/conservative. Higher = broader vocabulary of choices.<br><br>Default: 0.9 &nbsp;|&nbsp; Range: 0.1 – 1.0" min={0.1} max={1.0} step={0.05} bind:value={topP} formatValue={(v) => v.toFixed(2)} />
    </div>
  {/if}

  <div class="control-row">
    <button class="btn-primary" class:playing={audio.isPlaying} onclick={handleStartStop}>
      {audio.isPlaying ? 'Stop' : 'Start'}
    </button>
    <button type="button" class="secondary-btn" disabled={!audio.canSave} onclick={() => audio.saveRecording()}>Save Audio</button>
  </div>
</section>

<section class="panel">
  <div class="metrics">
    <span>Model Generated<strong>{formatSeconds(audio.modelGenerated)}</strong><span class="metric-unit">s</span></span>
    <span>Playback<strong>{formatSeconds(audio.playbackElapsed)}</strong><span class="metric-unit">s</span></span>
    <span>RTF<strong>{audio.rtf}</strong><span class="metric-unit">x</span></span>
  </div>
</section>

<TranscriptDisplay wordTimings={audio.wordTimings} />
