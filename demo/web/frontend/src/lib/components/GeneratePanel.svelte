<script lang="ts">
  import type { AppState } from '$lib/stores/app.svelte';
  import type { SpeakerTurn, GenerateProgressEvent, SimilarityScores } from '$lib/types';
  import { postGenerate } from '$lib/api';
  import SpeakerEditor from './shared/SpeakerEditor.svelte';
  import PauseInsert from './shared/PauseInsert.svelte';
  import VoiceUploadRow from './shared/VoiceUploadRow.svelte';
  import RangeControl from './shared/RangeControl.svelte';
  import SimilarityDisplay from './shared/SimilarityDisplay.svelte';

  let { app }: { app: AppState } = $props();

  let turns = $state<SpeakerTurn[]>([{ speakerId: 0, text: '' }]);
  let cfg = $state(3.0);
  let steps = $state(20);
  let speed = $state(1.0);
  let seed = $state('');
  let maxWords = $state(200);
  let isGenerating = $state(false);
  let genError = $state('');
  let genChunk = $state(0);
  let genTotalChunks = $state(0);
  let genStep = $state(0);
  let genTotalSteps = $state(0);
  let genStatus = $state('');
  let stepStartTime = 0;
  let avgStepTime = 0;
  let audioUrl = $state<string | null>(null);
  let similarityScores = $state<SimilarityScores | null>(null);
  let focusedTextarea: HTMLTextAreaElement | null = null;
  let focusedTurnIndex: number = -1;
  let voiceFiles = $state<Record<string, File | null>>({});

  function getUniqueSpeakerIds(): number[] {
    const ids = new Set(turns.map((t) => t.speakerId));
    return [...ids].sort();
  }

  function assembleSpeakerText(): string {
    return turns
      .filter((t) => t.text.trim())
      .map((t) => `Speaker ${t.speakerId}: ${t.text.trim()}`)
      .join('\n\n');
  }

  function handlePauseInsert(tag: string) {
    if (!focusedTextarea || focusedTurnIndex < 0 || focusedTurnIndex >= turns.length) return;
    const start = focusedTextarea.selectionStart;
    const end = focusedTextarea.selectionEnd;
    const val = turns[focusedTurnIndex].text;
    turns[focusedTurnIndex] = { ...turns[focusedTurnIndex], text: val.slice(0, start) + tag + val.slice(end) };
    const ta = focusedTextarea;
    const newPos = start + tag.length;
    requestAnimationFrame(() => {
      ta.setSelectionRange(newPos, newPos);
      ta.focus();
    });
  }

  function randomizeSeed() {
    seed = String(Math.floor(Math.random() * 2 ** 32));
  }

  function formatEta(seconds: number): string {
    if (seconds < 60) return `~${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `~${m}m ${s}s`;
  }

  function handleProgress(event: GenerateProgressEvent) {
    const now = performance.now();

    if (event.type === 'loading') {
      app.isModelLoading = true;
      genStatus = 'Loading model...';
    } else if (event.type === 'loaded') {
      app.isModelLoading = false;
      app.setModelLoaded(event.model_id || app.currentModel, true);
      genStatus = 'Model loaded, generating...';
    } else if (event.type === 'progress') {
      genChunk = event.chunk || 0;
      genTotalChunks = event.total_chunks || 0;
      // Reset step tracking for the new chunk
      genStep = 0;
      genTotalSteps = 0;
      stepStartTime = now;
      avgStepTime = 0;
    } else if (event.type === 'step') {
      genChunk = event.chunk || genChunk;
      genTotalChunks = event.total_chunks || genTotalChunks;
      genStep = event.step || 0;
      genTotalSteps = event.total_steps || 0;

      if (genStep >= 1 && stepStartTime > 0) {
        avgStepTime = ((now - stepStartTime) / 1000) / genStep;
      }
    } else if (event.type === 'similarity' && event.similarity) {
      similarityScores = event.similarity;
      const modelLabel = app.currentModel === 'large' ? 'Large' : '1.5B';
      app.appendLog(`[${modelLabel}] Voice match: ${(event.similarity.ensemble * 100).toFixed(1)}%`);
    }
  }

  function getOverallProgress(): number {
    if (genTotalChunks <= 0) return 0;
    // Progress = completed chunks + fraction of current chunk's steps
    const chunkFraction = genTotalSteps > 0 ? genStep / genTotalSteps : 0;
    const completedChunks = genChunk - 1 + chunkFraction;
    return Math.min(100, Math.round((completedChunks / genTotalChunks) * 100));
  }

  function getStatusText(): string {
    if (genTotalChunks <= 0) return genStatus;
    const chunkLabel = genTotalChunks > 1 ? `Chunk ${genChunk}/${genTotalChunks} ` : '';
    if (genTotalSteps > 0) {
      return `${chunkLabel}Step ${genStep}/${genTotalSteps}`;
    }
    return `${chunkLabel}Generating...`;
  }

  function getEtaText(): string {
    if (avgStepTime <= 0 || genTotalSteps <= 0) return '';
    // Steps remaining in current chunk + steps for remaining chunks
    const stepsLeft = genTotalSteps - genStep;
    const futureChunks = genTotalChunks - genChunk;
    const totalStepsLeft = stepsLeft + (futureChunks * genTotalSteps);
    if (totalStepsLeft <= 0) return '';
    return formatEta(totalStepsLeft * avgStepTime);
  }

  async function generate() {
    const text = assembleSpeakerText();
    if (!text) {
      app.appendLog('[Generate] Text is empty');
      return;
    }

    const modelLabel = app.currentModel === 'large' ? 'Large' : '1.5B';
    isGenerating = true;
    genError = '';
    genChunk = 0;
    genTotalChunks = 0;
    genStep = 0;
    genTotalSteps = 0;
    genStatus = 'Starting...';
    similarityScores = null;
    stepStartTime = 0;
    avgStepTime = 0;
    app.appendLog(`[${modelLabel}] Sending generate request...`);

    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('cfg_scale', String(cfg));
      formData.append('model_id', app.currentModel);
      formData.append('steps', String(steps));
      formData.append('speed', String(speed));

      const seedVal = seed.trim();
      if (seedVal && parseInt(seedVal, 10) >= 0) {
        formData.append('seed', seedVal);
      }

      formData.append('max_words', String(maxWords));

      let voiceCount = 0;
      for (const [sid, file] of Object.entries(voiceFiles)) {
        if (file) {
          formData.append(`voice_${sid}`, file);
          voiceCount++;
        }
      }
      if (voiceCount > 0) {
        app.appendLog(`[${modelLabel}] Voice cloning: ${voiceCount} speaker reference(s)`);
      }

      const result = await postGenerate(formData, handleProgress);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
      audioUrl = URL.createObjectURL(result.audioBlob);
      // Similarity scores arrive via the 'similarity' SSE event (handled in handleProgress)
      // but also set from the API result as a fallback
      if (result.similarity) {
        similarityScores = result.similarity;
      }
      app.setModelLoaded(app.currentModel, true);
      app.appendLog(`[${modelLabel}] Audio generated — ready to play`);
    } catch (err: any) {
      const msg = err.message || String(err);
      app.appendLog(`[Error] Generate failed: ${msg}`);
      genError = msg;
    }

    app.isModelLoading = false;
    isGenerating = false;
    app.refreshModelStatus();
  }

  function saveAudio() {
    if (!audioUrl) return;
    const link = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    link.href = audioUrl;
    link.download = `vibevoice_${app.currentModel}_${ts}.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    app.appendLog('[Frontend] Audio download triggered');
  }
</script>

<section class="panel">
  <SpeakerEditor bind:turns onfocustextarea={(el, index) => { focusedTextarea = el; focusedTurnIndex = index; }} />
  <PauseInsert oninsert={handlePauseInsert} />
</section>

<section class="panel control-panel">
  <div class="inline-field">
    <span class="field-label">
      Voice References
      <span class="tip"><span class="tip-icon">ⓘ</span>
        <span class="tip-box">Upload 3–30 second WAV or MP3 clips to clone specific voices per speaker. The model matches tone, accent, and speaking style from each reference.<br><br>Speed adjustment is applied to the reference audio before model conditioning — this is pitch-preserving (unlike output resampling). When no reference is provided, speed falls back to output resampling.<br><br>Best results: clean audio, single speaker, no background music.</span>
      </span>
    </span>
    <div style="display:flex; flex-direction:column; gap:6px; margin-top:4px">
      {#each getUniqueSpeakerIds() as sid}
        <VoiceUploadRow speakerId={sid} onfilechange={(file) => { voiceFiles[sid] = file; }} />
      {/each}
    </div>
    <span class="helper-text" style="font-size:11px">Optional — leave blank to use model-default voices. Upload more to enable per-speaker cloning.</span>
  </div>

  <div class="control-row">
    <RangeControl label="CFG" tipText="Classifier-Free Guidance — controls how closely the model follows your text. Higher = more expressive and text-faithful. Lower = more natural variation. For voice cloning, higher values (3–4) tend to preserve the cloned voice more strongly.<br><br>Default: 3.0 &nbsp;|&nbsp; Range: 1.0 – 5.0" min={1.0} max={5.0} step={0.1} bind:value={cfg} formatValue={(v) => v.toFixed(1)} />
    <RangeControl label="Inference Steps" tipText="Diffusion denoising steps for the acoustic head. More steps = slightly cleaner audio, but slower generation. Does not affect speech rate.<br><br>Default: 20 &nbsp;|&nbsp; Range: 5 – 50 &nbsp;|&nbsp; Sweet spot: 15–30" min={5} max={50} step={1} bind:value={steps} formatValue={(v) => String(v)} />
  </div>

  <div class="control-row" style="margin-top:10px">
    <RangeControl label="Speed" tipText="When a voice reference is uploaded, speed is applied to the reference audio before conditioning — this is <strong>pitch-preserving</strong> (linear interpolation, same technique as VibeVoice-ComfyUI).<br><br>Without a voice reference, falls back to output resampling which shifts pitch slightly.<br><br>Use <code>...</code> or <code>,</code> for natural pacing, or <code>[pause:Xs]</code> for timed gaps.<br><br>Default: 1.0 &nbsp;|&nbsp; Range: 0.75 – 1.25" min={0.75} max={1.25} step={0.05} bind:value={speed} formatValue={(v) => v.toFixed(2) + '×'} />
    <button type="button" class="btn-primary" disabled={isGenerating} onclick={generate}>Generate</button>
  </div>

  <div class="control-row" style="margin-top:10px; flex-wrap:wrap; gap:14px 24px">
    <label class="inline-field" style="flex-direction:row; align-items:center; gap:8px">
      <span class="tip" style="font-size:13px; font-weight:500">Seed <span class="tip-icon">ⓘ</span>
        <span class="tip-box">RNG seed for reproducible generation. Same seed + same inputs = same audio. Set to –1 or leave blank for random output each run.<br><br>Default: random &nbsp;|&nbsp; ComfyUI default: 42</span>
      </span>
      <input type="number" min="0" max="4294967295" placeholder="random" bind:value={seed}
        style="width:110px; border:1px solid rgba(31,39,66,0.18); border-radius:8px; padding:5px 8px; font-size:13px">
      <button type="button" class="secondary-btn" style="padding:5px 10px; font-size:12px" title="Randomize seed" onclick={randomizeSeed}>&#10227;</button>
    </label>
    <label class="inline-field" style="flex-direction:row; align-items:center; gap:8px">
      <span class="tip" style="font-size:13px; font-weight:500">Max words/chunk <span class="tip-icon">ⓘ</span>
        <span class="tip-box">Long texts are automatically split at sentence boundaries to prevent instability. Each chunk is generated separately and concatenated. Lower = more stable but more generations per request.<br><br>Default: 200 &nbsp;|&nbsp; Range: 50 – 500</span>
      </span>
      <input type="number" min="50" max="500" step="10" bind:value={maxWords}
        style="width:72px; border:1px solid rgba(31,39,66,0.18); border-radius:8px; padding:5px 8px; font-size:13px">
    </label>
  </div>

  {#if isGenerating}
    <div class="gen-progress" style="margin-top:8px">
      <span class="gen-status">
        {getStatusText()}
        {#if getEtaText()}
          <span class="metric-unit">{getEtaText()}</span>
        {/if}
      </span>
      {#if genTotalChunks > 0}
        <div class="progress-track">
          <div class="progress-fill" style="width: {getOverallProgress()}%"></div>
        </div>
        <span class="progress-label">{getOverallProgress()}% complete</span>
      {/if}
    </div>
  {/if}

  {#if genError}
    <div style="margin-top:6px">
      <span style="color:#c0392b">Error: {genError}</span>
    </div>
  {/if}

  {#if similarityScores}
    <SimilarityDisplay scores={similarityScores} />
  {/if}

  {#if audioUrl}
    <div class="audio-container" style="margin-top:12px">
      <audio controls src={audioUrl}></audio>
      <button type="button" class="secondary-btn" onclick={saveAudio}>Save WAV</button>
    </div>
  {/if}
</section>
