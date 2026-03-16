<script lang="ts">
  import type { AppState } from '$lib/stores/app.svelte';
  import type { AsrSegment, TranscribeProgressEvent } from '$lib/types';
  import { postTranscribe } from '$lib/api';
  import AsrSegments from './shared/AsrSegments.svelte';
  import { formatSeconds } from '$lib/utils/format';

  let { app }: { app: AppState } = $props();

  let fileInput: HTMLInputElement;
  let isTranscribing = $state(false);
  let transcribeError = $state('');
  let resultText = $state('');
  let segments = $state<AsrSegment[]>([]);
  let hasResult = $state(false);
  let asrTokens = $state(0);
  let asrMaxTokens = $state(0);
  let asrStatus = $state('');
  let tokenStartTime = 0;
  let avgTokenRate = 0;

  // Post-transcription stats
  let audioDuration = $state(0);
  let transcribeTime = $state(0);

  function formatEta(seconds: number): string {
    if (seconds < 60) return `~${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `~${m}m ${s}s`;
  }

  function formatDuration(sec: number): string {
    if (sec < 60) return `${sec.toFixed(1)}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}m ${s.toFixed(1)}s`;
  }

  function handleProgress(event: TranscribeProgressEvent) {
    if (event.type === 'loading') {
      app.isModelLoading = true;
      asrStatus = 'Loading ASR model...';
    } else if (event.type === 'loaded') {
      app.isModelLoading = false;
      app.setModelLoaded('asr', true);
      asrStatus = 'Transcribing...';
      tokenStartTime = performance.now();
    } else if (event.type === 'progress') {
      asrTokens = event.tokens_generated || 0;
      asrMaxTokens = event.max_tokens || 0;
      if (tokenStartTime > 0 && asrTokens > 0) {
        const elapsed = (performance.now() - tokenStartTime) / 1000;
        avgTokenRate = asrTokens / elapsed;
      }
    } else if (event.type === 'complete') {
      if (event.audio_duration) {
        audioDuration = event.audio_duration;
      }
    }
  }

  function getProgressPercent(): number {
    if (asrMaxTokens <= 0) return 0;
    return Math.min(100, Math.round((asrTokens / asrMaxTokens) * 100));
  }

  function getEtaText(): string {
    if (avgTokenRate <= 0 || asrTokens <= 0 || asrMaxTokens <= 0) return '';
    const remaining = asrMaxTokens - asrTokens;
    if (remaining <= 0) return '';
    return formatEta(remaining / avgTokenRate);
  }

  async function transcribe() {
    if (!fileInput?.files?.length) {
      app.appendLog('[ASR] No audio file selected');
      return;
    }

    isTranscribing = true;
    transcribeError = '';
    hasResult = false;
    asrTokens = 0;
    asrMaxTokens = 0;
    asrStatus = 'Starting...';
    tokenStartTime = 0;
    avgTokenRate = 0;
    audioDuration = 0;
    transcribeTime = 0;
    app.appendLog(`[ASR] Transcribing ${fileInput.files[0].name}...`);

    const startTime = performance.now();

    try {
      const formData = new FormData();
      formData.append('audio', fileInput.files[0]);

      const result = await postTranscribe(formData, handleProgress);
      transcribeTime = (performance.now() - startTime) / 1000;
      resultText = result.text || '(no transcription)';
      segments = Array.isArray(result.segments) ? result.segments : [];
      hasResult = true;
      app.setModelLoaded('asr', true);
      app.appendLog('[ASR] Transcription complete');
    } catch (err: any) {
      const msg = err.message || String(err);
      app.appendLog(`[Error] ASR failed: ${msg}`);
      transcribeError = msg;
    }

    app.isModelLoading = false;
    isTranscribing = false;
    app.refreshModelStatus();
  }
</script>

<section class="panel">
  <hr class="section-divider">
  <span class="field-label" style="font-size:17px">Speech Recognition</span>
  <div class="field">
    <span class="field-label" style="font-weight:500">Upload Audio File</span>
    <input type="file" accept="audio/*,video/*" class="file-input" bind:this={fileInput}>
    <span class="helper-text">Supports audio (WAV, MP3, FLAC) and video (MP4, MKV, WebM)</span>
  </div>
  <div class="control-row">
    <button type="button" class="btn-primary" disabled={isTranscribing} onclick={transcribe}>Transcribe</button>
  </div>

  {#if isTranscribing}
    <div class="gen-progress" style="margin-top:8px">
      <span class="gen-status">
        {asrStatus}
        {#if getEtaText()}
          <span class="metric-unit">{getEtaText()}</span>
        {/if}
      </span>
      {#if asrMaxTokens > 0}
        <div class="progress-track">
          <div class="progress-fill" style="width: {getProgressPercent()}%"></div>
        </div>
        <span class="progress-label">Tokens: {asrTokens}/{asrMaxTokens} ({getProgressPercent()}%)</span>
      {/if}
    </div>
  {/if}

  {#if transcribeError}
    <span style="color:#c0392b">Error: {transcribeError}</span>
  {/if}

  {#if hasResult}
    <div style="margin-top:4px">
      {#if audioDuration > 0 && transcribeTime > 0}
        <div class="asr-stats">
          <div class="asr-stat">
            <span class="asr-stat-value">{formatDuration(audioDuration)}</span>
            <span class="asr-stat-label">Audio Length</span>
          </div>
          <div class="asr-stat">
            <span class="asr-stat-value">{formatDuration(transcribeTime)}</span>
            <span class="asr-stat-label">Transcribe Time</span>
          </div>
          <div class="asr-stat">
            <span class="asr-stat-value">{(audioDuration / transcribeTime).toFixed(1)}x</span>
            <span class="asr-stat-label">Realtime Factor</span>
          </div>
        </div>
      {/if}
      <div class="field">
        <span class="field-label">Transcription</span>
        <pre class="asr-result">{resultText}</pre>
      </div>
      <AsrSegments {segments} />
    </div>
  {/if}
</section>
