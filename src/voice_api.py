"""Voice interface for the lecture RAG assistant.

Runs a small FastAPI server on its own port (default 8765). A browser page at
`/` records microphone audio, POSTs it to `/voice/ask`, and plays back the
synthesized answer. The audio flows: STT -> RAG -> TTS.
"""
import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.voice_pipeline import VoicePipeline, VoicePipelineError

app = FastAPI(
    title="Lecture RAG Voice Interface",
    description="Browser-based voice interface for the local RAG pipeline.",
    version="1.0.0",
)

pipeline = VoicePipeline()


@app.get("/health")
async def health_check():
    return {"status": "ok", "interface": "voice"}


@app.post("/voice/ask")
async def voice_ask(file: UploadFile = File(...)):
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="No audio received.")

    try:
        result = pipeline.process_audio(audio, filename=file.filename or "audio.wav")
    except VoicePipelineError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "transcript": result["transcript"],
        "answer": result["answer"],
        "sources": result["sources"],
        "is_relevant": result["is_relevant"],
        "vector_distance": result["vector_distance"],
        "audio_base64": base64.b64encode(result["audio"]).decode("ascii"),
        "audio_content_type": "audio/mpeg",
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=INDEX_HTML)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lecture RAG Voice Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
    button { font-size: 1.1rem; padding: 0.8rem 1.4rem; cursor: pointer; border: none; border-radius: 8px; }
    #record { background: #2563eb; color: #fff; }
    #record.recording { background: #dc2626; }
    .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
    label { font-weight: 600; }
    .muted { color: #64748b; }
    #status { margin-top: 0.75rem; }
  </style>
</head>
<body>
  <h1>&#127908; Lecture RAG Voice Assistant</h1>
  <p class="muted">Press the button, ask a question about your lecture slides, then release to get a spoken answer.</p>

  <button id="record">Hold to talk</button>
  <p id="status" class="muted">Idle</p>

  <div class="card" id="result" hidden>
    <div><label>You said:</label> <span id="transcript"></span></div>
    <div><label>Answer:</label> <span id="answer"></span></div>
    <div id="sources"></div>
  </div>

  <script>
    const recordBtn = document.getElementById('record');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    const transcriptEl = document.getElementById('transcript');
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');

    // n8n orchestrates STT -> RAG -> TTS and logs each exchange per session.
    // The voice UI posts to the n8n webhook instead of the local /voice/ask
    // endpoint so n8n sits in the middle of the flow.
    const N8N_WEBHOOK_URL = 'http://' + window.location.hostname + ':5678/webhook/voice-rag';

    function getSessionId() {
      let id = localStorage.getItem('voice_session_id');
      if (!id) {
        id = (crypto.randomUUID && crypto.randomUUID()) || ('session-' + Date.now());
        localStorage.setItem('voice_session_id', id);
      }
      return id;
    }

    let audioContext = null;
    let processor = null;
    let stream = null;
    let recording = false;
    let samples = [];

    function encodeWAV(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      const writeString = (offset, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
      };
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, samples.length * 2, true);
      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
      return new Blob([view], { type: 'audio/wav' });
    }

    async function ensureAudio() {
      if (audioContext) return;
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        if (recording) {
          const input = e.inputBuffer.getChannelData(0);
          samples.push(...new Float32Array(input));
        }
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
    }

    async function start() {
      await ensureAudio();
      samples = [];
      recording = true;
      recordBtn.classList.add('recording');
      recordBtn.textContent = 'Release to send';
      statusEl.textContent = 'Listening...';
    }

    async function stop() {
      recording = false;
      recordBtn.classList.remove('recording');
      recordBtn.textContent = 'Hold to talk';
      statusEl.textContent = 'Transcribing and answering...';

      if (samples.length === 0) {
        statusEl.textContent = 'No audio captured.';
        return;
      }

      const wav = encodeWAV(samples, audioContext.sampleRate);
      const form = new FormData();
      form.append('file', wav, 'recording.wav');
      form.append('session_id', getSessionId());

      try {
        const res = await fetch(N8N_WEBHOOK_URL, { method: 'POST', body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || ('HTTP ' + res.status));
        }
        const data = await res.json();
        transcriptEl.textContent = data.transcript;
        answerEl.textContent = data.answer;
        if (data.sources && data.sources.length) {
          sourcesEl.innerHTML = '<label>Sources:</label> ' +
            data.sources.map(s => `${s.file} (slide ${s.slide})`).join(', ');
        } else {
          sourcesEl.innerHTML = '';
        }
        resultEl.hidden = false;
        statusEl.textContent = 'Playing answer...';

        const audio = new Audio(`data:${data.audio_content_type};base64,${data.audio_base64}`);
        audio.onended = () => { statusEl.textContent = 'Done.'; };
        audio.onerror = () => { statusEl.textContent = 'Answer generated, but audio playback failed.'; };
        await audio.play();
      } catch (err) {
        statusEl.textContent = 'Error: ' + err.message;
      }
    }

    recordBtn.addEventListener('pointerdown', (e) => { e.preventDefault(); start(); });
    recordBtn.addEventListener('pointerup', stop);
    recordBtn.addEventListener('pointerleave', () => { if (recording) stop(); });
  </script>
</body>
</html>
"""
