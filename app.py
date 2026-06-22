import asyncio
import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from db.lookup import init_db, search_by_keywords
from llm.extractor import extract_clinical_concepts
from soap.generator import generate_soap_note

# ── Shared state (single session for now) ─────────────────────────────────────
class SessionState:
    def __init__(self):
        self.recording = False
        self.transcript: list[str] = []
        self.suggestions: list[dict] = []
        self.accepted_codes: list[dict] = []
        self.text_queue: queue.Queue = queue.Queue()
        self.suggestion_queue: queue.Queue = queue.Queue()
        self._audio_capture = None
        self._stop_event: threading.Event | None = None

    def reset(self):
        self.transcript.clear()
        self.suggestions.clear()
        self.accepted_codes.clear()
        self.text_queue = queue.Queue()
        self.suggestion_queue = queue.Queue()

SESSION = SessionState()


# ── Background: LLM extraction thread ─────────────────────────────────────────
def _extraction_loop(text_queue: queue.Queue, suggestion_queue: queue.Queue, stop_event: threading.Event):
    buffer = []
    last_run = time.time()

    while not stop_event.is_set():
        try:
            text = text_queue.get(timeout=1)
            buffer.append(text)
        except queue.Empty:
            pass

        now = time.time()
        if buffer and (now - last_run >= 10 or len(buffer) >= 3):
            segment = " ".join(buffer)
            buffer = []
            last_run = now

            result = extract_clinical_concepts(segment)
            terms = result.get("clinical_terms", [])
            if terms:
                existing = {s["icd_code"] for s in SESSION.suggestions}
                matches = search_by_keywords(terms, limit=5)
                for m in matches:
                    if m["icd_code"] not in existing:
                        SESSION.suggestions.append(m)
                        existing.add(m["icd_code"])
                        suggestion_queue.put(m)


# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    # Reset session on every page load
    if SESSION.recording:
        if SESSION._audio_capture:
            SESSION._audio_capture.stop()
        if SESSION._stop_event:
            SESSION._stop_event.set()
    SESSION.recording = False
    SESSION.reset()
    return (STATIC_DIR / "index.html").read_text()


@app.post("/api/start")
async def start_session():
    if SESSION.recording:
        return {"status": "already_recording"}

    SESSION.reset()
    SESSION.recording = True

    stop_event = threading.Event()
    SESSION._stop_event = stop_event

    # Audio comes from browser via WebSocket — just start the extraction thread
    t = threading.Thread(
        target=_extraction_loop,
        args=(SESSION.text_queue, SESSION.suggestion_queue, stop_event),
        daemon=True,
    )
    t.start()

    return {"status": "started"}


@app.post("/api/stop")
async def stop_session():
    if not SESSION.recording:
        return {"status": "not_recording"}

    SESSION.recording = False
    if SESSION._stop_event:
        SESSION._stop_event.set()

    return {"status": "stopped"}


@app.post("/api/accept_code")
async def accept_code(payload: dict):
    icd_code = payload.get("icd_code")
    selected_cpts = payload.get("selected_cpts", None)
    match = next((s for s in SESSION.suggestions if s["icd_code"] == icd_code), None)
    if match and not any(a["icd_code"] == icd_code for a in SESSION.accepted_codes):
        entry = dict(match)
        if selected_cpts is not None:
            entry["cpt_codes"] = selected_cpts
        SESSION.accepted_codes.append(entry)
    return {"accepted": SESSION.accepted_codes}


@app.post("/api/remove_code")
async def remove_code(payload: dict):
    icd_code = payload.get("icd_code")
    SESSION.accepted_codes = [a for a in SESSION.accepted_codes if a["icd_code"] != icd_code]
    return {"accepted": SESSION.accepted_codes}


@app.post("/api/soap")
async def soap():
    full_tx = " ".join(SESSION.transcript)
    note = generate_soap_note(full_tx, SESSION.accepted_codes)
    return {"soap_note": note}


@app.post("/api/analyze_text")
async def analyze_text(payload: dict):
    """Analyze a pasted transcript and return ICD/CPT suggestions immediately."""
    text = payload.get("text", "").strip()
    if not text:
        return {"suggestions": []}

    SESSION.transcript.append(text)

    result = extract_clinical_concepts(text)
    terms = result.get("clinical_terms", [])
    new_suggestions = []
    if terms:
        existing = {s["icd_code"] for s in SESSION.suggestions}
        matches = search_by_keywords(terms, limit=8)
        for m in matches:
            if m["icd_code"] not in existing:
                SESSION.suggestions.append(m)
                existing.add(m["icd_code"])
                new_suggestions.append(m)

    return {"suggestions": new_suggestions, "all_suggestions": SESSION.suggestions}


@app.post("/api/clear")
async def clear():
    if SESSION.recording:
        SESSION.recording = False
        if SESSION._audio_capture:
            SESSION._audio_capture.stop()
        if SESSION._stop_event:
            SESSION._stop_event.set()
    SESSION.reset()
    return {"status": "cleared"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def _push_loop():
        """Push transcript + suggestions to browser every second."""
        while True:
            new_texts = []
            while not SESSION.text_queue.empty():
                try:
                    t = SESSION.text_queue.get_nowait()
                    SESSION.transcript.append(t)
                    new_texts.append(t)
                except queue.Empty:
                    break

            new_suggestions = []
            while not SESSION.suggestion_queue.empty():
                try:
                    s = SESSION.suggestion_queue.get_nowait()
                    new_suggestions.append(s)
                except queue.Empty:
                    break

            if new_texts or new_suggestions:
                await websocket.send_text(json.dumps({
                    "type": "update",
                    "transcript_segments": new_texts,
                    "suggestions": new_suggestions,
                    "recording": SESSION.recording,
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "recording": SESSION.recording,
                }))
            await asyncio.sleep(1)

    push_task = asyncio.create_task(_push_loop())

    try:
        while True:
            # Receive either binary audio chunks or text control messages
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Browser sent a raw PCM audio chunk — transcribe in thread pool
                audio_bytes = message["bytes"]
                if SESSION.recording:
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(None, _transcribe_bytes, audio_bytes)
                    if text and text.strip():
                        SESSION.text_queue.put(text.strip())

            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        push_task.cancel()


def _transcribe_bytes(audio_bytes: bytes) -> str:
    """Transcribe raw Float32 PCM bytes from browser Web Audio API."""
    try:
        import numpy as np
        from faster_whisper import WhisperModel

        global _whisper_model
        if _whisper_model is None:
            _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

        # Browser sends Float32 PCM — convert directly
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        segments, _ = _whisper_model.transcribe(audio_array, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        print(f"[transcribe] error: {e}")
        return ""

_whisper_model = None
