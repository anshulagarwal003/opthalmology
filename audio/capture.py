import threading
import queue
import time
import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 5
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION_SEC
WHISPER_MODEL_SIZE = "small"

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_chunk(audio_array: np.ndarray) -> str:
    model = _get_model()
    audio_float = audio_array.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_float, language="en", vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)


class AudioCapture:
    def __init__(self, text_queue: queue.Queue):
        self.text_queue = text_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue = queue.Queue()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        try:
            import sounddevice as sd
        except Exception as e:
            print(f"[audio] sounddevice import error: {e}")
            return

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] status: {status}")
            self._audio_queue.put(indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=1024,
                callback=callback,
            ):
                print("[audio] microphone stream started")
                accumulated = np.array([], dtype=np.int16)

                while not self._stop_event.is_set():
                    # Drain audio queue into accumulator
                    while not self._audio_queue.empty():
                        chunk = self._audio_queue.get_nowait()
                        accumulated = np.concatenate([accumulated, chunk])

                    if len(accumulated) >= CHUNK_SIZE:
                        chunk = accumulated[:CHUNK_SIZE]
                        accumulated = accumulated[CHUNK_SIZE:]
                        try:
                            text = transcribe_chunk(chunk)
                            if text.strip():
                                print(f"[audio] transcribed: {text[:80]}")
                                self.text_queue.put(text.strip())
                        except Exception as e:
                            print(f"[audio] transcription error: {e}")

                    time.sleep(0.1)

        except Exception as e:
            print(f"[audio] stream error: {e}")
