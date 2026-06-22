# Railway deploy

This app runs as a single FastAPI service.

## Deploy settings

- Build command: leave blank and let Railway detect `requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Python: use the default Railway Python runtime, or pin one with `runtime.txt` if needed

## Environment variables

Set these in Railway if you want the full experience:

- `GEMINI_API_KEY`

## Notes

- `.env` should stay untracked locally.
- The web app uses browser microphone input, so the server does not need `pyaudio`.
