import asyncio
import base64
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import GestureRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sign Language Recognition")

static_dir   = Path(__file__).parent / "static"
# Default: two levels up from this file (web interface/app/ → handle keypoints/) + /alphabet
_project_root = Path(__file__).parent.parent.parent
alphabet_dir = Path(os.environ.get("ALPHABET_DIR", str(_project_root / "alphabet")))

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve alphabet videos if the directory exists
if alphabet_dir.exists():
    app.mount("/alphabet", StaticFiles(directory=str(alphabet_dir)), name="alphabet")

recognizer: GestureRecognizer | None = None


@app.on_event("startup")
async def startup():
    global recognizer
    try:
        recognizer = GestureRecognizer()
        logger.info("Recognizer initialised successfully")
    except Exception as exc:
        logger.error("Failed to initialise recognizer: %s", exc)


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


_RO_ORDER = "AĂÂBCDEFGHIÎJKLMNOPQRSȘTȚUVWXYZ"
_RO_RANK  = {c: i for i, c in enumerate(_RO_ORDER)}


def _letter_sort_key(path):
    name = path.name.upper()
    # Use defined Romanian-alphabet rank; unknown names sort last
    return _RO_RANK.get(name, len(_RO_ORDER))


@app.get("/videos")
async def list_videos():
    """Return sorted list of {letter, url} for every alphabet video."""
    videos = []
    try:
        dirs = sorted(alphabet_dir.iterdir(), key=_letter_sort_key)
    except OSError:
        return {"videos": videos}
    for letter_dir in dirs:
        try:
            if not letter_dir.is_dir():
                continue
            for vf in sorted(letter_dir.iterdir()):
                if vf.suffix.lower() in (".mp4", ".avi", ".webm", ".mov"):
                    videos.append({
                        "letter": letter_dir.name.upper(),
                        "url": f"/alphabet/{letter_dir.name}/{vf.name}",
                    })
                    break  # one video per letter folder
        except OSError:
            continue
    return {"videos": videos}


@app.get("/health")
async def health():
    loaded = recognizer is not None and recognizer.model_loaded
    return {"status": "ok", "model_loaded": loaded}


@app.post("/reset")
async def reset_buffer():
    if recognizer is not None:
        recognizer.reset()
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            message = await ws.receive_text()

            if not recognizer:
                await ws.send_json({"error": "Recognizer not initialised"})
                continue

            # Always process the frame — landmarks work even without a trained model.
            # Decode base64 JPEG sent by browser ("data:image/jpeg;base64,...")
            try:
                raw = message.split(",", 1)[1] if "," in message else message
                img_bytes = base64.b64decode(raw)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    await ws.send_json({"error": "Could not decode frame"})
                    continue

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, recognizer.process_frame, frame)
                await ws.send_json(result)
            except Exception as exc:
                logger.exception("Frame processing error")
                await ws.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
