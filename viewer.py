"""
Streaming viewer for floor plan vectorization pipeline.

Serves the viewer HTML and provides an SSE endpoint that streams
pipeline events in real time as each layer is generated, preprocessed,
and vectorized.

Usage:
    uvicorn viewer:app --reload --port 8001
"""

import asyncio
import io
import json
import logging
import tempfile
import traceback
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from fresh_vectorize_layered import run_pipeline, _load_env

# Load .env so GEMINI_API_KEY is available
_load_env()

logger = logging.getLogger("viewer")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Floor Plan Vectorizer Viewer", version="1.0.0")

BASE_DIR = Path(__file__).parent

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# Stores the last pipeline's outputs for download
_last_outputs: dict[str, bytes] = {}
_last_filename: str = ""


def sse_encode(event_type: str, data: str) -> bytes:
    """Format a single SSE frame as bytes."""
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def viewer_page():
    """Serve the single-file viewer frontend."""
    html_path = BASE_DIR / "viewer.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/test-sse")
async def test_sse():
    """Minimal SSE test endpoint to verify streaming works in the browser."""
    async def gen():
        for i in range(5):
            yield sse_encode("tick", json.dumps({"count": i}))
            await asyncio.sleep(1)
        yield sse_encode("done", json.dumps({"msg": "finished"}))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/vectorize/stream")
async def vectorize_stream(file: UploadFile = File(...)):
    """Stream pipeline events via SSE as the vectorization runs."""
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Use PNG or JPEG.",
        )

    img_bytes = await file.read()
    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "input.png"
    mime_type = (
        "image/jpeg"
        if content_type in ("image/jpeg", "image/jpg")
        else "image/png"
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event: dict):
        """Called from the pipeline thread — bridges to async queue."""
        logger.info(f"[SSE] emit: {event['type']}")
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_blocking():
        """Runs the pipeline synchronously in a thread."""
        global _last_outputs, _last_filename
        tmp_dir = tempfile.mkdtemp(prefix="viewer_")
        try:
            logger.info(f"[Pipeline] Starting: {filename} ({len(img_bytes)} bytes)")
            outputs = run_pipeline(
                img_bytes=img_bytes,
                filename=filename,
                mime_type=mime_type,
                output_dir=Path(tmp_dir),
                on_event=on_event,
            )
            _last_outputs = outputs
            _last_filename = filename
            logger.info("[Pipeline] Completed successfully")
        except Exception as e:
            logger.error(f"[Pipeline] Error: {e}\n{traceback.format_exc()}")
            on_event({"type": "pipeline_error", "error": str(e)})
        finally:
            on_event({"type": "__done__"})

    async def event_generator():
        logger.info("[SSE] Generator started, launching pipeline thread")
        # Start pipeline in thread pool
        fut = loop.run_in_executor(None, run_blocking)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60)
                except asyncio.TimeoutError:
                    # Send SSE comment as keepalive
                    yield ": keepalive\n\n".encode("utf-8")
                    continue

                if event["type"] == "__done__":
                    logger.info("[SSE] Done signal received")
                    break

                yield sse_encode(event["type"], json.dumps(event))
        except asyncio.CancelledError:
            logger.info("[SSE] Client disconnected")
        except Exception as e:
            logger.error(f"[SSE] Generator error: {e}\n{traceback.format_exc()}")
        finally:
            if not fut.done():
                fut.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download")
async def download_results():
    """Download all pipeline results as a zip file."""
    if not _last_outputs:
        raise HTTPException(status_code=404, detail="No results available yet")

    buf = io.BytesIO()
    stem = Path(_last_filename).stem if _last_filename else "floorplan"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in _last_outputs.items():
            zf.writestr(name, data)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}_results.zip"',
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "viewer"}
