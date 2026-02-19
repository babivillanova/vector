"""
FastAPI app for the floor plan vectorizer.

POST /vectorize  — upload PNG/JPEG, receive ZIP with layers + SVG
GET  /health     — health check
"""

import io
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from fresh_vectorize_layered import run_pipeline

app = FastAPI(title="Floor Plan Vectorizer", version="1.0.0")

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    # Validate content type
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Use PNG or JPEG.",
        )

    # Read file
    img_bytes = await file.read()
    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")
    if len(img_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "input.png"
    mime_type = "image/jpeg" if content_type in ("image/jpeg", "image/jpg") else "image/png"

    # Run pipeline in a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            outputs = run_pipeline(
                img_bytes=img_bytes,
                filename=filename,
                mime_type=mime_type,
                output_dir=Path(tmp_dir),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Build ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(outputs.items()):
            zf.writestr(name, data)
    zip_buffer.seek(0)

    stem = Path(filename).stem.replace(" ", "_")
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}_vectorized.zip"'
        },
    )
