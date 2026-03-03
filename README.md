## Floor Plan Vectorizer

Converts raster floor plan images into layered SVGs (walls, doors, windows, stairs) using Gemini AI for element detection and potrace for vectorization.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the viewer

```bash
uvicorn viewer:app --reload --port 8001
```

Open `http://127.0.0.1:8001` in your browser.

### Inputs

The viewer accepts two image uploads:

1. **Floor Plan Image** (required) — The black-and-white floor plan to vectorize. This is the image that gets processed through the pipeline.

2. **Reference Image** (optional) — A point cloud or other visual reference displayed underneath the generated SVG layers as a background. This image is **not processed** by the pipeline in any way; it is purely a visual overlay for the end user.

### Reference image requirements

The reference image must be prepared carefully so it aligns correctly with the generated SVG:

- **No background.** The reference image should have its background removed (or be on a transparent/white background). A dark or cluttered background will obscure the SVG layers drawn on top.
- **Matching scale.** The reference image must be scaled to match the floor plan input. Both images should represent the same physical area at the same pixel-per-meter ratio so walls, doors, and other elements line up when overlaid.
- **Matching proportions.** The aspect ratio of the reference image should match the floor plan. If the floor plan is 1200x1000 px, the reference should have the same proportions — otherwise the overlay will appear stretched or misaligned.
- **Matching orientation.** Ensure the reference image is rotated/flipped to match the floor plan's orientation (north-up, etc.).

### Pipeline steps

1. **Element detection** — Gemini vision identifies which layers are present (walls, doors, windows, stairs)
2. **Layer generation** — Gemini generates a separate image for each detected layer
3. **Preprocessing** — OpenCV cleans and refines each layer mask
4. **Vectorization** — Potrace converts each mask into SVG paths
5. **Composition** — All layers are composed into a single layered SVG

### Environment

Requires a `GEMINI_API_KEY` in a `.env` file at the project root.
