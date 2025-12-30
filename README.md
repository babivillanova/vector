## Arc vectorization (image → SVG quarter arcs)

This repo contains a **repeatable pipeline** that takes input images of circular quarter-arcs and outputs an SVG where each mark becomes a **true circular 90° arc** using SVG `A` commands.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

- **Folder → one SVG per image**:

```bash
python arc_vectorize.py ./images --out ./svgs
```

- **Single file**:

```bash
python arc_vectorize.py input.png --out ./svgs
```

- **Multiple files**:

```bash
python arc_vectorize.py A.png B.png --out ./svgs
```

### Options

- **Default**: snaps each detected arc to a perfect **90°** interval.
- **Pixel-faithful angular span**:

```bash
python arc_vectorize.py input.png --out ./svgs --no-snap
```

- **Include smaller arcs (lower minimum skeleton component size)**:

```bash
python arc_vectorize.py input.png --out ./svgs --min-component-points 10
```

Notes:
- The script accepts one or more inputs; each input can be an image file or a directory of images.
- Output SVGs are named after the input stem (e.g. `A.png` → `A.svg`).




