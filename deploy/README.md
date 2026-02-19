# Floor Plan Vectorizer API

API service that takes a floor plan image (PNG/JPEG) and returns a ZIP containing vectorized SVG layers.

```
POST /vectorize (multipart/form-data image)
  -> API Gateway / Lambda Function URL
    -> AWS Lambda (Docker container)
      -> FastAPI + Mangum
        -> Gemini API (layer generation)
        -> OpenCV + potrace (clean + vectorize)
  <- ZIP file (layer PNGs + layered SVG)
```

## What's in this folder

| File | Purpose |
|---|---|
| `fresh_vectorize_layered.py` | Core pipeline — Gemini layer extraction + OpenCV cleanup + potrace vectorization |
| `api.py` | FastAPI app — `POST /vectorize` and `GET /health` endpoints |
| `handler.py` | AWS Lambda entry point (Mangum ASGI adapter) |
| `Dockerfile` | Amazon Linux + potrace compiled from source + Python deps |
| `requirements.txt` | Python dependencies |
| `deploy.sh` | One-command deploy script (ECR + Lambda + Function URL) |

## Prerequisites

- **AWS CLI** configured with credentials (`aws configure`)
- **Docker** installed and running
- **Gemini API key** from Google AI Studio (https://aistudio.google.com/apikey)

## Deploy

```bash
export GEMINI_API_KEY=your-gemini-api-key-here
bash deploy.sh
```

The script will:
1. Create an ECR repository (`vectorizer-api`)
2. Build the Docker image (linux/amd64)
3. Push to ECR
4. Create a Lambda function (1GB RAM, 5min timeout)
5. Create a public Function URL and print it

Default region is `us-east-1`. Override with `AWS_REGION=eu-west-1 bash deploy.sh`.

## Test locally with Docker

```bash
docker build --platform linux/amd64 -t vectorizer .
docker run -p 9000:8080 -e GEMINI_API_KEY=$GEMINI_API_KEY vectorizer
```

Then in another terminal:

```bash
# Health check
curl http://localhost:9000/health

# Vectorize an image
curl -F "file=@floorplan.jpeg" http://localhost:9000/vectorize -o result.zip
unzip -l result.zip
```

## Test deployed version

After `deploy.sh` prints the Function URL:

```bash
curl -F "file=@floorplan.jpeg" https://xxxxx.lambda-url.us-east-1.on.aws/vectorize -o result.zip
unzip -l result.zip
```

## ZIP output contents

| File | Description |
|---|---|
| `gemini_walls.png` | Raw Gemini-generated walls layer |
| `gemini_doors.png` | Raw Gemini-generated doors layer |
| `gemini_windows.png` | Raw Gemini-generated windows layer |
| `layer_walls.png` | Cleaned binary mask (after OpenCV processing) |
| `layer_doors.png` | Cleaned binary mask |
| `layer_windows.png` | Cleaned binary mask |
| `*_layered.svg` | Final layered SVG with color-coded groups |
| `layer_debug_overlay.png` | Debug overlay showing all layers |

Only layers detected in the input image are included.

## Lambda configuration

| Setting | Value |
|---|---|
| Runtime | Container image (Python 3.12) |
| Memory | 1024 MB |
| Timeout | 300s (5 min) |
| Ephemeral storage | 1024 MB |
| Auth | None (public Function URL) |

To add auth, change `--auth-type NONE` to `--auth-type AWS_IAM` in `deploy.sh`.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (set by deploy.sh) |
| `AWS_REGION` | No | AWS region (default: us-east-1) |
