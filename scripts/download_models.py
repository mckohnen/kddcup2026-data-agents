"""Download fastembed model weights to models/fastembed/ for offline use.

Run this once before building the Docker image:
    uv run python scripts/download_models.py

The weights are saved to models/fastembed/ (git-ignored).
The Dockerfile copies this folder into the image so the container
has the weights on disk and never needs internet access at eval time.
"""
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "models" / "fastembed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading BAAI/bge-small-en-v1.5 weights to", CACHE_DIR)
print("(~130 MB, one-time download)")

from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(CACHE_DIR))
list(model.embed(["warmup"]))  # triggers actual download

print("Done. Weights saved to:", CACHE_DIR)
print("Next step: ./scripts/build_submission.sh team1210 <N>")
