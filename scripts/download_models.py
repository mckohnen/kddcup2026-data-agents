"""Download fastembed model weights to models/fastembed/ for offline use.

Run this once before building the Docker image:
    uv run python scripts/download_models.py

The weights are saved to models/fastembed/ (git-ignored).
The Dockerfile copies this folder into the image so the container
has the weights on disk and never needs internet access at eval time.
"""
from pathlib import Path
from fastembed import TextEmbedding

CACHE_DIR = Path(__file__).parent.parent / "models" / "fastembed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("BAAI/bge-small-en-v1.5", "~130 MB"),
    ("sentence-transformers/all-MiniLM-L6-v2", "~90 MB"),
]

for name, size in MODELS:
    print(f"Downloading {name} ({size}) to {CACHE_DIR}")
    model = TextEmbedding(name, cache_dir=str(CACHE_DIR))
    list(model.embed(["warmup"]))  # triggers actual download
    print(f"  Done: {name}")

print("\nAll models saved to:", CACHE_DIR)
print("Next step: ./scripts/build_submission.sh team1210 <N>")
