#!/usr/bin/env bash
# Build and package a Docker image for KDD Cup 2026 submission.
#
# Usage:  ./scripts/build_submission.sh <team_id> <version>
# Example: ./scripts/build_submission.sh team0042 3
#
# Produces: <team_id>_v<version>.tar.gz  (ready to upload to Google Drive)
# Subject line template: [KDDCup2026 Data Agents] Submission - <team_id> - v<N>
set -euo pipefail

TEAM_ID="${1:?Usage: $0 <team_id> <version>}"
VERSION="${2:?Usage: $0 <team_id> <version>}"
IMAGE="${TEAM_ID}:v${VERSION}"
ARCHIVE="${TEAM_ID}_v${VERSION}.tar.gz"

# Move to repo root regardless of where the script is called from.
cd "$(dirname "$0")/.."

# Ensure fastembed model weights are present before building the image.
# The Dockerfile COPYs models/fastembed/ into the image; without it the build fails.
if [ ! -d "models/fastembed" ]; then
  echo "ERROR: models/fastembed/ not found."
  echo "Run 'uv run python scripts/download_models.py' first to download the model weights."
  exit 1
fi

echo "==> Building image: ${IMAGE}"
docker build --platform linux/amd64 -t "${IMAGE}" .

echo "==> Saving to archive: ${ARCHIVE}"
docker save "${IMAGE}" | gzip > "${ARCHIVE}"

SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
echo "==> Done: ${ARCHIVE} (${SIZE})"
echo ""
echo "Upload to Google Drive, set sharing to 'Anyone with the link can view', then email:"
echo "  Subject: [KDDCup2026 Data Agents] Submission - ${TEAM_ID} - v${VERSION}"
