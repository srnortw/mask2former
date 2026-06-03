#!/usr/bin/env bash
# One-time: copy/sync local project into gdrive:mask2former-mlops/mask2former
# Then work from ~/rclone-gdrive/mask2former instead of ~/Desktop/mask2former.
#
# Usage:
#   ./scripts/mount_gdrive.sh
#   ./scripts/migrate_project_to_drive.sh [/path/to/mask2former]

set -euo pipefail

SRC="${1:-$HOME/Desktop/mask2former}"
MOUNT_POINT="${RCLONE_MOUNT:-$HOME/rclone-gdrive}"
DEST="$MOUNT_POINT/mask2former"

if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  echo "Mount Drive first: ./scripts/mount_gdrive.sh"
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC"
  exit 1
fi

mkdir -p "$DEST"
echo "Syncing $SRC → $DEST (this may take a while)..."
rclone sync "$SRC" "$DEST" \
  --exclude ".venv/**" \
  --exclude ".git/objects/**" \
  --exclude ".dvc/cache/**" \
  --exclude "__pycache__/**" \
  --exclude "checkpoints/*.pth" \
  --exclude "checkpoints/*.onnx" \
  --progress

echo ""
echo "Done. Use the project on Drive:"
echo "  cd $DEST"
echo "  source .venv/bin/activate   # create venv here if needed"
echo ""
echo "Optional: keep Desktop copy as backup, or replace with symlink:"
echo "  mv $SRC ${SRC}.bak"
echo "  ln -s $DEST $SRC"
