#!/usr/bin/env bash
# Mount Google Drive via rclone FUSE (whole-project workflow).
# Usage: ./scripts/mount_gdrive.sh
# Run after each reboot before working on the project or dvc push/pull.

set -euo pipefail

REMOTE="${RCLONE_REMOTE:-gdrive}"
BUCKET="${RCLONE_BUCKET:-mask2former-mlops}"
MOUNT_POINT="${RCLONE_MOUNT:-$HOME/rclone-gdrive}"

mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  echo "Already mounted: $MOUNT_POINT"
  exit 0
fi

if ! command -v rclone &>/dev/null; then
  echo "Install rclone: sudo apt install rclone"
  exit 1
fi

rclone mount "${REMOTE}:${BUCKET}" "$MOUNT_POINT" \
  --daemon \
  --vfs-cache-mode writes \
  --dir-cache-time 72h \
  --poll-interval 10s

echo "Mounted ${REMOTE}:${BUCKET} → $MOUNT_POINT"
echo "Project path: $MOUNT_POINT/mask2former"
