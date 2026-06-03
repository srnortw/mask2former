# 11 — Whole Project on Google Drive (rclone mount)

## Goal

Keep the **entire project** on Google Drive so Colab restarts do not wipe checkpoints, data, or training state. Edits under the mount path are written to Drive (via rclone FUSE).

```
gdrive:mask2former-mlops/
└── mask2former/          ← git repo, data/, checkpoints/, config.yaml, .env
```

GitHub stays the source of truth for **code**. Drive holds **data + artifacts + local secrets**.

---

## Ubuntu (daily workflow)

### One-time: put project on Drive

```bash
cd ~/Desktop/mask2former
./scripts/mount_gdrive.sh
./scripts/migrate_project_to_drive.sh   # sync Desktop → Drive (excludes .venv, large ckpts)
cd ~/rclone-gdrive/mask2former
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or your usual pip installs
```

Optional symlink so `~/Desktop/mask2former` points at Drive:

```bash
mv ~/Desktop/mask2former ~/Desktop/mask2former.bak
ln -s ~/rclone-gdrive/mask2former ~/Desktop/mask2former
```

### Every session (after reboot)

```bash
./scripts/mount_gdrive.sh
cd ~/rclone-gdrive/mask2former
source .venv/bin/activate
```

DVC remote stays `url = /home/serkanrob/rclone-gdrive` (parent bucket root) — see `.dvc/config`.

### Unmount

```bash
fusermount -u ~/rclone-gdrive
```

---

## Colab (notebook workflow)

### One-time: rclone config in Colab

Copy your Ubuntu config to Drive **once**:

```bash
# On Ubuntu (with mount running)
rclone copy ~/.config/rclone/rclone.conf gdrive:mask2former-mlops/.secrets/rclone.conf
```

**Or** add Colab Secret `RCLONE_CONFIG` = full contents of `~/.config/rclone/rclone.conf`.

### Every Colab session

Run cells in order:

| Cell | What |
|------|------|
| 1–3 | GPU, deps, secrets |
| **4** | rclone mount + `PROJECT_ROOT` on Drive + `git pull` + Roboflow only if missing |
| 5+ | Training / ONNX / registry — all paths use `PROJECT_ROOT` |

`PROJECT_ROOT` is typically `/content/gdrive-mlops/mask2former`.

Cells **11** and **18** (old Drive copy) are **skipped** when using this workflow — checkpoints and ONNX files are already on Drive.

### After restart — test Phase 5 only

```
1 → 2 → 3 → 4 → 20
```

No re-training, no re-quantize if artifacts exist on Drive.

---

## What to keep on Drive vs local

| On Drive (mount) | Keep local only |
|------------------|-----------------|
| `src/`, `docs/`, `notebooks/` | — |
| `data/raw`, `data/processed` | — |
| `checkpoints/*.pth`, `*.onnx` | — |
| `.env` | — |
| `.git/` | — |
| — | `.venv/` (slow on FUSE; recreate per machine) |
| — | `.dvc/cache/` (large; DVC still uses remote on Drive) |

**Training speed:** If epochs are slow on FUSE, train with `data/` on `/content/data` (symlink or copy once per session) but keep `checkpoints/` on the mount.

---

## Copy rclone.conf to Colab (reference)

```bash
# Ubuntu
rclone config file
# → ~/.config/rclone/rclone.conf

rclone copy ~/.config/rclone/rclone.conf gdrive:mask2former-mlops/.secrets/
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Mount failed` | Check `rclone lsd gdrive:` on Ubuntu; re-auth `rclone config reconnect gdrive` |
| Colab: no rclone config | Add `RCLONE_CONFIG` secret or upload `.secrets/rclone.conf` on Drive |
| DVC push slow / fails | Ensure `mount_gdrive.sh` ran; mount still alive: `mountpoint ~/rclone-gdrive` |
| Two copies diverged | `git pull` before editing; one canonical folder: `mask2former/` on Drive |

---

## Summary

| Tool | Role |
|------|------|
| `scripts/mount_gdrive.sh` | Mount bucket on Ubuntu |
| `scripts/migrate_project_to_drive.sh` | One-time sync project to Drive |
| `scripts/colab_drive_setup.py` | Mount + clone/pull in Colab |
| `notebooks/train_colab.ipynb` Cell 4 | Uses drive setup |

**See also:** [01 — Data Management](01_data_management.md) (DVC + rclone remote)
