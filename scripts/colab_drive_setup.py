"""
Colab helper: mount gdrive:mask2former-mlops and use mask2former/ as PROJECT_ROOT.
Import or run as script after Colab secrets are loaded (Cell 3).
"""
import os
import subprocess
import time
from pathlib import Path

REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
BUCKET = os.environ.get("RCLONE_BUCKET", "mask2former-mlops")
MOUNT_POINT = os.environ.get("RCLONE_MOUNT", "/content/gdrive-mlops")
PROJECT_NAME = "mask2former"


def _ensure_rclone_config():
    """Load rclone.conf from Colab secret or Drive bootstrap path."""
    conf_dir = Path.home() / ".config" / "rclone"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / "rclone.conf"

    if conf_path.exists() and conf_path.stat().st_size > 0:
        return str(conf_path)

    # Colab secret: paste contents of ~/.config/rclone/rclone.conf from Ubuntu
    try:
        from google.colab import userdata
        if "RCLONE_CONFIG" in userdata.secrets:
            conf_path.write_text(userdata.get("RCLONE_CONFIG"))
            print("rclone.conf loaded from Colab secret RCLONE_CONFIG")
            return str(conf_path)
    except ImportError:
        pass

    # Bootstrap via GNOME Drive mount (one-time upload of rclone.conf)
    bootstrap = Path("/content/drive/MyDrive/mask2former-mlops/.secrets/rclone.conf")
    if bootstrap.exists():
        conf_path.write_text(bootstrap.read_text())
        print(f"rclone.conf loaded from {bootstrap}")
        return str(conf_path)

    raise RuntimeError(
        "No rclone config found. Do ONE of:\n"
        "  1. Colab Secrets → add RCLONE_CONFIG (paste ~/.config/rclone/rclone.conf from Ubuntu)\n"
        "  2. Upload rclone.conf to Drive: mask2former-mlops/.secrets/rclone.conf\n"
        "     (run drive.mount once, then re-run this cell)"
    )


def mount_drive():
    """Mount gdrive bucket; return mount point path."""
    _ensure_rclone_config()
    mount = Path(MOUNT_POINT)
    mount.mkdir(parents=True, exist_ok=True)

    # Already mounted
    if os.path.ismount(str(mount)):
        print(f"Already mounted: {MOUNT_POINT}")
        return MOUNT_POINT

    subprocess.run(
        [
            "rclone", "mount",
            f"{REMOTE}:{BUCKET}",
            str(mount),
            "--vfs-cache-mode", "writes",
            "--daemon",
        ],
        check=True,
    )
    for _ in range(30):
        if os.path.ismount(str(mount)):
            print(f"Mounted {REMOTE}:{BUCKET} → {MOUNT_POINT}")
            return MOUNT_POINT
        time.sleep(0.5)
    raise RuntimeError(f"Mount failed: {MOUNT_POINT}")


def setup_project(github_token: str = "", pull: bool = True) -> str:
    """
    Ensure repo exists on mounted Drive, git pull, set PROJECT_ROOT env var.
    Returns absolute project root path.
    """
    mount_drive()
    root = Path(MOUNT_POINT) / PROJECT_NAME
    root.mkdir(parents=True, exist_ok=True)

    if not (root / ".git").exists():
        if not github_token:
            raise ValueError("GITHUB_TOKEN required to clone into empty Drive folder")
        url = f"https://{github_token}@github.com/srnortw/mask2former.git"
        print(f"Cloning into {root}...")
        subprocess.run(["git", "clone", url, str(root)], check=True)
    elif pull:
        print(f"git pull in {root}...")
        subprocess.run(["git", "-C", str(root), "pull"], check=True)

    project_root = str(root.resolve())
    os.environ["PROJECT_ROOT"] = project_root
    print(f"PROJECT_ROOT={project_root}")
    return project_root


def skip_roboflow_download(raw_train_ann: str) -> bool:
    """True if train annotations already on Drive (skip Roboflow API)."""
    return os.path.isfile(raw_train_ann)
