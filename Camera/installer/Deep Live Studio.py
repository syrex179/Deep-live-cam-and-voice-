"""Windowed entry point used by the installed Deep Live Studio launcher."""

import sys
from pathlib import Path

import deep_live_studio

# PyInstaller unpacks the launcher into a temporary folder.  Studio resources
# and its camera runtime instead live beside the installed executable.
install_root = Path(sys.executable).resolve().parent
deep_live_studio.ROOT = install_root
deep_live_studio.SCENES_PATH = install_root / "studio_scenes.json"
deep_live_studio.CAMERA_STATE_PATH = install_root / "switch_states.json"
from studio_shell import run_studio


if __name__ == "__main__":
    raise SystemExit(run_studio())
