from __future__ import annotations

import grp
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def in_group(name: str) -> bool:
    try:
        gid = grp.getgrnam(name).gr_gid
    except KeyError:
        return False
    return gid in os.getgroups()


def check_video_group() -> bool:
    ok = in_group("video")
    if not ok:
        log.error(
            "current user is not in the 'video' group — /dev/video* access will fail. "
            "run: sudo usermod -aG video $USER && re-login"
        )
    return ok


def check_camera_node(index: int) -> bool:
    path = Path(f"/dev/video{index}")
    ok = path.exists()
    if not ok:
        log.error("camera device %s missing", path)
    return ok


def run(camera_index: int) -> bool:
    return check_video_group() & check_camera_node(camera_index)
