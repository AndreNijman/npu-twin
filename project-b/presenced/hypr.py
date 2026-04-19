from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class HyprBridge:
    binary: str = "hyprctl"

    def available(self) -> bool:
        if shutil.which(self.binary) is None:
            return False
        return bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"))

    async def dispatch(self, command: str) -> int:
        if not self.available():
            log.warning("hyprctl unavailable; would dispatch: %s", command)
            return -1
        argv = [self.binary, "dispatch", *command.split()]
        log.info("hyprctl %s", " ".join(argv[1:]))
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            log.error("hyprctl failed (%d): %s", proc.returncode, err.decode().strip())
        return proc.returncode or 0
