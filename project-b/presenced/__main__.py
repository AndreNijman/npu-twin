from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import suppress

from . import __version__
from .config import Config
from .face import build as build_face
from .fsm import PresenceFSM, State
from .hypr import HyprBridge
from .preflight import run as preflight
from .xdna_probe import probe as xdna_probe

log = logging.getLogger("presenced")


async def _loop(cfg: Config, stop: asyncio.Event) -> int:
    detector = build_face(cfg.detector, cfg.camera_index)
    hypr = HyprBridge()
    fsm = PresenceFSM(grace_period_s=cfg.grace_period_s)
    log.info("entered main loop (state=%s, grace=%.1fs)", fsm.state.value, cfg.grace_period_s)
    try:
        while not stop.is_set():
            face = await asyncio.to_thread(detector.detect)
            tr = fsm.observe(face)
            if tr:
                log.info("%s -> %s (%s)", tr.frm.value, tr.to.value, tr.reason)
                if tr.to == State.AWAY:
                    await hypr.dispatch(cfg.away_action)
                elif tr.to == State.PRESENT and tr.frm == State.AWAY:
                    await hypr.dispatch(cfg.present_action)
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.frame_interval_s)
            except asyncio.TimeoutError:
                pass
    finally:
        detector.close()
    return 0


def _install_signals(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="presenced")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--probe-only", action="store_true", help="run XDNA probe and exit")
    parser.add_argument("--dry-run", action="store_true", help="preflight only, do not start loop")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    status = xdna_probe(cfg.xdna_device)
    log.info(
        "xdna probe: device=%s xrt=%s vitis=%s -> %s",
        status.device_present, status.xrt_cli_present, status.vitis_ep_importable, status.verdict,
    )
    if args.probe_only:
        return 0

    if not preflight(cfg.camera_index):
        return 2
    if args.dry_run:
        return 0

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop = asyncio.Event()
    _install_signals(loop, stop)
    try:
        return loop.run_until_complete(_loop(cfg, stop))
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        with suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
