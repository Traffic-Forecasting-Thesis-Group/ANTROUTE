"""Live collection inside the API: one HERE capture every 15 minutes while the backend serves live mode.

At :00, :15, :30 and :45 (Asia/Manila) the collector saves HERE flow and incidents, and weather on the first tick of
each clock hour (scripts/collect_live.collect_tick), then builds that capture's live snapshot so the planner reads it
without waiting. At startup it collects at once when the current slot has no capture yet.

Only one collector runs per machine (a lock file in data/live); a second backend on the same machine serves the saved
captures without collecting. Set ANTROUTE_COLLECT=0 to turn collection off, for example on a machine that should only
read captures copied from the study's collecting PC.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from src.routing.condition_update import MANILA

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data/live/collector.lock"
INTERVAL_MINUTES = 15
logger = logging.getLogger("antroute.live_collector")


def _try_lock(handle) -> bool:
    try:
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except ImportError:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    except OSError:
        return False


class LiveCollector:
    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock_handle = None
        self.state = {"enabled": False, "interval_minutes": INTERVAL_MINUTES, "last_tick": None, "last_result": None,
                      "next_tick": None, "error": None}

    def start(self) -> None:
        if os.getenv("ANTROUTE_COLLECT", "1") != "1":
            self.state["error"] = "Collection is turned off (ANTROUTE_COLLECT=0)"
            return
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = open(LOCK, "a+")
        if not _try_lock(self._lock_handle):
            self._lock_handle.close()
            self._lock_handle = None
            self.state["error"] = "Another process on this machine is already collecting"
            return
        self.state["enabled"] = True
        self._thread = threading.Thread(target=self._run, name="live-collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._lock_handle is not None:
            self._lock_handle.close()

    def _run(self) -> None:
        from scripts.collect_live import floor_tick, next_tick
        from src.live.snapshot import captures
        current = floor_tick(datetime.now(MANILA), INTERVAL_MINUTES)
        if current not in captures():
            self._tick(current)
        while not self._stop.is_set():
            tick = next_tick(datetime.now(MANILA), INTERVAL_MINUTES)
            self.state["next_tick"] = tick.isoformat()
            if self._stop.wait(max(0.0, (tick - datetime.now(MANILA)).total_seconds())):
                return
            self._tick(tick)

    def _tick(self, moment: datetime) -> None:
        from scripts.collect_live import collect_tick
        from src.live.snapshot import LiveDataUnavailable, live_snapshot
        try:
            rows = collect_tick(moment)
            self.state.update(last_tick=moment.isoformat(), error=None,
                              last_result={row["kind"]: row["status"] for row in rows})
        except Exception as error:  # Logged and retried at the next tick; the API keeps serving saved captures.
            self.state.update(last_tick=moment.isoformat(), error=f"{type(error).__name__}: {error}"[:300])
            logger.warning("Live collection failed at %s: %s", moment, self.state["error"])
            return
        try:
            live_snapshot(moment)  # Match the new capture now, so the first planner request does not wait for it.
        except LiveDataUnavailable as error:
            self.state["error"] = f"Capture saved but not usable yet: {error}"


collector = LiveCollector()
