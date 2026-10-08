from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Callable, Dict, Optional

_LOCK = threading.Lock()
_STOP_REQUESTED = False
_FIRST_SIGNAL: Optional[int] = None
_FIRST_SIGNAL_AT: Optional[float] = None
_PREV_HANDLERS: Dict[int, Callable] = {}


def install_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that stop after the current repeat.

    First signal: mark a graceful stop request (do not abort immediately).
    Second signal: restore the previous handler and re-send the signal to self,
    allowing an immediate/forceful exit.
    """

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(signum)
            _PREV_HANDLERS.setdefault(signum, prev)
            signal.signal(signum, _handle_signal)
            # Avoid interrupting syscalls (best-effort) so the current repeat can finish.
            try:
                signal.siginterrupt(signum, False)
            except AttributeError:
                # Not available on some platforms.
                pass
        except Exception:
            # If signal install fails (e.g., not in main thread), do not crash the run.
            logging.getLogger(__name__).debug(
                "Failed to install graceful shutdown handler for signal %s", signum
            )


def stop_requested() -> bool:
    with _LOCK:
        return bool(_STOP_REQUESTED)


def first_signal_info() -> tuple[Optional[int], Optional[float]]:
    with _LOCK:
        return _FIRST_SIGNAL, _FIRST_SIGNAL_AT


def _handle_signal(signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
    logger = logging.getLogger(__name__)
    with _LOCK:
        global _STOP_REQUESTED, _FIRST_SIGNAL, _FIRST_SIGNAL_AT
        if not _STOP_REQUESTED:
            _STOP_REQUESTED = True
            _FIRST_SIGNAL = signum
            _FIRST_SIGNAL_AT = time.time()
            logger.warning(
                "Received signal %s; will exit after the current repeat finishes. "
                "Send the signal again to force exit now.",
                signum,
            )
            return

    # Second signal: force exit now.
    prev = _PREV_HANDLERS.get(signum, signal.SIG_DFL)
    try:
        signal.signal(signum, prev)
    except Exception:
        # If we can't restore, fall back to hard exit.
        os._exit(128 + int(signum))

    try:
        os.kill(os.getpid(), signum)
    except Exception:
        os._exit(128 + int(signum))

