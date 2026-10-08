from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from typing import Optional

import pytest


def _readline_with_timeout(stream, timeout_s: float) -> Optional[str]:
    if stream is None:
        return None

    try:
        import selectors
    except Exception:  # pragma: no cover
        return stream.readline()

    sel = selectors.DefaultSelector()
    try:
        sel.register(stream, selectors.EVENT_READ)
        events = sel.select(timeout_s)
        if not events:
            return None
        return stream.readline()
    finally:
        try:
            sel.unregister(stream)
        except Exception:
            pass
        sel.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals required")
def test_sigterm_exits_after_current_repeat() -> None:
    program = textwrap.dedent(
        """
        import time
        from astrolabe.graceful_shutdown import install_handlers, stop_requested

        install_handlers()

        repeats = 3
        for i in range(repeats):
            print(f"START {i}", flush=True)
            # Sleep in small increments to be resilient to EINTR on some platforms.
            end = time.time() + 1.0
            while time.time() < end:
                time.sleep(0.05)
            print(f"END {i}", flush=True)
            if stop_requested():
                print("STOPPING", flush=True)
                break
        print("DONE", flush=True)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None

    lines: list[str] = []
    start = time.time()
    saw_start0 = False

    while time.time() - start < 10:
        line = _readline_with_timeout(proc.stdout, 1.0)
        if line is None:
            continue
        lines.append(line.rstrip("\n"))
        if line.startswith("START 0"):
            saw_start0 = True
            break

    assert saw_start0, f"Did not observe START 0. Output so far: {lines}"

    proc.send_signal(signal.SIGTERM)

    out, err = proc.communicate(timeout=30)
    if out:
        lines.extend([l for l in out.splitlines() if l])

    assert proc.returncode == 0, f"returncode={proc.returncode}\nstdout={lines}\nstderr={err}"
    assert "END 0" in lines
    assert "START 1" not in lines
    assert "STOPPING" in lines


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals required")
def test_second_sigterm_forces_exit() -> None:
    program = textwrap.dedent(
        """
        import time
        from astrolabe.graceful_shutdown import install_handlers

        install_handlers()

        print("START 0", flush=True)
        end = time.time() + 30.0
        while time.time() < end:
            time.sleep(0.05)
        print("END 0", flush=True)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None

    line = _readline_with_timeout(proc.stdout, 5.0)
    assert line is not None and line.startswith("START 0")

    proc.send_signal(signal.SIGTERM)
    time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)

    out, err = proc.communicate(timeout=10)
    # On POSIX, negative returncode means terminated by signal.
    assert proc.returncode in {-signal.SIGTERM, 128 + signal.SIGTERM}, (
        f"returncode={proc.returncode}\nstdout={out}\nstderr={err}"
    )

