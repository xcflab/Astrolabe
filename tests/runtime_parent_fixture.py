"""Simulate abrupt session-owner death with an active broker request."""
import os
import pathlib
import subprocess
import sys
import time

from astrolabe.runtime.session import BrokerSession

event_log, outcome, fixture = sys.argv[1:]
session = BrokerSession().__enter__()
child = r"""
import os,sys
from astrolabe.runtime.client import DaemonClient
identity = {'environment':'test','code':'test','weights':'none','device':'cpu'}
try:
    with DaemonClient([sys.executable, sys.argv[3]], identity) as client:
        client.call({'sleep': 15, 'event_log': sys.argv[1], 'owner': os.getpid()}, timeout=30)
except Exception as exc:
    with open(sys.argv[2], 'w') as f:
        f.write(type(exc).__name__)
else:
    with open(sys.argv[2], 'w') as f:
        f.write('unexpected success')
"""
subprocess.Popen([sys.executable, '-c', child, event_log, outcome, fixture],
                 env=session.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    if pathlib.Path(event_log).exists() and 'forward_start' in pathlib.Path(event_log).read_text():
        os._exit(0)
    time.sleep(.02)
os._exit(2)
