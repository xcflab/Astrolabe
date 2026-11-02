"""Test-only instrumentation of the real DummyEvaluator trajectory."""
import os
if os.environ.get('ASTROLABE_TEST_BROKER') and os.environ.get('ASTROLABE_EXECUTION_ENDPOINT'):
    import hashlib
    import json
    import sys
    import time
    from astrolabe.evaluators.dummy import DummyEvaluator
    from astrolabe.runtime.client import DaemonClient

    original = DummyEvaluator.evaluate
    event_log = os.environ['ASTROLABE_TEST_EVENTS']
    identity = {'environment': 'test', 'code': 'test', 'weights': 'none', 'device': 'cpu'}
    def event(kind, **kwargs):
        record = {'kind': kind, 'time': time.monotonic(), 'owner': os.getpid(), **kwargs}
        fd = os.open(event_log, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (json.dumps(record) + '\n').encode())
        finally:
            os.close(fd)

    def evaluate(self, inputs=None, runtime_config=None):
        event('prepare_start')
        digest = b'cpu-stage'
        for _ in range(150000):
            digest = hashlib.sha256(digest).digest()
        time.sleep(.35)
        event('prepare_end')
        with DaemonClient([sys.executable, os.environ['ASTROLABE_TEST_FIXTURE']], identity) as client:
            event('resident', worker=client.hello['pid'], loads=client.hello['load_count'])
            client.call({'value': digest.hex(), 'sleep': .8,
                         'event_log': event_log, 'owner': os.getpid()}, timeout=20)
        return original(self, inputs, runtime_config)
    DummyEvaluator.evaluate = evaluate
