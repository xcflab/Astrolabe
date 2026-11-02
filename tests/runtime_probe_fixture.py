import json
import os
import time

log = os.environ['RUNTIME_PROBE_LOG']
def event(kind):
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps({'kind': kind, 'time': time.monotonic()}) + '\n').encode())
    finally:
        os.close(fd)

event('probe_start')
time.sleep(.2)
event('probe_end')
print(json.dumps({'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}))
