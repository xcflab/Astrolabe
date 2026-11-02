import hashlib
import os
import time


def cpu_work(payload):
    start = time.monotonic()
    digest = b'astrolabe'
    for _ in range(payload['rounds']):
        digest = hashlib.sha256(digest).digest()
    time.sleep(payload['delay'])
    return {'start': start, 'end': time.monotonic(), 'pid': os.getpid(),
            'device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'digest': digest.hex()}


def daemon_handler(payload):
    if payload.get('sleep'):
        time.sleep(payload['sleep'])
    return {'value': payload.get('value'), 'pid': os.getpid()}


if __name__ == '__main__':
    from astrolabe.runtime.server import serve
    serve(daemon_handler, {'environment': 'test', 'code': 'test', 'weights': 'none', 'device': 'cpu'})
