"""Worker-side JSONL loop. The adapter loads and verifies its own model identity."""
import json
import os
import sys


def serve(handler, identity, *, load_count=1, max_message_bytes=1048576):
    if not isinstance(identity, dict):
        raise TypeError('identity must be an object')
    out = sys.stdout.buffer
    def send(obj):
        data = (json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(data) > max_message_bytes:
            raise ValueError('response exceeds max_message_bytes')
        out.write(data)
        out.flush()
    send(dict(type='hello', protocol=1, identity=identity, pid=os.getpid(), load_count=load_count))
    used = set()
    while True:
        line = sys.stdin.buffer.readline(max_message_bytes + 1)
        if not line:
            return
        if len(line) > max_message_bytes or not line.endswith(b'\n'):
            raise ValueError('request exceeds max_message_bytes')
        req = json.loads(line)
        if req.get('type') == 'shutdown':
            return
        rid, epoch = req.get('request_id'), req.get('epoch')
        if (req.get('type') != 'call' or not isinstance(rid, str) or not rid or
            not isinstance(epoch, str) or not epoch or not isinstance(req.get('payload'), dict) or
            (epoch, rid) in used):
            raise ValueError('invalid or duplicate request')
        used.add((epoch, rid))
        try:
            result = handler(req['payload'])
            if not isinstance(result, dict):
                raise TypeError('handler result must be an object')
            send(dict(type='result', request_id=rid, epoch=epoch, result=result))
        except Exception as exc:
            send(dict(type='error', request_id=rid, epoch=epoch,
                      error=f'{type(exc).__name__}: {exc}'))
