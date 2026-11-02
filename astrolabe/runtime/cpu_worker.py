"""Persistent CPU-only worker, launched with GPU visibility cleared."""
import contextlib
import importlib
import json
import sys


def main():
    limit = int(sys.argv[1])
    from astrolabe.runtime.server import serve
    def invoke(payload):
        encoded = json.dumps(payload['item'], allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) > limit:
            raise ValueError('CPU work item exceeds byte limit')
        module, attr = payload['handler'].split(':', 1)
        with contextlib.redirect_stdout(sys.stderr):
            result = getattr(importlib.import_module(module), attr)(payload['item'])
        if len(json.dumps(result, allow_nan=False, separators=(',', ':')).encode()) > payload['result_limit']:
            raise ValueError('CPU result exceeds byte limit')
        return {'value': result}
    serve(invoke, {'worker': 'cpu', 'protocol': 1}, max_message_bytes=limit + 65536)


if __name__ == '__main__':
    main()
