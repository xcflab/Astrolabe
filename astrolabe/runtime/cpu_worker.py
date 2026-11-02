"""CPU-only child; visibility is cleared in the parent before interpreter startup."""
import contextlib
import importlib
import json
import sys


def main():
    name, limit = sys.argv[1], int(sys.argv[2])
    data = sys.stdin.buffer.read(limit + 1)
    if len(data) > limit:
        raise ValueError('CPU work item exceeds byte limit')
    module, attr = name.split(':', 1)
    with contextlib.redirect_stdout(sys.stderr):
        fn = getattr(importlib.import_module(module), attr)
        result = fn(json.loads(data))
    encoded = json.dumps(result, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > limit:
        raise ValueError('CPU result exceeds byte limit')
    sys.stdout.buffer.write(encoded)


if __name__ == '__main__':
    main()
