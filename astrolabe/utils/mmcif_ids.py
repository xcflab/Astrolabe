from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=256)
def int_id_to_str_id(num: int) -> str:
    """Encode positive int as mmCIF-style chain id (reverse spreadsheet naming).

    Examples: 1->A, 2->B, ..., 26->Z, 27->AA, 28->BA, ...
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num -= 1  # 1-based indexing.
    out = []
    while num >= 0:
        out.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(out)


@lru_cache(maxsize=256)
def str_id_to_int_id(str_id: str) -> int:
    if not str_id or any(not ("A" <= c <= "Z") for c in str_id):
        raise ValueError(f"String ID must be upper case letters, got {str_id!r}.")
    offset = ord("A") - 1
    out = 0
    for i, c in enumerate(str_id):
        out += (ord(c) - offset) * int(26**i)
    return out

