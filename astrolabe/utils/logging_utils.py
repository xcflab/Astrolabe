from __future__ import annotations

import copy
import logging
import math
import re
from collections.abc import Mapping
from statistics import mean
from typing import Any


ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "console_level": "INFO",
    "info_file_level": "INFO",
    "debug_file_level": "DEBUG",
    "color": {
        "console": "auto",  # auto | always | never
        "file": "strip",  # strip | ansi
        "dump": "strip",  # strip | ansi
    },
}

_VALID_CONSOLE_COLOR_MODES = {"auto", "always", "never"}
_VALID_FILE_COLOR_MODES = {"strip", "ansi"}


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def colorize(text: str, color: str | None) -> str:
    if not color:
        return text
    if isinstance(color, str) and color.startswith("ansi256:"):
        try:
            code = int(color.split(":", 1)[1])
            if 0 <= code <= 255:
                return f"\033[38;5;{code}m{text}{ANSI_RESET}"
        except Exception:
            return text
    prefix = ANSI_COLORS.get(color)
    if not prefix:
        return text
    return f"{prefix}{text}{ANSI_RESET}"


def colorize_delta(delta: float | None, text: str) -> str:
    if delta is None:
        return text
    if delta > 0:
        return colorize(text, "green")
    if delta < 0:
        return colorize(text, "red")
    return colorize(text, "yellow")


def parse_log_level(value: Any, default: int = logging.INFO) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    name = str(value).strip().upper()
    level = logging.getLevelName(name)
    if isinstance(level, int):
        return level
    return default


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def normalize_logging_config(raw_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(copy.deepcopy(DEFAULT_LOGGING_CONFIG), raw_cfg or {})

    color_cfg = cfg.setdefault("color", {})
    console_mode = str(color_cfg.get("console", "auto")).lower().strip()
    file_mode = str(color_cfg.get("file", "strip")).lower().strip()
    dump_mode = str(color_cfg.get("dump", "strip")).lower().strip()

    if console_mode not in _VALID_CONSOLE_COLOR_MODES:
        raise ValueError(
            "logging.color.console must be one of: auto, always, never"
        )
    if file_mode not in _VALID_FILE_COLOR_MODES:
        raise ValueError("logging.color.file must be one of: strip, ansi")
    if dump_mode not in _VALID_FILE_COLOR_MODES:
        raise ValueError("logging.color.dump must be one of: strip, ansi")

    color_cfg["console"] = console_mode
    color_cfg["file"] = file_mode
    color_cfg["dump"] = dump_mode
    return cfg


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream is not None and stream.isatty())
    except Exception:
        return False


def should_strip_console_ansi(mode: str, stream: Any) -> bool:
    normalized = str(mode or "auto").lower().strip()
    if normalized == "always":
        return False
    if normalized == "never":
        return True
    return not _is_tty(stream)


def should_strip_file_ansi(mode: str) -> bool:
    return str(mode or "strip").lower().strip() != "ansi"


def format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def to_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    return None


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


class AnsiPolicyFormatter(logging.Formatter):
    def __init__(self, *args, strip_ansi_sequences: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.strip_ansi_sequences = bool(strip_ansi_sequences)

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if self.strip_ansi_sequences:
            return strip_ansi(rendered)
        return rendered
