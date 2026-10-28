"""
logger.py — Logging setup for the AI Property Video Generator.

File handler: DEBUG and above -> logs/app.log (with run_id and secret scrubbing)
Console handler: WARNING and above (so normal runs stay quiet)
"""

import os
import re
import uuid
import logging
from pathlib import Path

_CURRENT_RUN_ID = uuid.uuid4().hex[:8]


def set_run_id(run_id: str | None = None) -> str:
    """Set or generate a new active run ID for log grouping."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id or uuid.uuid4().hex[:8]
    return _CURRENT_RUN_ID


def get_run_id() -> str:
    """Return the current active run ID."""
    return _CURRENT_RUN_ID


class SecretScrubberFilter(logging.Filter):
    """Filter that attaches run_id and sanitizes any sensitive API keys from logs."""

    # Patterns matching typical LLM / Google / OpenAI secret keys
    SECRET_PATTERNS = [
        re.compile(r"AIza[0-9A-Za-z\-_]{30,40}"),       # Google Gemini / Maps
        re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),          # OpenAI / generic
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = get_run_id()

        # Sanitize known env keys
        known_keys = [
            os.getenv("GEMINI_API_KEY", "").strip(),
            os.getenv("LLM_API_KEY", "").strip(),
        ]

        def _clean_str(val: str) -> str:
            for k in known_keys:
                if k and len(k) >= 8:
                    val = val.replace(k, "[REDACTED_API_KEY]")
            for pat in self.SECRET_PATTERNS:
                val = pat.sub("[REDACTED_API_KEY]", val)
            return val

        if isinstance(record.msg, str):
            record.msg = _clean_str(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                cleaned_dict = {}
                for ak, av in record.args.items():
                    cleaned_dict[ak] = _clean_str(av) if isinstance(av, str) else av
                record.args = cleaned_dict
            elif isinstance(record.args, tuple):
                cleaned_args = [
                    _clean_str(a) if isinstance(a, str) else a
                    for a in record.args
                ]
                record.args = tuple(cleaned_args)

        return True


def setup_logger(name: str = "property_video") -> logging.Logger:
    """Configure and return the application logger.

    Creates logs/ next to this project's root if it doesn't exist.
    Safe to call multiple times (handlers are added only once).
    """
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    scrubber = SecretScrubberFilter()
    logger.addFilter(scrubber)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [run:%(run_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — full detail with run_id
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(scrubber)
    logger.addHandler(fh)

    # Console handler — only warnings and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    ch.addFilter(scrubber)
    logger.addHandler(ch)

    return logger
