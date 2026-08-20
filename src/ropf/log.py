"""Run logging: one callable that writes to the screen and to a file.

Every module that reports progress takes a ``log`` argument of type
``Callable[[str], None]`` and calls it with an already-newline-terminated
string.  That is the whole interface.  A plain callable rather than a class
means `print`, a list's `append`, or nothing at all are all valid loggers, which
is what lets `ropf.model` and `ropf.algorithm` be exercised in tests without a
log file appearing beside the test run.

`Log` is the implementation the command line uses: it writes to stdout and to a
file at once, so the transcript on screen and the transcript on disk are the
same bytes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, TextIO

WIDTH = 79


class Log:
    """A logger that is also a ``Callable[[str], None]``."""

    def __init__(self, path: Optional[str] = None, echo: bool = True):
        self.path = path
        self.echo = echo
        self.started = time.time()
        self._handle: Optional[TextIO] = None
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._handle = open(path, "w", buffering=1)

    def __call__(self, message: str) -> None:
        if self.echo:
            sys.stdout.write(message)
            sys.stdout.flush()
        if self._handle is not None:
            self._handle.write(message)

    def section(self, title: str) -> None:
        self("\n" + "=" * WIDTH + "\n")
        self(f"{title}\n")
        self("=" * WIDTH + "\n")

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "Log":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
