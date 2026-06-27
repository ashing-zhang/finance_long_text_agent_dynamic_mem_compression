from __future__ import annotations

import sys
from dataclasses import dataclass as _stdlib_dataclass


def dataclass(_cls=None, **kwargs):
    """Backport-friendly dataclass decorator.

    Python < 3.10 does not support the ``slots`` keyword, so we drop it there
    while preserving the call signature used across the project.
    """

    if sys.version_info < (3, 10):
        kwargs.pop("slots", None)

    def wrap(cls):
        return _stdlib_dataclass(cls, **kwargs)

    if _cls is None:
        return wrap
    return wrap(_cls)
