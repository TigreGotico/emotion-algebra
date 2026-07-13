"""Python-version compatibility shims.

Internal — not part of the public API.
"""
from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of :class:`enum.StrEnum` for Python < 3.11.

        Plain ``Enum`` formats ``str(member)`` as ``"ClassName.MEMBER"``; the
        real 3.11+ ``enum.StrEnum`` overrides ``__str__`` to return the value.
        Match that, because callers rely on ``str(member)`` being the value —
        ``needs.py`` builds its lookup dict that way, and without this override
        every key on 3.10 comes out as ``"MaxNeefNeed.PROTECTION"`` instead of
        ``"protection"``, silently breaking every lookup.
        """

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
