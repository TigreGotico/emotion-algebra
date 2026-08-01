"""Versioned JSON serialization for every emotion type.

Every type in the library already knows how to render itself as a dict.  This
module wraps that in a **versioned envelope** so persisted data stays readable
across releases:

.. code-block:: json

    {"schema": 1, "type": "emotion", "name": "joy"}

Reading is dispatched on ``type``, so :func:`from_dict` round-trips whatever
:func:`to_dict` produced without the caller having to know which class it was.
An unknown ``schema`` version raises rather than silently mis-parsing — the one
thing a persistence layer must never do.

Examples
--------
>>> from emotion_algebra.emotions import get_emotion
>>> blob = to_json(get_emotion("joy"))
>>> from_json(blob).name
'joy'
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase

#: Current envelope version.  Bump only on a **breaking** payload change, and
#: teach :func:`from_dict` to read every version it claims to support.
SCHEMA_VERSION = 1

#: Envelope versions this build can read.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

#: Key holding the envelope version.
SCHEMA_KEY = "schema"


class UnsupportedSchemaError(ValueError):
    """Raised when a payload's ``schema`` version is not one we can read."""


def _loaders() -> dict:
    """Map the ``type`` discriminator to the class that reads it.

    Built lazily: these modules import each other, so a module-level table would
    be a circular import.
    """
    from emotion_algebra.composite_emotions import CompositeEmotion
    from emotion_algebra.feelings import Feeling
    from emotion_algebra.float_emotion import FloatEmotion
    from emotion_algebra.lovheim import LovheimPoint
    from emotion_algebra.plutchik import Emotion, Neutrality
    from emotion_algebra.state import EmotionalState, EmotionTimeline

    return {
        "emotion": Emotion,
        "neutrality": Neutrality,
        "composite": CompositeEmotion,
        "feeling": Feeling,
        "float_emotion": FloatEmotion,
        "lovheim_point": LovheimPoint,
        "emotional_state": EmotionalState,
        "emotion_timeline": EmotionTimeline,
    }


def to_dict(obj: Any) -> dict:
    """Serialize *obj* to a versioned, JSON-compatible dict.

    Parameters
    ----------
    obj:
        Anything with a ``to_dict()`` method — every emotion type,
        :class:`~emotion_algebra.lovheim.LovheimPoint`,
        :class:`~emotion_algebra.state.EmotionalState`, and
        :class:`~emotion_algebra.state.EmotionTimeline`.

    Returns
    -------
    dict
        The object's own payload plus a ``schema`` key.

    Raises
    ------
    TypeError
        If *obj* has no ``to_dict``, or its payload carries no ``type``
        discriminator for :func:`from_dict` to dispatch on.
    """
    if not hasattr(obj, "to_dict"):
        raise TypeError(f"{type(obj).__name__} is not serializable (no to_dict)")

    payload = dict(obj.to_dict())
    if "type" not in payload:
        raise TypeError(
            f"{type(obj).__name__}.to_dict() produced no 'type' key; "
            "from_dict() would have nothing to dispatch on"
        )
    payload[SCHEMA_KEY] = SCHEMA_VERSION
    return payload


def from_dict(data: dict) -> Any:
    """Rebuild an object from a dict produced by :func:`to_dict`.

    Parameters
    ----------
    data:
        A versioned payload.  A payload with no ``schema`` key is assumed to be
        version 1 — that predates the envelope, and reading it is exactly what
        back-compat means.

    Returns
    -------
    object
        An instance of whichever class the ``type`` key names.

    Raises
    ------
    UnsupportedSchemaError
        If ``schema`` names a version this build cannot read.
    ValueError
        If ``type`` is missing or names no known class.

    Examples
    --------
    >>> from emotion_algebra.emotions import get_emotion
    >>> from_dict(to_dict(get_emotion("anger"))).name
    'anger'
    >>> from_dict({"type": "emotion", "name": "fear"}).name  # pre-envelope payload
    'fear'
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a dict payload, got {type(data).__name__}")

    version = data.get(SCHEMA_KEY, 1)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaError(
            f"schema version {version!r} is not readable by this build "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}). "
            "Upgrade emotion-algebra to read it."
        )

    kind = data.get("type")
    if kind is None:
        raise ValueError("payload has no 'type' key; cannot tell what to build")

    loaders = _loaders()
    if kind not in loaders:
        raise ValueError(
            f"unknown type {kind!r}; expected one of {sorted(loaders)}"
        )

    return loaders[kind].from_dict(data)


def to_json(obj: Any, **kwargs) -> str:
    """Serialize *obj* to a versioned JSON string.

    Parameters
    ----------
    obj:
        Anything :func:`to_dict` accepts.
    **kwargs:
        Passed through to :func:`json.dumps` (``indent``, ``sort_keys``, ...).

    Returns
    -------
    str
    """
    return json.dumps(to_dict(obj), **kwargs)


def from_json(blob: str) -> Any:
    """Rebuild an object from a JSON string produced by :func:`to_json`.

    Parameters
    ----------
    blob:
        JSON text.

    Returns
    -------
    object

    Raises
    ------
    UnsupportedSchemaError
        If the payload's schema version is unreadable.
    ValueError
        If the text is not valid JSON, or the payload is malformed.
    """
    return from_dict(json.loads(blob))
