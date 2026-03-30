"""Continuous-valued emotion in the 4-axis Hourglass space — v2.0.

:class:`FloatEmotion` is an unrestricted-float counterpart to the integer-lattice
:class:`~emotion_algebra.plutchik.Emotion`.  Use it for:

- Output of ML regression models (VAD regression)
- Weighted blends of named emotions via :class:`~emotion_algebra.state.EmotionalState`
- Projection of high-dimensional embeddings via :meth:`FloatEmotion.from_embedding`
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from emotion_algebra.base import EmotionBase


class FloatEmotion(EmotionBase):
    """A continuous-valued point in the 4-axis Hourglass space.

    Unlike :class:`~emotion_algebra.plutchik.Emotion`, axis values are floats
    and are not constrained to the integer lattice ``{-3, …, 3}``.

    Parameters
    ----------
    sensitivity:
        Sensitivity axis value (anger↑ / fear↓ pole).
    attention:
        Attention axis value (vigilance↑ / surprise↓ pole).
    pleasantness:
        Pleasantness axis value (joy↑ / sadness↓ pole).
    aptitude:
        Aptitude axis value (trust↑ / disgust↓ pole).
    name:
        Optional explicit name override.
    """

    def __init__(
        self,
        sensitivity: float = 0.0,
        attention: float = 0.0,
        pleasantness: float = 0.0,
        aptitude: float = 0.0,
        name: str = "",
    ) -> None:
        self._vector = np.array(
            [sensitivity, attention, pleasantness, aptitude], dtype=float
        )
        self._name = name

    # ------------------------------------------------------------------
    # EmotionBase abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Explicit name if set, otherwise ``~<nearest named emotion>``."""
        if self._name:
            return self._name
        from emotion_algebra.distance import closest_emotion
        return "~" + closest_emotion(self._vector).name

    @property
    def emotional_flow(self) -> float:  # type: ignore[override]
        """Sum of all axis values."""
        return float(np.sum(self._vector))

    @property
    def valence(self) -> float:  # type: ignore[override]
        """Pleasantness-axis component (index 2)."""
        return float(self._vector[2])

    @property
    def arousal(self) -> float:  # type: ignore[override]
        """Maximum absolute axis value."""
        return float(np.max(np.abs(self._vector)))

    @property
    def type(self) -> str:
        """Russell Circumplex category."""
        from emotion_algebra.plutchik import _circumplex_type
        return _circumplex_type(self.valence, self.arousal)

    @property
    def emotion_vector(self) -> list:
        """4-element proxy list for :class:`~emotion_algebra.base.EmotionBase` compatibility."""

        class _Proxy:
            def __init__(self, flow: float) -> None:
                self.emotional_flow = flow

            def __int__(self) -> int:
                return int(self.emotional_flow)

        return [_Proxy(float(v)) for v in self._vector]

    @property
    def as_array(self) -> np.ndarray:  # type: ignore[override]
        """The internal float vector, shape ``(4,)``."""
        return self._vector.copy()

    @property
    def as_matrix(self) -> np.ndarray:  # type: ignore[override]
        """2×2 float array — ``[[sensitivity, attention], [pleasantness, aptitude]]``."""
        s, a, p, ap = self._vector
        return np.array([[s, a], [p, ap]])

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: object) -> "FloatEmotion":
        if isinstance(other, FloatEmotion):
            return FloatEmotion(*self._vector + other._vector)
        if isinstance(other, EmotionBase):
            return FloatEmotion(*(self._vector + other.as_array.astype(float)))
        if isinstance(other, (int, float)):
            return FloatEmotion(*(self._vector + float(other)))
        return NotImplemented

    def __sub__(self, other: object) -> "FloatEmotion":
        if isinstance(other, FloatEmotion):
            return FloatEmotion(*self._vector - other._vector)
        if isinstance(other, EmotionBase):
            return FloatEmotion(*(self._vector - other.as_array.astype(float)))
        if isinstance(other, (int, float)):
            return FloatEmotion(*(self._vector - float(other)))
        return NotImplemented

    def __mul__(self, other: object) -> "FloatEmotion":
        if isinstance(other, (int, float)):
            return FloatEmotion(*(self._vector * float(other)))
        return NotImplemented

    def __truediv__(self, other: object) -> "FloatEmotion":
        if isinstance(other, (int, float)) and other != 0:
            return FloatEmotion(*(self._vector / float(other)))
        return NotImplemented

    def __neg__(self) -> "FloatEmotion":
        return FloatEmotion(*(-self._vector))

    def __abs__(self) -> "FloatEmotion":
        return FloatEmotion()

    def __int__(self) -> int:
        return int(self.emotional_flow)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FloatEmotion):
            return bool(np.allclose(self._vector, other._vector))
        return NotImplemented

    def __repr__(self) -> str:
        return f"FloatEmotion{tuple(round(float(v), 3) for v in self._vector)}"

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "float_emotion",
            "name": self._name,
            "vector": self._vector.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FloatEmotion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        v = data.get("vector", [0.0, 0.0, 0.0, 0.0])
        return cls(*v, name=data.get("name", ""))

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_embedding(
        cls,
        vec: np.ndarray,
        projection_matrix: Optional[np.ndarray] = None,
    ) -> "FloatEmotion":
        """Project a high-dimensional embedding into the 4-axis Hourglass space.

        Parameters
        ----------
        vec:
            1-D array of any length.
        projection_matrix:
            Optional ``(4, len(vec))`` matrix.  When ``None``, the first 4
            components are taken and L2-normalised to ``[-3, 3]`` (the
            Hourglass integer extremes).

        Returns
        -------
        FloatEmotion
        """
        vec = np.asarray(vec, dtype=float).ravel()
        if projection_matrix is not None:
            projected = (projection_matrix @ vec).astype(float)
        else:
            if len(vec) >= 4:
                projected = vec[:4].copy()
            else:
                projected = np.pad(vec.astype(float), (0, 4 - len(vec)))
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm * 3.0
        return cls(*projected[:4])

    @classmethod
    def from_emotion(cls, emotion: EmotionBase) -> "FloatEmotion":
        """Lift a named :class:`~emotion_algebra.base.EmotionBase` into float space.

        Parameters
        ----------
        emotion:
            Any :class:`~emotion_algebra.base.EmotionBase` instance.

        Returns
        -------
        FloatEmotion
            Float copy of *emotion*'s 4-axis vector.
        """
        arr = emotion.as_array.astype(float)
        return cls(*arr)
