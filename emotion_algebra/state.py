"""Mutable emotional state — weighted 4-axis float accumulator with decay — v1.1.

An :class:`EmotionalState` is a continuous-valued position in the Hourglass
space.  It can be updated by :meth:`~EmotionalState.apply`-ing emotions with
optional weight, decayed toward neutral, and queried for its dominant named
emotion.

An :class:`EmotionTimeline` records a sequence of snapshots for analysis.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from emotion_algebra.base import EmotionBase


class EmotionalState:
    """Mutable weighted position in the 4-axis Hourglass space.

    Internally stored as a float ``np.ndarray`` of shape ``(4,)``:
    ``[sensitivity, attention, pleasantness, aptitude]``.

    Parameters
    ----------
    vector:
        Optional initial 4-element array-like.  Defaults to zeros (neutral).
    """

    def __init__(self, vector=None) -> None:
        if vector is not None:
            self._vector = np.array(vector, dtype=float)
        else:
            self._vector = np.zeros(4, dtype=float)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply(self, emotion: EmotionBase, weight: float = 1.0) -> "EmotionalState":
        """Blend *emotion* into the state.

        Parameters
        ----------
        emotion:
            Any :class:`~emotion_algebra.base.EmotionBase` instance.
        weight:
            Scalar multiplier.  Negative weights suppress an emotion.

        Returns
        -------
        EmotionalState
            *self* — for chaining.
        """
        self._vector += weight * emotion.as_array.astype(float)
        return self

    def decay(self, factor: float = 0.9) -> "EmotionalState":
        """Exponentially decay all axis values toward zero.

        Parameters
        ----------
        factor:
            Multiplier in ``(0, 1]``.  ``0.9`` = 10 % decay per step.

        Returns
        -------
        EmotionalState
            *self* — for chaining.
        """
        self._vector *= factor
        return self

    def reset(self) -> "EmotionalState":
        """Zero all axis values.

        Returns
        -------
        EmotionalState
            *self* — for chaining.
        """
        self._vector[:] = 0.0
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def dominant(self) -> Optional[EmotionBase]:
        """Return the named emotion nearest to the current state.

        Returns
        -------
        Emotion or None
            ``None`` if the state is exactly zero (neutral).
        """
        if np.allclose(self._vector, 0):
            return None
        from emotion_algebra.distance import closest_emotion
        return closest_emotion(self._vector)

    def snapshot(self) -> np.ndarray:
        """Return a copy of the current 4D float vector.

        Returns
        -------
        np.ndarray
            Shape ``(4,)`` — ``[sensitivity, attention, pleasantness, aptitude]``.
        """
        return self._vector.copy()

    def valence(self) -> float:
        """Pleasantness-axis component of the current state."""
        return float(self._vector[2])

    def arousal(self) -> float:
        """Peak activation — max absolute axis value."""
        return float(np.max(np.abs(self._vector)))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {"vector": self._vector.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(vector=data["vector"])

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def __add__(self, other: "EmotionalState") -> "EmotionalState":
        if isinstance(other, EmotionalState):
            return EmotionalState(self._vector + other._vector)
        return NotImplemented

    def __mul__(self, scalar: float) -> "EmotionalState":
        if isinstance(scalar, (int, float)):
            return EmotionalState(self._vector * float(scalar))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EmotionalState):
            return bool(np.allclose(self._vector, other._vector))
        return NotImplemented

    def __repr__(self) -> str:
        dom = self.dominant()
        name = dom.name if dom else "neutral"
        return f"EmotionalState({name}, {self._vector.round(2).tolist()})"


class EmotionTimeline:
    """Ordered sequence of :class:`EmotionalState` snapshots.

    Useful for tracking emotional state evolution across conversation turns,
    game events, or any time-ordered sequence.

    Parameters
    ----------
    label:
        Optional human-readable name for this timeline.
    """

    def __init__(self, label: str = "") -> None:
        self.label: str = label
        self._snapshots: List[np.ndarray] = []

    def append(self, state: EmotionalState) -> "EmotionTimeline":
        """Record the current vector of *state*.

        Parameters
        ----------
        state:
            An :class:`EmotionalState`; its current vector is copied.

        Returns
        -------
        EmotionTimeline
            *self* — for chaining.
        """
        self._snapshots.append(state.snapshot())
        return self

    def drift(self) -> np.ndarray:
        """Net displacement vector from first to last snapshot.

        Returns
        -------
        np.ndarray
            Shape ``(4,)``.  Zero array if fewer than 2 snapshots.
        """
        if len(self._snapshots) < 2:
            return np.zeros(4, dtype=float)
        return self._snapshots[-1] - self._snapshots[0]

    def dominant_sequence(self) -> list:
        """Return the nearest named emotion for each recorded snapshot.

        Returns
        -------
        list of Emotion
        """
        from emotion_algebra.distance import closest_emotion
        return [closest_emotion(s) for s in self._snapshots]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "label": self.label,
            "snapshots": [s.tolist() for s in self._snapshots],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionTimeline":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        tl = cls(label=data.get("label", ""))
        tl._snapshots = [np.array(s) for s in data.get("snapshots", [])]
        return tl

    def __len__(self) -> int:
        return len(self._snapshots)

    def __repr__(self) -> str:
        return f"EmotionTimeline({self.label!r}, {len(self)} snapshots)"
