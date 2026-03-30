"""emotion_data/base.py — Abstract base for all emotion types.

All three concrete types (Emotion, CompositeEmotion, Feeling) are points in
or composites of Cambria's Hourglass space.  This ABC defines the minimal
interface they all must satisfy, and provides concrete shared implementations
for the numeric operators and numpy helpers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmotionBase(ABC):
    """Shared interface for Emotion, CompositeEmotion, and Feeling.

    All three types are points in or composites of Cambria's Hourglass space.
    This ABC defines the minimal interface that all three must satisfy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical display name."""
        ...

    @property
    @abstractmethod
    def emotional_flow(self) -> int:
        """Signed net intensity."""
        ...

    @property
    @abstractmethod
    def valence(self) -> int:
        """Pleasantness axis component only."""
        ...

    @property
    @abstractmethod
    def arousal(self) -> int:
        """|emotional_flow| or max component arousal."""
        ...

    @property
    @abstractmethod
    def type(self) -> str:
        """Russell Circumplex category."""
        ...

    @property
    @abstractmethod
    def emotion_vector(self) -> list:
        """4-element list [sensitivity, attention, pleasantness, aptitude]."""
        ...

    # --- Concrete shared implementations ---

    @property
    def as_array(self) -> np.ndarray:
        """Flow values as a numpy array of shape (4,)."""
        return np.array([e.emotional_flow for e in self.emotion_vector])

    @property
    def as_matrix(self) -> np.ndarray:
        """Flow values as a 2×2 numpy array."""
        s, a, p, ap = self.emotion_vector
        return np.array([[int(s), int(a)], [int(p), int(ap)]])

    def __bool__(self) -> bool:
        return self.emotional_flow != 0

    def __int__(self) -> int:
        """Net activation scalar: sum of all axis emotional_flow values.

        This is *not* a hedonic score. ``int(love)`` returns 4 because joy (flow=2)
        and trust (flow=2) each contribute 2. Negative flows reduce the total.
        Use ``valence`` for positive/negative polarity.
        """
        return self.emotional_flow

    def __float__(self) -> float:
        return float(self.emotional_flow)

    def __lt__(self, other: object) -> bool:
        return int(self) < int(other)  # type: ignore[arg-type]

    def __le__(self, other: object) -> bool:
        return int(self) <= int(other)  # type: ignore[arg-type]

    def __gt__(self, other: object) -> bool:
        return int(self) > int(other)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> bool:
        return int(self) >= int(other)  # type: ignore[arg-type]
