"""Mutable emotional state — weighted 4-axis float accumulator with decay — v1.1.

An :class:`EmotionalState` is a continuous-valued position in the Hourglass
space.  It can be updated by :meth:`~EmotionalState.apply`-ing emotions with
optional weight, decayed toward neutral, and queried for its dominant named
emotion.

An :class:`EmotionTimeline` records a sequence of snapshots for analysis.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from emotion_algebra.base import EmotionBase

if TYPE_CHECKING:
    from emotion_algebra.lovheim import LovheimPoint


class EmotionalState:
    """Mutable weighted position in the 4-axis Hourglass space.

    Internally stored as a float ``np.ndarray`` of shape ``(4,)``:
    ``[sensitivity, attention, pleasantness, aptitude]``.

    Parameters
    ----------
    vector:
        Optional initial 4-element array-like.  Defaults to zeros (neutral).
    """

    #: Smoothing factor for :attr:`mood` — the weight each new state carries in
    #: the long-window average.  0.05 means a mood takes roughly 20 updates to
    #: track a sustained change in emotion, which is the point: Mehrabian (1996)
    #: separates *emotion* (a momentary state) from *temperament/mood* (the slow
    #: average it fluctuates around), and a mood that moved as fast as emotion
    #: would not be a mood.
    MOOD_ALPHA = 0.05

    def __init__(self, vector=None) -> None:
        if vector is not None:
            self._vector = np.array(vector, dtype=float)
        else:
            self._vector = np.zeros(4, dtype=float)
        self._mood = self._vector.copy()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply(self, emotion: EmotionBase, weight: float = 1.0) -> "EmotionalState":
        """Blend *emotion* into the state.

        Also nudges :attr:`mood` toward the new state, at :attr:`MOOD_ALPHA`.

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
        self._mood += self.MOOD_ALPHA * (self._vector - self._mood)
        return self

    def decay(self, factor: float = 0.9) -> "EmotionalState":
        """Exponentially decay all axis values toward zero, by a fixed factor.

        Parameters
        ----------
        factor:
            Multiplier in ``(0, 1]``.  ``0.9`` = 10 % decay per step.

        Returns
        -------
        EmotionalState
            *self* — for chaining.

        Raises
        ------
        ValueError
            If *factor* is outside ``(0, 1]``.

        See Also
        --------
        decay_halflife : decay in real time rather than in steps.
        """
        if not 0.0 < float(factor) <= 1.0:
            raise ValueError(f"factor must be in (0, 1], got {factor!r}")
        self._vector *= float(factor)
        return self

    def decay_halflife(self, dt: float, half_life: float) -> "EmotionalState":
        """Decay toward neutral in **real time**, by half-life.

        :meth:`decay` is per-*step*, which means the decay rate is implicitly
        tied to how often you happen to call it.  This is the same exponential
        curve parameterised by something physical instead: after ``dt ==
        half_life`` the state is at exactly half its intensity, regardless of
        how many updates happened in between.

        .. math:: v \\leftarrow v \\cdot 2^{-dt / t_{1/2}}

        Parameters
        ----------
        dt:
            Elapsed time, in the same unit as *half_life*.  Must be ``>= 0``.
        half_life:
            Time for the state to fall to half intensity.  Must be ``> 0``.

        Returns
        -------
        EmotionalState
            *self* — for chaining.

        Raises
        ------
        ValueError
            If ``dt < 0`` or ``half_life <= 0``.

        Examples
        --------
        >>> from emotion_algebra.emotions import get_emotion
        >>> s = EmotionalState().apply(get_emotion("rage"))
        >>> _ = s.decay_halflife(dt=30.0, half_life=30.0)
        >>> float(s.snapshot()[0])  # rage was 3.0
        1.5
        """
        if float(dt) < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt!r}")
        if float(half_life) <= 0.0:
            raise ValueError(f"half_life must be > 0, got {half_life!r}")
        self._vector *= 2.0 ** (-float(dt) / float(half_life))
        return self

    def reset(self) -> "EmotionalState":
        """Zero all axis values, and the mood along with them.

        Returns
        -------
        EmotionalState
            *self* — for chaining.
        """
        self._vector[:] = 0.0
        self._mood[:] = 0.0
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

    def polarity(self) -> float:
        """Hourglass sentiment polarity of the current state, in ``[-1, 1]``."""
        from emotion_algebra.base import hourglass_polarity
        return hourglass_polarity(self._vector)

    @property
    def mood(self) -> np.ndarray:
        """The slow-moving average the emotion fluctuates around.

        Mehrabian (1996) distinguishes *emotion* — where you are right now,
        which is :meth:`snapshot` — from *mood/temperament*, the baseline you
        return to.  This is an exponential moving average of the state at
        :attr:`MOOD_ALPHA`, so it lags: one burst of anger barely moves it,
        while a sustained one eventually does.

        Returns
        -------
        np.ndarray
            Shape ``(4,)``.  A copy — mutating it does not affect the state.

        Examples
        --------
        >>> from emotion_algebra.emotions import get_emotion
        >>> s = EmotionalState()
        >>> _ = s.apply(get_emotion("rage"))
        >>> float(s.snapshot()[0]) > float(s.mood[0])  # mood lags the spike
        True
        """
        return self._mood.copy()

    def dominant_mood(self) -> Optional[EmotionBase]:
        """The named emotion nearest the current :attr:`mood`.

        Returns
        -------
        Emotion or None
            ``None`` if the mood is exactly neutral.
        """
        if np.allclose(self._mood, 0):
            return None
        from emotion_algebra.distance import closest_emotion
        return closest_emotion(self._mood)

    def to_lovheim(self) -> "LovheimPoint":
        """Read the current state out as a point in Lövheim's neurochemical cube.

        The live neurotransmitter readout for an agent: feed it appraisals, ask
        it what its monoamines are doing.  See :mod:`emotion_algebra.lovheim`
        for what this projection can and cannot represent.

        Returns
        -------
        LovheimPoint

        Examples
        --------
        >>> from emotion_algebra.emotions import get_emotion
        >>> EmotionalState().apply(get_emotion("rage")).to_lovheim().closest_affect()
        'anger/rage'
        """
        from emotion_algebra.float_emotion import FloatEmotion
        from emotion_algebra.lovheim import LovheimPoint
        return LovheimPoint.from_float_emotion(FloatEmotion(*self._vector))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "emotional_state",
            "vector": self._vector.tolist(),
            "mood": self._mood.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        """Deserialize from a dict produced by :meth:`to_dict`.

        A payload with no ``mood`` key predates the mood axis; its mood seeds
        from the state vector, exactly as a fresh state's would.
        """
        state = cls(vector=data["vector"])
        if "mood" in data:
            state._mood = np.array(data["mood"], dtype=float)
        return state

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

    @property
    def snapshots(self) -> List[np.ndarray]:
        """The recorded vectors, oldest first (a copy — mutating it is a no-op)."""
        return list(self._snapshots)

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

    def lovheim_sequence(self) -> list:
        """Project every snapshot into Lövheim's cube.

        The neurochemical drift of the timeline — what
        :func:`~emotion_algebra.viz.plot_timeline` plots when asked for the
        monoamine view.

        Returns
        -------
        list of LovheimPoint
        """
        from emotion_algebra.float_emotion import FloatEmotion
        from emotion_algebra.lovheim import LovheimPoint
        return [
            LovheimPoint.from_float_emotion(FloatEmotion(*s)) for s in self._snapshots
        ]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "emotion_timeline",
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
