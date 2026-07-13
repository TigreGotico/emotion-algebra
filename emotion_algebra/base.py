"""Abstract base for all emotion types.

Every concrete type (:class:`~emotion_algebra.plutchik.Emotion`,
:class:`~emotion_algebra.composite_emotions.CompositeEmotion`,
:class:`~emotion_algebra.feelings.Feeling`,
:class:`~emotion_algebra.float_emotion.FloatEmotion`) is a point in — or a
composite over — Cambria's Hourglass space.  This ABC defines the minimal
interface they must satisfy and supplies the shared numeric operators, numpy
helpers, and the two scalar summaries of an affective state:

``valence``
    The **hedonic axis only** (Pleasantness).  Anger has ``valence == 0``:
    reactivity is orthogonal to hedonics (Russell 1980; Posner et al. 2005).
``polarity``
    Cambria's published Hourglass sentiment formula over *all four* axes.
    Anger has ``polarity < 0``, because a highly sensitive state is aversive
    regardless of which pole it sits on.

The two are deliberately distinct and must not be conflated —
see ``docs/valence_arousal.md``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

#: Hourglass axis order used by ``as_array`` / ``emotion_vector`` everywhere.
AXES = ("sensitivity", "attention", "pleasantness", "aptitude")

#: Absolute value of the most intense named emotion on any axis (``rage`` = 3).
#: Used to normalise integer flows into the ``[-1, 1]`` range Cambria's
#: polarity formula is defined over.
AXIS_MAX = 3.0


def hourglass_polarity(vector) -> float:
    """Cambria's Hourglass polarity for a 4-axis vector.

    Implements the published sentiment formula (Cambria et al. 2012; refined in
    Susanto et al. 2020, *The Hourglass Model Revisited*)::

        polarity = (Pleasantness + |Attention| - |Sensitivity| + Aptitude) / 3

    evaluated over axes normalised to ``[-1, 1]``.  The absolute values on
    Attention and Sensitivity are not a typo: both poles of Sensitivity (anger
    *and* fear) are aversive, and both poles of Attention (vigilance *and*
    surprise) are engaging.  Only Pleasantness and Aptitude carry a signed
    hedonic contribution.

    Parameters
    ----------
    vector:
        Any 4-element array-like in Hourglass axis order
        ``[sensitivity, attention, pleasantness, aptitude]``.

    Returns
    -------
    float
        Sentiment polarity in ``[-1, 1]``.  Positive = pleasant/approach,
        negative = unpleasant/avoid, 0 = neutral or exactly balanced.

    Examples
    --------
    >>> round(hourglass_polarity([0, 0, 3, 0]), 3)   # ecstasy — one axis only
    0.333
    >>> round(hourglass_polarity([2, 0, 0, 0]), 3)   # anger — aversive
    -0.222
    """
    s, at, p, ap = (float(v) / AXIS_MAX for v in np.asarray(vector, dtype=float).ravel()[:4])
    raw = (p + abs(at) - abs(s) + ap) / 3.0
    # Named emotions never leave [-1, 1], but hyper-intense and free-float
    # vectors can overshoot the lattice; the polarity contract is a bounded
    # sentiment score, so saturate rather than leak an out-of-range number.
    return float(max(-1.0, min(1.0, raw)))


@runtime_checkable
class SupportsEmotionVector(Protocol):
    """Structural type for anything with a 4-axis Hourglass representation.

    Consumers that only need the numbers (ML adapters, downstream engines)
    should depend on this Protocol rather than on :class:`EmotionBase`, so
    they stay decoupled from the class hierarchy.
    """

    @property
    def as_array(self) -> np.ndarray:
        """4-element vector ``[sensitivity, attention, pleasantness, aptitude]``."""
        ...


class EmotionBase(ABC):
    """Shared interface for every emotion type.

    All operators return a *new* value rather than mutating ``self`` —
    emotions are immutable value objects. There is no in-place operator
    family (``+=``, ``-=``, ...); ``e += 1`` rebinds the name to a new
    Emotion, it does not mutate the original in place.
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
        """Pleasantness axis component only — the hedonic axis."""
        ...

    @property
    @abstractmethod
    def arousal(self) -> int:
        """Activation magnitude."""
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
    def polarity(self) -> float:
        """Hourglass sentiment polarity in ``[-1, 1]`` — see :func:`hourglass_polarity`.

        Unlike :attr:`valence` (Pleasantness only) this weighs all four axes,
        so anger and fear are both negative and trust is positive.

        Examples
        --------
        >>> from emotion_algebra.emotions import get_emotion
        >>> round(get_emotion("ecstasy").polarity, 3)
        0.333
        >>> get_emotion("anger").polarity < 0
        True
        """
        return hourglass_polarity(self.as_array)

    @property
    def as_array(self) -> np.ndarray:
        """Flow values as a numpy array of shape (4,)."""
        return np.array([e.emotional_flow for e in self.emotion_vector])

    @property
    def as_matrix(self) -> np.ndarray:
        """Flow values as a 2×2 numpy array."""
        s, a, p, ap = self.emotion_vector
        return np.array([[int(s), int(a)], [int(p), int(ap)]])

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.  Overridden by every concrete type."""
        raise NotImplementedError

    def to_json(self, **kwargs) -> str:
        """Serialize to a versioned JSON string.

        See :func:`emotion_algebra.serialization.to_json`.
        """
        from emotion_algebra.serialization import to_json
        return to_json(self, **kwargs)

    def __bool__(self) -> bool:
        return self.emotional_flow != 0

    def __int__(self) -> int:
        """Net activation scalar: sum of all axis emotional_flow values.

        This is *not* a hedonic score. ``int(love)`` returns 4 because joy (flow=2)
        and trust (flow=2) each contribute 2. Negative flows reduce the total.
        Use :attr:`valence` for hedonic tone or :attr:`polarity` for sentiment.
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
