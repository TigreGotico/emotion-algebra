"""PAD / VAD interoperability — the field's lingua franca.

Most affective-computing datasets and models speak **PAD**: Pleasure, Arousal,
Dominance (Mehrabian & Russell 1974; Mehrabian 1996), each conventionally in
``[-1, 1]``.  "VAD" (Valence-Arousal-Dominance) is the same space under a
different name, and is what the NRC-VAD lexicon and most dimensional emotion
regressors emit.

This module projects between PAD and the 4-axis Hourglass space.

Dominance is a projection, not an axis
--------------------------------------
The Hourglass has no dominance axis and this library will not invent one.  What
it has is **Aptitude** (trust ↔ disgust) and **Sensitivity** (anger ↔ fear),
and the appraisal literature is clear that felt dominance/potency tracks
exactly those two: anger is high-dominance and fear is low-dominance at
identical valence and arousal, which is the *coping* distinction (Scherer 2001)
that Sensitivity encodes.  So dominance is *derived*.

The round trip ``PAD → Hourglass → PAD`` is nonetheless **exact on pleasure and
dominance** wherever the target is reachable — see :func:`from_pad`, which
inverts :func:`to_pad` in closed form.  Arousal is the one component that cannot
be recovered exactly, because :func:`to_pad` reads it as a *max* over the axes.

References
----------
Mehrabian, A. (1996). Pleasure-arousal-dominance: A general framework for
describing and measuring individual differences in temperament.  *Current
Psychology*, 14(4), 261–292.

Mehrabian, A., & Russell, J. A. (1974). *An Approach to Environmental
Psychology*.  MIT Press.

Mohammad, S. M. (2018). Obtaining reliable human ratings of valence, arousal,
and dominance for 20,000 English words.  *ACL 2018*.  (The NRC-VAD lexicon.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from emotion_algebra.base import AXIS_MAX, hourglass_polarity

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.float_emotion import FloatEmotion


class PAD(NamedTuple):
    """A point in Pleasure-Arousal-Dominance space, each component in ``[-1, 1]``."""

    pleasure: float
    arousal: float
    dominance: float

    #: Alias — the VAD literature calls :attr:`pleasure` "valence".
    @property
    def valence(self) -> float:
        """Alias for :attr:`pleasure` (the VAD naming)."""
        return self.pleasure


#: Weights of the dominance projection over the Hourglass axes.
#:
#: Dominance is felt *potency*: the sense of being able to act on the situation.
#:
#: * **Sensitivity** is the dominant term and the only signed one.  Its two
#:   poles are precisely the high- and low-potency responses to obstruction —
#:   anger (fight, dominant) and fear (flee, submissive).  This is the single
#:   most-replicated fact about dominance in the PAD literature, and it is why
#:   PAD needs a third axis at all: anger and fear are near-identical in
#:   pleasure and arousal and differ almost entirely here.
#: * **Aptitude** adds felt competence — trust/mastery reads as potent,
#:   disgust/rejection as impotent.
#: * **Attention** contributes weakly: engagement is mildly empowering, but
#:   surprise (the negative pole) is a loss of control, so the signed value is
#:   what matters.
_DOMINANCE_WEIGHTS = {
    "sensitivity": 0.60,   # calibrated: anger ≫ fear on dominance at equal valence/arousal
    "aptitude":    0.30,   # calibrated: competence is potency
    "attention":   0.10,   # calibrated: engagement is weakly empowering
}


def to_pad(emotion: "EmotionBase") -> PAD:
    """Project any emotion into PAD space.

    * **Pleasure** = :attr:`~emotion_algebra.base.EmotionBase.polarity` —
      Cambria's four-axis sentiment score, already normalised to ``[-1, 1]`` and
      already the library's answer to "how pleasant is this".
    * **Arousal** = peak axis magnitude, normalised — the same activation
      notion :attr:`~emotion_algebra.base.EmotionBase.arousal` uses.  PAD
      arousal is unsigned in practice (no emotion has "negative activation"),
      so this lands in ``[0, 1]``.
    * **Dominance** = the weighted projection in :data:`_DOMINANCE_WEIGHTS`.

    Parameters
    ----------
    emotion:
        Any :class:`~emotion_algebra.base.EmotionBase`.

    Returns
    -------
    PAD

    Examples
    --------
    >>> from emotion_algebra.emotions import get_emotion
    >>> to_pad(get_emotion("rage")).dominance > 0     # anger is dominant
    True
    >>> to_pad(get_emotion("terror")).dominance < 0   # fear is submissive
    True
    >>> round(to_pad(get_emotion("ecstasy")).pleasure, 3)
    0.333
    """
    s, at, p, ap = (float(v) / AXIS_MAX for v in np.asarray(emotion.as_array, dtype=float).ravel()[:4])

    pleasure = hourglass_polarity(emotion.as_array)
    arousal = float(min(1.0, max(abs(s), abs(at), abs(p), abs(ap))))
    dominance = (
        s * _DOMINANCE_WEIGHTS["sensitivity"]
        + ap * _DOMINANCE_WEIGHTS["aptitude"]
        + at * _DOMINANCE_WEIGHTS["attention"]
    )
    return PAD(
        pleasure=float(min(1.0, max(-1.0, pleasure))),
        arousal=arousal,
        dominance=float(min(1.0, max(-1.0, dominance))),
    )


def from_pad(pleasure: float, arousal: float, dominance: float) -> "FloatEmotion":
    """Lift a PAD point into the 4-axis Hourglass space.

    Three numbers cannot pin down four axes, so the lift is under-determined and
    is closed by one stated choice: the hedonic load splits evenly between
    Pleasantness and Aptitude, which is what Cambria's polarity formula itself
    implies — it weights the two identically.  With that choice the system is
    square and solvable in closed form, and the result is an **exact right
    inverse of :func:`to_pad` on pleasure and dominance**: ``to_pad(from_pad(P,
    A, D))`` returns ``P`` and ``D`` back to machine precision.

    **Arousal is the exception, by construction.**  :func:`to_pad` reads arousal
    as a *max* over the axes, and a max destroys the information needed to undo
    it, so arousal is honoured as an activation floor rather than recovered
    exactly.  Where a stronger hedonic or dominance demand forces an axis above
    the requested arousal, the returned arousal is higher than asked.

    **Not every PAD triple is reachable.**  The Hourglass axes are bounded, so
    combinations like "maximally pleasant, no arousal" have no pre-image; those
    saturate at the cube face and round-trip only approximately.  Roughly 40% of
    the raw PAD cube is unreachable this way — a property of the two spaces, not
    a defect of the map.

    Outputs are on the Hourglass scale ``[-3, 3]``, matching the integer lattice.

    Parameters
    ----------
    pleasure, arousal, dominance:
        PAD components.  ``pleasure`` and ``dominance`` in ``[-1, 1]``,
        ``arousal`` in ``[0, 1]``.

    Returns
    -------
    FloatEmotion

    Raises
    ------
    ValueError
        If any component is outside its documented range, or is not finite.

    Examples
    --------
    >>> fe = from_pad(pleasure=0.9, arousal=0.9, dominance=0.1)
    >>> fe.valence > 0
    True
    """
    from emotion_algebra.float_emotion import FloatEmotion

    for label, value, lo, hi in (
        ("pleasure", pleasure, -1.0, 1.0),
        ("arousal", arousal, 0.0, 1.0),
        ("dominance", dominance, -1.0, 1.0),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value!r}")
        if not lo <= float(value) <= hi:
            raise ValueError(f"{label} must be in [{lo}, {hi}], got {value!r}")

    P, A, D = float(pleasure), float(arousal), float(dominance)

    # Attention absorbs the requested activation; the other axes then have to
    # work around it.  Arousal is the one component that cannot be inverted
    # exactly — to_pad reads it as a *max* over the axes, and a max discards the
    # information needed to undo it — so it is honoured as an activation floor.
    attention = A

    # Pleasure and dominance, by contrast, are exactly invertible.  to_pad reads
    #   pleasure   = (p + |at| - |s| + ap) / 3          (Cambria's polarity)
    #   dominance  = 0.60 s + 0.30 ap + 0.10 at         (_DOMINANCE_WEIGHTS)
    # The polarity formula weights Pleasantness and Aptitude identically (both
    # +1), so it gives no reason to prefer either: split the hedonic load evenly
    # (p = ap).  That leaves two equations in the two unknowns s and ap, which
    # solve in closed form once the sign of s is known.  |s| makes the system
    # piecewise, so try both branches and keep the self-consistent one.
    w_s = _DOMINANCE_WEIGHTS["sensitivity"]
    w_ap = _DOMINANCE_WEIGHTS["aptitude"]
    w_at = _DOMINANCE_WEIGHTS["attention"]

    sensitivity = hedonic = None
    for sign in (1.0, -1.0) if D >= 0 else (-1.0, 1.0):
        # p + ap = 3P - |at| + sign*s  =:  S,  with p = ap = S/2
        # D = w_s*s + w_ap*(S/2) + w_at*at
        denom = w_s + w_ap * sign / 2.0
        s = (D - w_at * attention - w_ap * (3.0 * P - abs(attention)) / 2.0) / denom
        if (s >= 0.0) == (sign > 0.0):
            sensitivity = s
            hedonic = (3.0 * P - abs(attention) + sign * s) / 2.0
            break
    if sensitivity is None:  # pragma: no cover — one branch is always consistent
        sensitivity = D
        hedonic = (3.0 * P - abs(attention) + abs(D)) / 2.0

    pleasantness = aptitude = hedonic

    axes = np.clip(
        np.array([sensitivity, attention, pleasantness, aptitude]), -1.0, 1.0
    )
    return FloatEmotion(*(axes * AXIS_MAX))


def pad_distance(a: "EmotionBase", b: "EmotionBase") -> float:
    """Euclidean distance between two emotions **in PAD space**.

    Useful for comparing against PAD-native resources (the NRC-VAD lexicon,
    dimensional regressors) on their own terms.  For distance in the library's
    own space use :func:`~emotion_algebra.distance.emotion_distance`.
    """
    return float(np.linalg.norm(np.array(to_pad(a)) - np.array(to_pad(b))))
