"""The set point — what the affect core falls toward, and why it is not zero.

"Lack of emotion" is not a state
--------------------------------
Core affect is *always on*.  Barrett & Bliss-Moreau (2009) ground it in ongoing
interoceptive/allostatic signalling: there is no moment at which an organism has
no valence and no arousal, any more than there is a moment at which it has no
body temperature.  So the coordinate origin of
:class:`~emotion_algebra.affect.AffectState` is a **mathematical reference, not a
psychological state**.  Nothing lives there.

What an organism actually falls toward is a **set point**, and the evidence says
that set point is *not* the origin.  Cacioppo & Berntson's evaluative space model
gives two asymmetries, both replicated:

* **Positivity offset** — with no stimulus at all, positivity slightly exceeds
  negativity.  This is why an organism at rest *explores* rather than freezing:
  the resting state is mildly appetitive.
* **Negativity bias** — negativity, once engaged, rises more steeply than
  positivity.  "Bad is stronger than good" (Baumeister, Bratslavsky,
  Finkenauer & Vohs, 2001, *Review of General Psychology* 5(4):323-70).

So rest is *mildly positive, low-arousal, mildly in control* — and that, not the
origin, is the attractor.

Three timescales
----------------
This gives the dynamical system an agent actually needs:

===========  =========================================  ==================
layer        what it is                                 timescale
===========  =========================================  ==================
emotion      the current displacement                   seconds - minutes
mood         a slow-moving estimate of where you sit    hours - days
temperament  the constitutional set point (attractor)   stable, per-agent
===========  =========================================  ==================

Emotion relaxes toward mood; mood drifts toward temperament.  This is
Mehrabian's temperament/mood/emotion distinction, made mechanical.

Drives
------
A **need deficit is a displacement from the set point**, and the emotion is the
felt signal of that displacement.  :func:`drive` returns the restoring vector —
the direction and magnitude of what the organism is motivated to reduce.  That is
the quantity a needs-driven agent minimises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from emotion_algebra.affect import CORE_AXES, AffectState

#: Default half-life of a displacement, in seconds.
#:
#: Arbitrary, and exposed as a parameter precisely because it is arbitrary — set
#: it to whatever your tick rate needs.  Emotion-duration data does exist
#: (Verduyn & Lavrijsen 2015 find sadness longest-lasting), but the per-emotion
#: figures are paywalled and could not be verified, so nothing here is fitted to
#: them.  Nothing in the library depends on this value being right.
DEFAULT_HALF_LIFE: float = 300.0

#: How close to the set point counts as "at rest" — a threshold on the core's own
#: metric, roughly 5% of the space's diameter.
#:
#: Worth knowing: the claim ":func:`at_rest` is False at the ORIGIN" survives only
#: **45%** of perturbations (``scripts/robustness.py``), because it holds only
#: while the set point sits further from the origin than this tolerance. The
#: CONCEPT — that core affect is always on, and rest is not a blank state
#: (Barrett & Bliss-Moreau 2009) — is evidenced. The NUMERICAL assertion depends
#: on magnitudes nobody has measured. Do not cite the latter as though it were
#: the former.
REST_TOLERANCE: float = 0.1

#: The default resting set point.
#:
#: Not the origin.  Mildly positive (the positivity offset), low arousal, mildly
#: in control — the state of an organism that is safe, unhurried, and disposed to
#: explore.  Magnitudes are calibrated, not measured: the evidence establishes
#: the *sign* and *ordering* (positivity > negativity at rest), not the numbers.
SET_POINT = AffectState(
    positivity=0.20,        # calibrated: the positivity offset (ESM)
    negativity=0.05,        # calibrated: near-zero, but not zero
    potency=0.15,           # calibrated: mild baseline sense of agency
    arousal=0.20,           # calibrated: awake, not activated
    unpredictability=0.15,  # calibrated: the world is mostly as expected
)

#: How much more steeply negativity responds than positivity.
#:
#: The **direction** is well-evidenced: negative events weigh more than positive
#: ones of equal size.  The **magnitude is invented**, and it is worth being
#: precise about why, because the obvious citation does not say what it is
#: usually quoted as saying.
#:
#: "Bad is stronger than good" (Baumeister, Bratslavsky, Finkenauer & Vohs, 2001,
#: *Review of General Psychology* 5(4):323-70) is a **narrative review**.  It
#: surveys many literatures and reports **no ratio**.  The familiar "bad counts
#: about twice as much" is Kahneman & Tversky's loss-aversion coefficient
#: (λ ≈ 2.25) — fitted to **monetary gambles**, which is a different domain, and
#: never established for affective weighting.
#:
#: So 1.5 is a number we chose.  It is registered ``ASSUMED`` in
#: :mod:`emotion_algebra.provenance`, and ``scripts/robustness.py`` reports which
#: of the library's claims (if any) depend on it.
NEGATIVITY_BIAS: float = 1.5


@dataclass(frozen=True)
class Temperament:
    """An agent's constitutional set point — the attractor it returns to.

    Two agents in identical circumstances feel differently because they fall
    toward different points.  This is what "disposition" means mechanically:
    a cheerful agent has a higher ``set_point.positivity``; an anxious one has a
    lower ``set_point.potency`` and a higher ``set_point.unpredictability``.

    Parameters
    ----------
    set_point:
        The attractor.  Defaults to :data:`SET_POINT`.
    negativity_bias:
        How much more steeply this agent's negativity responds.
    resilience:
        Half-life, in seconds, of a displacement from the set point.  Higher is
        slower to recover.
    """

    set_point: AffectState = SET_POINT
    negativity_bias: float = NEGATIVITY_BIAS
    resilience: float = DEFAULT_HALF_LIFE

    def __post_init__(self) -> None:
        if self.negativity_bias <= 0:
            raise ValueError(
                f"negativity_bias must be positive, got {self.negativity_bias!r}"
            )
        if self.resilience <= 0:
            raise ValueError(f"resilience must be positive, got {self.resilience!r}")


#: The default disposition.
BASELINE_TEMPERAMENT = Temperament()


def drive(state: AffectState, set_point: AffectState = SET_POINT) -> Dict[str, float]:
    """The restoring vector: how far *state* sits from *set_point*, per axis.

    This is the quantity a needs-driven agent minimises.  A positive value means
    the axis must **rise** to reach rest; negative means it must fall.

    Examples
    --------
    >>> from emotion_algebra.prototypes import prototype
    >>> d = drive(prototype("terror"))
    >>> d["negativity"] < 0      # negativity must come down
    True
    >>> d["potency"] > 0         # and control must be restored
    True
    """
    return {
        axis: float(getattr(set_point, axis) - getattr(state, axis))
        for axis in CORE_AXES
    }


def drive_magnitude(state: AffectState, set_point: AffectState = SET_POINT) -> float:
    """How far *state* is from rest, as a single scalar.

    The agent's total unmet regulatory demand.  Zero only at the set point.
    """
    return float(np.linalg.norm(state.as_array - set_point.as_array))


def at_rest(
    state: AffectState,
    set_point: AffectState = SET_POINT,
    tolerance: float = REST_TOLERANCE,
) -> bool:
    """``True`` when *state* is within *tolerance* of the set point.

    This — not ``state == NEUTRAL`` — is what "no particular emotion right now"
    means.  The origin is not a state; the set point is.
    """
    return drive_magnitude(state, set_point) <= float(tolerance)


def relax(
    state: AffectState,
    dt: float,
    half_life: float = DEFAULT_HALF_LIFE,
    toward: AffectState = SET_POINT,
) -> AffectState:
    """Contract *state* toward the set point, halving the gap every *half_life*.

    This is what :meth:`AffectState.decay` should mean.  Decaying toward the
    *origin* would drive an organism to a state it never occupies; decaying
    toward the *set point* returns it to rest.

    Raises
    ------
    ValueError
        If *dt* is negative or *half_life* is not positive.
    """
    if float(dt) < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt!r}")
    if float(half_life) <= 0.0:
        raise ValueError(f"half_life must be positive, got {half_life!r}")

    keep = 0.5 ** (float(dt) / float(half_life))
    moved = toward.as_array + (state.as_array - toward.as_array) * keep
    return AffectState(**dict(zip(CORE_AXES, moved)))


def perturb(
    state: AffectState,
    delta: Dict[str, float],
    negativity_bias: float = NEGATIVITY_BIAS,
) -> AffectState:
    """Displace *state* by *delta*, applying the negativity bias.

    A push on ``negativity`` lands harder than an equal push on ``positivity`` —
    that is the bias, and it is why one insult outweighs one compliment.
    Results are clamped back into the space.

    Raises
    ------
    ValueError
        If *delta* names an axis that does not exist.
    """
    unknown = set(delta) - set(CORE_AXES)
    if unknown:
        raise ValueError(f"unknown axes: {sorted(unknown)}")

    values = {axis: float(getattr(state, axis)) for axis in CORE_AXES}
    for axis, d in delta.items():
        d = float(d)
        if axis == "negativity" and d > 0:
            d *= float(negativity_bias)
        values[axis] += d

    return AffectState(
        positivity=min(1.0, max(0.0, values["positivity"])),
        negativity=min(1.0, max(0.0, values["negativity"])),
        potency=min(1.0, max(-1.0, values["potency"])),
        arousal=min(1.0, max(0.0, values["arousal"])),
        unpredictability=min(1.0, max(0.0, values["unpredictability"])),
    )
