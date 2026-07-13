"""Every model, registered as a view on the affect core.

Importing this module wires up the conversion graph.  After it, any registered
model converts to any other — see :mod:`emotion_algebra.projection`.

The headline: **PAD is now an exact coordinate projection.**  Pleasure is valence,
Arousal is arousal, Dominance is potency.  The old implementation had to
*reconstruct* dominance by fitted regression over three hand-tuned weights,
because the Hourglass axes had thrown the dimension away — and roughly 40% of the
PAD cube was unreachable as a result.  Give the core a potency axis, as the
evidence says you must, and the whole apparatus evaporates.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from emotion_algebra.affect import AffectState
from emotion_algebra.neuro import BASELINE_LEVEL, MODULATORS, NeuroState
from emotion_algebra.projection import Fidelity, register_view

# ---------------------------------------------------------------------------
# PAD / VAD — exact
# ---------------------------------------------------------------------------


def _pad_to_core(pad: Tuple[float, float, float]) -> AffectState:
    pleasure, arousal, dominance = (float(v) for v in pad)
    # Split signed pleasure back into the two hedonic channels. This is the one
    # place PAD is genuinely poorer than the core: it cannot express ambivalence,
    # so the lift assumes none.
    return AffectState(
        positivity=max(0.0, pleasure),
        negativity=max(0.0, -pleasure),
        potency=dominance,
        arousal=arousal,
        unpredictability=0.0,
    )


def _core_to_pad(state: AffectState) -> Tuple[float, float, float]:
    return (state.valence, state.arousal, state.potency)


register_view(
    "pad",
    to_core=_pad_to_core,
    from_core=_core_to_pad,
    fidelity=Fidelity.LOSSY,
    loses=(
        "unpredictability (PAD has no such axis), and ambivalence — PAD's single "
        "signed Pleasure cannot represent positivity and negativity co-active, so "
        "bittersweet reads as mild. Pleasure/Arousal/Dominance themselves are "
        "carried EXACTLY: they are valence, arousal and potency."
    ),
)


# ---------------------------------------------------------------------------
# Russell's circumplex — exact on its two axes
# ---------------------------------------------------------------------------


def _circumplex_to_core(va: Tuple[float, float]) -> AffectState:
    valence, arousal = (float(v) for v in va)
    return AffectState(
        positivity=max(0.0, valence),
        negativity=max(0.0, -valence),
        arousal=arousal,
    )


def _core_to_circumplex(state: AffectState) -> Tuple[float, float]:
    return (state.valence, state.arousal)


register_view(
    "circumplex",
    to_core=_circumplex_to_core,
    from_core=_core_to_circumplex,
    fidelity=Fidelity.LOSSY,
    loses=(
        "potency and unpredictability. This is why the circumplex cannot tell "
        "anger from fear: they differ on potency, and it has no potency axis. "
        "Valence and arousal themselves are carried exactly."
    ),
)


# ---------------------------------------------------------------------------
# Cambria's Hourglass — heuristic
# ---------------------------------------------------------------------------

_AXIS_MAX = 3.0


def _hourglass_to_core(vec) -> AffectState:
    sensitivity, attention, pleasantness, aptitude = (
        float(v) / _AXIS_MAX for v in np.asarray(vec, dtype=float).ravel()[:4]
    )
    # Sensitivity conflates two things the evidence separates: it is negative in
    # valence at BOTH poles (which is why the Hourglass's own polarity formula
    # has to take its absolute value) while its SIGN is really potency.
    hedonic = pleasantness + aptitude * 0.5
    return AffectState(
        positivity=max(0.0, min(1.0, hedonic)),
        negativity=max(0.0, min(1.0, -hedonic + abs(sensitivity) * 0.5)),
        potency=max(-1.0, min(1.0, sensitivity)),
        arousal=max(abs(sensitivity), abs(attention), abs(pleasantness), abs(aptitude)),
        unpredictability=max(0.0, min(1.0, -attention if attention < 0 else attention * 0.3)),
    )


def _core_to_hourglass(state: AffectState) -> np.ndarray:
    return np.array(
        [
            state.potency * _AXIS_MAX,
            (0.3 * state.arousal - state.unpredictability) * _AXIS_MAX,
            state.valence * _AXIS_MAX,
            0.0,
        ],
        dtype=float,
    )


register_view(
    "hourglass",
    to_core=_hourglass_to_core,
    from_core=_core_to_hourglass,
    fidelity=Fidelity.HEURISTIC,
    loses=(
        "Sensitivity conflates negative valence with potency: it is unpleasant at "
        "BOTH poles (anger and fear alike), which is why the Hourglass's own "
        "polarity formula must take its absolute value. That conflation cannot be "
        "undone, so the split back into valence and potency is a judgement call. "
        "The Hourglass also has no arousal axis (it derives one as max|axis|) and "
        "no way to represent ambivalence. It is graded METAPHOR — see "
        "emotion_algebra.evidence."
    ),
)


# ---------------------------------------------------------------------------
# Plutchik's wheel — a lexicon, not a geometry
# ---------------------------------------------------------------------------


def _plutchik_to_core(name: str) -> AffectState:
    from emotion_algebra.prototypes import prototype

    return prototype(name)


def _core_to_plutchik(state: AffectState) -> str:
    from emotion_algebra.readout import dominant

    return dominant(state)


register_view(
    "plutchik",
    to_core=_plutchik_to_core,
    from_core=_core_to_plutchik,
    fidelity=Fidelity.LOSSY,
    loses=(
        "everything except the name. A single label is the argmax of a "
        "distribution (see emotion_algebra.readout), so the runners-up — where "
        "the gradient structure lives — are discarded. Note also that the wheel's "
        "ANTIPODAL structure is graded METAPHOR: Smith & Schneider (2009) found "
        "no empirical support for it, and its flagship pair is refuted outright — "
        "anger and fear are neighbours, not opposites."
    ),
)


# ---------------------------------------------------------------------------
# Lövheim's cube — speculative, kept because downstream consumes it
# ---------------------------------------------------------------------------


def _lovheim_to_core(point) -> AffectState:
    # The cube's three monoamines are a subset of ours, so this is a lift.
    neuro = NeuroState(
        dopamine=float(point.dopamine),
        noradrenaline=float(point.noradrenaline),
        serotonin=float(point.serotonin),
    )
    return neuro.to_affect()


def _core_to_lovheim(state: AffectState):
    from emotion_algebra.lovheim import LovheimPoint

    neuro = NeuroState.from_affect(state)
    return LovheimPoint(
        serotonin=float(neuro.serotonin),
        dopamine=float(neuro.dopamine),
        noradrenaline=float(neuro.noradrenaline),
    )


register_view(
    "lovheim",
    to_core=_lovheim_to_core,
    from_core=_core_to_lovheim,
    fidelity=Fidelity.HEURISTIC,
    loses=(
        "four of the seven modulators (acetylcholine, cortisol, opioids, "
        "testosterone), so coping-related chemistry collapses into the three "
        "monoamines. More importantly the cube is graded SPECULATIVE: it maps "
        "monoamines onto discrete emotion NAMES, a claim that has never been "
        "empirically tested, in a venue that did not practise external peer "
        "review. Convert to it if downstream needs it; do not cite it."
    ),
)


# ---------------------------------------------------------------------------
# Our neurochemistry — lossy but principled
# ---------------------------------------------------------------------------

register_view(
    "neuro",
    to_core=lambda ns: ns.to_affect(),
    from_core=NeuroState.from_affect,
    fidelity=Fidelity.LOSSY,
    loses=(
        "Seven modulators do not determine five axes, nor the reverse. The "
        "inverse returns the MINIMUM-NORM chemistry consistent with the affect — "
        "the least dramatic explanation of the state — so a round trip through "
        "neuro flattens unusual chemistry toward the ordinary. The forward map "
        "(chemistry -> affect) is the trustworthy direction; it rests on "
        "computational roles, not on emotion labels."
    ),
)
