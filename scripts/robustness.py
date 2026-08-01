"""Which of this library's claims survive if its guessed numbers are wrong?

19 of the 22 registered constants are not backed by data or a publication (see
:mod:`emotion_algebra.provenance`). The data that would fit them is not
obtainable — the GRID coordinates are unpublished, Smith & Ellsworth's and
Frijda's tables are paywalled.

You cannot validate a constant you have no data for. You **can** find out which
of your conclusions depend on it.

So: perturb every guessed constant by a random factor, re-derive the library's
claims, and count how often each still holds. A claim that survives is *earned* —
it follows from the structure of the model, not from the numbers someone picked.
A claim that collapses is an **artefact of a chosen coefficient**, and has no
business being asserted in the documentation.

    python scripts/robustness.py [--runs 1000] [--spread 0.5]

The output is committed to ``docs/evidence.md``.
"""
from __future__ import annotations

import argparse
import copy
from typing import Callable, Dict, List

import numpy as np

import emotion_algebra.homeostasis as H
import emotion_algebra.neuro as N
import emotion_algebra.prototypes as P
import emotion_algebra.readout as R
import emotion_algebra.tendency as T
from emotion_algebra.affect import AffectState
from emotion_algebra.appraisal import Appraisal, appraisal_to_affect

# ---------------------------------------------------------------------------
# The claims. Every one of these is asserted somewhere in the docs or the tests.
# ---------------------------------------------------------------------------


def _risk(state: AffectState) -> float:
    """Lerner & Keltner's appraisal-tendency mechanism. Note: no valence term."""
    return 0.5 * ((1.0 - state.potency) / 2.0) + 0.5 * state.unpredictability


def _induce(coping: float, novelty: float, congruence: float = 0.0,
            relevance: float = 0.9) -> AffectState:
    return appraisal_to_affect(
        Appraisal(goal_relevance=relevance, goal_congruence=congruence,
                  coping_potential=coping, novelty=novelty)
    )


CLAIMS: Dict[str, Callable[[], bool]] = {
    # --- the crux: the reason the library exists ---
    "anger and fear separate on potency":
        lambda: P.prototype("anger").potency > 0 > P.prototype("fear").potency,

    "valence/arousal CANNOT separate anger from fear":
        lambda: (abs(P.prototype("anger").valence - P.prototype("fear").valence) < 0.2
                 and abs(P.prototype("anger").arousal - P.prototype("fear").arousal) < 0.2),

    "rage + terror -> distress, not neutrality":
        lambda: (lambda m: (not m.is_origin) and m.valence < -0.3 and m.arousal > 0.4)(
            P.prototype("rage").blend(P.prototype("terror"), 0.5)),

    # --- action readiness ---
    "anger approaches while being unpleasant":
        lambda: (T.action_readiness(P.prototype("anger"))["approach"]
                 > T.action_readiness(P.prototype("anger"))["avoidance"]
                 and P.prototype("anger").valence < 0),

    "anger -> antagonism":
        lambda: T.dominant_tendency(P.prototype("anger")) == "antagonism",
    "fear -> avoidance":
        lambda: T.dominant_tendency(P.prototype("fear")) == "avoidance",
    "sadness -> withdrawal":
        lambda: T.dominant_tendency(P.prototype("sadness")) == "withdrawal",
    "grief -> withdrawal":
        lambda: T.dominant_tendency(P.prototype("grief")) == "withdrawal",
    "disgust -> rejection":
        lambda: T.dominant_tendency(P.prototype("disgust")) == "rejection",
    "surprise -> attending":
        lambda: T.dominant_tendency(P.prototype("surprise")) == "attending",
    "shame -> withdrawal":
        lambda: T.dominant_tendency(P.prototype("shame")) == "withdrawal",
    "joy -> affiliation":
        lambda: T.dominant_tendency(P.prototype("joy")) == "affiliation",
    "at the set point, nothing is demanded (rest)":
        lambda: T.dominant_tendency(H.SET_POINT) == "rest",

    # --- appraisal ---
    "coping alone flips anger and fear":
        lambda: (_induce(0.9, 0.2).potency > 0 > _induce(0.1, 0.8).potency
                 and _induce(0.9, 0.2).valence < 0 and _induce(0.1, 0.8).valence < 0),

    # --- Lerner & Keltner (2001) ---
    "L&K: anger judges risk lower than fear":
        lambda: _risk(_induce(0.9, 0.2)) < _risk(_induce(0.1, 0.8)),
    "L&K: anger patterns with happiness, not fear":
        lambda: (abs(_risk(_induce(0.9, 0.2)) - _risk(_induce(0.8, 0.2, 1.0, 0.8)))
                 < abs(_risk(_induce(0.9, 0.2)) - _risk(_induce(0.1, 0.8)))),

    # --- homeostasis ---
    "rest is NOT the origin":
        lambda: (not H.at_rest(AffectState())) and H.at_rest(H.SET_POINT),
    "rest is mildly positive (the positivity offset)":
        lambda: H.SET_POINT.valence > 0,
    "every state relaxes home":
        lambda: H.at_rest(H.relax(P.prototype("terror"), dt=1e5, half_life=300)),
    "bad news lands harder than good (negativity bias)":
        lambda: (H.perturb(H.SET_POINT, {"negativity": 0.3}).negativity
                 - H.SET_POINT.negativity
                 > H.perturb(H.SET_POINT, {"positivity": 0.3}).positivity
                 - H.SET_POINT.positivity),

    # --- neurochemistry ---
    "coping chemistry flips anger and fear under identical threat":
        lambda: (N.NeuroState(noradrenaline=0.95, cortisol=0.95,
                              dopamine=0.15).to_affect().potency < 0
                 < N.NeuroState(noradrenaline=0.90, dopamine=0.85,
                                testosterone=0.9).to_affect().potency),

    # --- readout ---
    "every prototype names itself":
        lambda: all(__import__("emotion_algebra.readout", fromlist=["x"]).dominant(
            P.prototype(n)) == n for n in P.PROTOTYPES),
}


# ---------------------------------------------------------------------------
# Perturbation. Everything the provenance registry calls "not backed by data".
# ---------------------------------------------------------------------------

def _snapshot() -> dict:
    return {
        "PROTOTYPES": copy.deepcopy(P.PROTOTYPES),
        "LOADINGS": copy.deepcopy(N.LOADINGS),
        "SET_POINT": H.SET_POINT,
        "NEGATIVITY_BIAS": H.NEGATIVITY_BIAS,
        "TEMPERATURE": R.TEMPERATURE,
        "tendency": {k: getattr(T, k) for k in
                     ("HOSTILE_APPROACH_SHARE", "ORIENT_FLOOR", "FLIGHT_URGENCY",
                      "REJECTION_WEIGHT", "SUBMISSION_WEIGHT", "REST_FALLOFF")},
    }


def _restore(s: dict) -> None:
    P.PROTOTYPES.clear(); P.PROTOTYPES.update(s["PROTOTYPES"])
    N.LOADINGS.clear(); N.LOADINGS.update(s["LOADINGS"])
    H.SET_POINT = s["SET_POINT"]
    H.NEGATIVITY_BIAS = s["NEGATIVITY_BIAS"]
    R.TEMPERATURE = s["TEMPERATURE"]
    for k, v in s["tendency"].items():
        setattr(T, k, v)


def _jitter(rng, x: float, spread: float) -> float:
    """Scale by a uniform factor in [1-spread, 1+spread]."""
    return float(x * rng.uniform(1.0 - spread, 1.0 + spread))


def _perturb(rng, spread: float) -> None:
    """Perturb every guessed constant. Fitted ones are left alone."""
    # prototypes: ONLY potency and unpredictability are guesses. Valence and
    # arousal come from Warriner and are not touched.
    for name, s in list(P.PROTOTYPES.items()):
        P.PROTOTYPES[name] = AffectState(
            positivity=s.positivity,           # FITTED — untouched
            negativity=s.negativity,           # FITTED — untouched
            arousal=s.arousal,                 # FITTED — untouched
            potency=float(np.clip(_jitter(rng, s.potency, spread), -1, 1)),
            unpredictability=float(np.clip(_jitter(rng, s.unpredictability, spread), 0, 1)),
        )

    for mod, axes in list(N.LOADINGS.items()):
        N.LOADINGS[mod] = {a: _jitter(rng, w, spread) for a, w in axes.items()}

    sp = _snapshot()["SET_POINT"]
    H.SET_POINT = AffectState(
        positivity=float(np.clip(_jitter(rng, sp.positivity, spread), 0, 1)),
        negativity=float(np.clip(_jitter(rng, sp.negativity, spread), 0, 1)),
        potency=float(np.clip(_jitter(rng, sp.potency, spread), -1, 1)),
        arousal=float(np.clip(_jitter(rng, sp.arousal, spread), 0, 1)),
        unpredictability=float(np.clip(_jitter(rng, sp.unpredictability, spread), 0, 1)),
    )
    H.NEGATIVITY_BIAS = max(1e-6, _jitter(rng, H.NEGATIVITY_BIAS, spread))
    R.TEMPERATURE = max(1e-3, _jitter(rng, R.TEMPERATURE, spread))

    for k in ("HOSTILE_APPROACH_SHARE", "ORIENT_FLOOR", "FLIGHT_URGENCY",
              "REJECTION_WEIGHT", "SUBMISSION_WEIGHT"):
        setattr(T, k, max(1e-6, _jitter(rng, getattr(T, k), spread)))
    T.REST_FALLOFF = max(1, int(round(_jitter(rng, T.REST_FALLOFF, spread))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--spread", type=float, default=0.5, help="e.g. 0.5 = +/-50%%")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="below this, a claim is not earned")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    baseline = _snapshot()

    survived = {name: 0 for name in CLAIMS}
    holds_unperturbed = {}
    for name, claim in CLAIMS.items():
        try:
            holds_unperturbed[name] = bool(claim())
        except Exception:
            holds_unperturbed[name] = False

    for _ in range(args.runs):
        _perturb(rng, args.spread)
        R._cached = None
        for name, claim in CLAIMS.items():
            try:
                if claim():
                    survived[name] += 1
            except Exception:
                pass
        _restore(baseline)

    print(f"ROBUSTNESS  ({args.runs} runs, every guessed constant perturbed "
          f"±{args.spread:.0%})\n")
    print("19 of 22 registered constants are not backed by data. These are the")
    print("claims that survive anyway — and the ones that do not.\n")

    order = sorted(CLAIMS, key=lambda n: -survived[n])
    width = max(len(n) for n in CLAIMS)
    earned, fragile = [], []
    for name in order:
        frac = survived[name] / args.runs
        if not holds_unperturbed[name]:
            verdict = "BROKEN (fails even unperturbed!)"
        elif frac >= args.threshold:
            verdict = "EARNED"
            earned.append(name)
        elif frac >= 0.5:
            verdict = "fragile"
            fragile.append(name)
        else:
            verdict = "NOT EARNED — an artefact of a chosen number"
            fragile.append(name)
        print(f"  {name:<{width}}  {frac:6.1%}  {verdict}")

    print(f"\n{len(earned)}/{len(CLAIMS)} claims are earned "
          f"(hold in ≥{args.threshold:.0%} of perturbations).")
    if fragile:
        print("\nThese depend on numbers we chose, and must not be asserted as")
        print("findings without that caveat:")
        for n in fragile:
            print(f"  - {n}  ({survived[n] / args.runs:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
