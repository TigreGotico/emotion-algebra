"""Calibrate the prototype coordinates against human norms.

The prototypes shipped as *hand-calibrated* values, reasoned from the appraisal
profiles the literature reports. Benchmarking them against Warriner, Kuperman &
Brysbaert (2013) — human valence/arousal/dominance norms for 13,915 lemmas —
showed valence and arousal were close but **potency was systematically inflated
across the anger family**: humans rate ``rage`` at −0.21 dominance; the
hand-calibrated value was +0.80.

The lesson is a real one about the construct. *Being enraged is not being in
control.* Anger is **more** dominant than fear — that ordering is exactly what
the appraisal literature predicts, and it survives — but anger is not
**absolutely** dominant. Guessing got the sign of the contrast right and the
magnitude badly wrong.

So the three axes humans have rated are taken from the humans, and only
``unpredictability`` — which Warriner does not measure — stays reasoned from the
literature.

Run:  python scripts/validate/calibrate_prototypes.py --warriner <csv>
It prints a PROTOTYPES table to paste into emotion_algebra/prototypes.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

#: Unpredictability per emotion. Warriner has no such axis, so these stay
#: reasoned from the appraisal literature: fear and surprise are the
#: high-uncertainty emotions; anger is a *certain* emotion (you know who did it),
#: which is the Lerner & Keltner (2001) certainty contrast.
UNPREDICTABILITY = {
    "serenity": 0.05, "joy": 0.10, "ecstasy": 0.15,
    "pensiveness": 0.10, "sadness": 0.15, "grief": 0.20,
    "annoyance": 0.15, "anger": 0.15, "rage": 0.10,
    "apprehension": 0.45, "fear": 0.70, "terror": 0.90,
    "acceptance": 0.10, "trust": 0.10, "admiration": 0.15,
    "boredom": 0.05, "disgust": 0.15, "loathing": 0.15,
    "interest": 0.30, "anticipation": 0.35, "vigilance": 0.35,
    "distraction": 0.55, "surprise": 0.85, "amazement": 1.00,
    "distress": 0.50, "shame": 0.15,
}

#: Words with no Warriner entry, or whose entry is a homograph of something else.
#: Filled from the nearest rated synonym, named here so the choice is auditable.
SUBSTITUTES = {
    "pensiveness": "pensive",
    "amazement": "amazed",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warriner", required=True, type=Path)
    args = ap.parse_args()

    import pandas as pd

    w = pd.read_csv(args.warriner).drop_duplicates("Word").set_index("Word")

    print("PROTOTYPES = {")
    missing = []
    for name, unp in UNPREDICTABILITY.items():
        key = SUBSTITUTES.get(name, name)
        if key not in w.index:
            missing.append(name)
            continue
        g = w.loc[key]
        valence = (float(g["V.Mean.Sum"]) - 5.0) / 4.0   # -> [-1, 1]
        arousal = (float(g["A.Mean.Sum"]) - 1.0) / 8.0   # -> [0, 1]
        potency = (float(g["D.Mean.Sum"]) - 5.0) / 4.0   # -> [-1, 1]

        pos = max(0.0, valence)
        neg = max(0.0, -valence)
        note = f"  # {key}" if key != name else ""
        print(
            f'    "{name}":{" " * (14 - len(name))}'
            f"AffectState(positivity={pos:.2f}, negativity={neg:.2f}, "
            f"potency={potency:+.2f}, arousal={arousal:.2f}, "
            f"unpredictability={unp:.2f}),{note}"
        )
    print("}")
    if missing:
        print(f"\n# NOT IN WARRINER (keep the literature-reasoned value): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
