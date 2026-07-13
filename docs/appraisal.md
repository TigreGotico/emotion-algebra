# Appraisal

`emotion_algebra.appraisal` implements Scherer's Component Process Model — the
idea that an emotion is not triggered by an event but by an organism's
**appraisal** of it, along a fixed sequence of Stimulus Evaluation Checks (SECs).

Scherer, K. R. (2001), *Appraisal considered as a process of multilevel sequential
checking*, in Scherer, Schorr & Johnstone (eds), Appraisal Processes in Emotion,
OUP, 92-120.

```python
from emotion_algebra import Appraisal, appraisal_to_emotion

fight = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
                  agency="other", coping_potential="high")
flight = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
                   agency="other", coping_potential="low")

appraisal_to_emotion(fight).name    # 'anger'
appraisal_to_emotion(flight).name   # 'fear'
```

The *same* obstructing event yields anger or fear depending on nothing but
whether the organism believes it can cope. That flip is the model's whole point.

## The checks

`Appraisal` takes **categorical** values:

| Field | Values | Meaning |
|---|---|---|
| `novelty` | `expected` / `unexpected` | how unexpected the event is |
| `goal_relevance` | `relevant` / `irrelevant` | does it matter to an active goal |
| `goal_congruence` | `congruent` / `incongruent` | does it help or obstruct the goal |
| `agency` | `self` / `other` / `circumstance` | who caused it |
| `coping_potential` | `high` / `low` | can I act on it |

`Agency` is a `StrEnum`, so it compares equal to its string value and serializes
cleanly — `agency=Agency.OTHER` and `agency="other"` are interchangeable.

## Two mappings

`appraisal_to_emotion` is the **discrete** mapping: a rule table, first match
wins, `Neutrality` when nothing matches. It returns one of the 24 named emotions.

`appraisal_to_float_emotion` is the **continuous** mapping: it scores all four
Hourglass axes rather than snapping to a name, so it can express partial and
blended appraisals.

```python
appraisal_to_float_emotion(fight).as_array     # sensitivity +0.25  (anger pole)
appraisal_to_float_emotion(flight).as_array    # sensitivity -0.75  (fear pole)
```

Note the continuous map produces *mild* magnitudes here, so `closest_emotion` on
these vectors names the low-intensity tier (pensiveness / apprehension) rather
than anger / fear. The axis sign — which is what carries the fight-or-flight
distinction — flips exactly as the theory requires. If you want a name, use the
discrete map; if you want a vector to do arithmetic on, use the continuous one.

## Coefficients are named, not magic

Every calibrated number lives in `APPRAISAL_COEFFS`, a named module-level table,
and each entry carries either a citation or an explicit `# calibrated:` rationale.
No bare literals are buried in the arithmetic — if you disagree with a weight, you
can find it, change it, and see exactly which check it moves.

## Threat responds even when coping is zero

Sensitivity is computed as:

```
sensitivity = obstruction · coping · SENSITIVITY_COPING_GAIN
            − obstruction · SENSITIVITY_THREAT_FLOOR
```

The second term is the load-bearing one. Without it, a maximally-threatening
event appraised with *zero coping potential* yields **no fear at all** — the whole
sensitivity term multiplies through by `coping = 0`. An organism that cannot cope
with a threat should be more afraid, not less. The threat floor makes obstruction
register regardless of coping, and coping then decides whether it reads as fear
(flee) or anger (fight).

## Straight through to neurochemistry

```python
from emotion_algebra import appraisal_to_lovheim

appraisal_to_lovheim(fight).closest_affect()    # 'interest/excitement'
appraisal_to_lovheim(flight).closest_affect()   # 'fear/terror'
```

This is the agent loop end to end:

```
needs → appraisal → Hourglass → Lövheim cube → reaction
```

`float_emotion_to_lovheim_deltas()` gives the signed monoamine displacement from
baseline.

> **Deprecated:** `float_emotion_to_neuro_deltas()` returns **non-negative**
> deltas. The cube-derived replacement returns *signed* deltas — serotonin and
> adrenaline are not mutually exclusive in it — so it is **not** a drop-in
> substitution. Migrate deliberately, not mechanically.
