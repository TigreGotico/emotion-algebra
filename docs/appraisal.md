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

## Appraisal is the core, one layer up

Two of the core's axes **are** appraisal checks. This is not a fit; it is an
identity — and it is the strongest argument for these axes over any others.

| core axis | appraisal check |
| --- | --- |
| `potency` | **coping potential**, centred and signed |
| `unpredictability` | **novelty** |
| `positivity`/`negativity` | goal congruence, tempered by intrinsic pleasantness |
| `arousal` | goal relevance — how much is at stake |

```python
from emotion_algebra import Appraisal, dominant
from emotion_algebra.appraisal import appraisal_to_affect

# One obstructing event. Vary NOTHING but whether you can cope with it.
fight  = Appraisal(goal_relevance=0.9, goal_congruence=0.0, coping_potential=0.9)
flight = Appraisal(goal_relevance=0.9, goal_congruence=0.0, coping_potential=0.1)

dominant(appraisal_to_affect(fight))    # 'rage'
dominant(appraisal_to_affect(flight))   # 'fear'
```

The dimensions of felt emotion turn out to be the dimensions of appraisal —
because appraisal is what constructs the feeling.

## Straight through to neurochemistry

```python
from emotion_algebra import NeuroState

NeuroState.from_affect(appraisal_to_affect(flight))
# high cortisol, low dopamine -- the chemistry of not being able to cope
```

The agent loop, end to end:

```
needs -> appraisal -> affect core -> (tendency, drive) -> behaviour
                          |
                          +-> neuromodulators (learning rates, exploration)
```

See [neurochemistry](neurochemistry.md) and [building an agent](agents.md).

## The other mappings

`appraisal_to_emotion` (discrete, categorical) and `appraisal_to_float_emotion`
(continuous, Hourglass axes) map the same appraisal onto the Plutchik and
Hourglass [models](models.md). `appraisal_to_affect` maps it onto the core.

`float_emotion_to_neuro_deltas()` is **deprecated** — it returns non-negative
deltas, while its replacement returns *signed* ones, so it is not a drop-in
substitution. Migrate deliberately.
