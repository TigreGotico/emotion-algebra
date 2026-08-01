# Building an agent that feels

This is the page for anyone making something that has to have an emotional life
over time, a game NPC, a conversational agent, a companion, a simulation.

The short version: **an emotion is a displacement, and a need is the force that
caused it.**

---

## Rest is not zero

Start here, because everything else depends on it.

"No emotion" is not a state anything is ever in. Core affect is always on, you
are never without valence and arousal, any more than you are without a body
temperature. So the coordinate origin is a *mathematical reference that nothing
occupies*.

What an organism actually falls toward is a **set point**, and it is **not** at
zero:

```python
from emotion_algebra import ORIGIN, SET_POINT, at_rest

at_rest(ORIGIN)      # False
at_rest(SET_POINT)   # True

SET_POINT.valence    # +0.15  -- mildly POSITIVE
SET_POINT.potency    # +0.15  -- mildly in control
SET_POINT.arousal    #  0.20  -- awake, not activated
```
Rest is *mildly pleasant*. That's the **positivity offset**, and it is why a
creature at rest **explores** instead of freezing. An agent that decays to zero
is an agent that becomes inert. An agent that decays to the set point is one that
gets curious again.

## The three timescales

| layer | what it is | timescale |
| --- | --- | --- |
| **emotion** | the current displacement | seconds to minutes |
| **mood** | a slow-moving estimate of where you've been | hours to days |
| **temperament** | the constitutional set point, the attractor | stable, per-agent |

Emotion relaxes toward mood. Mood drifts toward temperament. This is Mehrabian's
distinction, made mechanical.

## Emotion decays home, not to nothing

```python
from emotion_algebra import prototype, relax, dominant, drive_magnitude

terror = prototype("terror")

for dt in (0, 150, 300, 900, 3600):
    s = relax(terror, dt=dt, half_life=300)
    print(dt, dominant(s), round(drive_magnitude(s), 2))

# 0     terror        1.80
# 150   fear          1.27
# 300   apprehension  0.90
# 900   pensiveness   0.23
# 3600  acceptance    0.00
```
That is a **recovery trajectory**, not a switch. The agent passes down through
fear and apprehension on its way back from terror, gets quietly pensive, and
finally settles.

This is a theorem, not a heuristic: relaxation is a **contraction semigroup**, so
by the Banach fixed-point theorem the set point is its **unique** attractor and
*every* state converges to it, exponentially, from anywhere. See
[the laws](core-laws.md).

## Needs are displacements

Here is the pattern.

**A need deficit *is* a displacement from the set point**, and the emotion is the
felt signal of that displacement. So `drive()`, the vector pointing home, is
exactly the quantity a needs-driven agent minimises.

```python
from emotion_algebra import drive, drive_magnitude

drive(prototype("terror"))
# {'positivity':      +0.20,   # needs to come up
#  'negativity':      -0.51,   # needs to come down
#  'potency':         +1.05,   # control must be RESTORED  <- the loudest signal
#  'arousal':         -0.47,   # needs to settle
#  'unpredictability':-0.75}   # the world must become predictable again

drive_magnitude(prototype("terror"))   # 1.80 -- total unmet regulatory demand
```
Read that as a to-do list. A terrified agent's single loudest need is **to regain
control**, and an agent that knows this can *act on it*, which an agent holding
only "negative, aroused" cannot.

### The loop

```python
from emotion_algebra import (
    Appraisal, affect_from_text, relax, drive, dominant_tendency,
)
from emotion_algebra.appraisal import appraisal_to_affect

class Agent:
    def __init__(self, temperament=None):
        from emotion_algebra import BASELINE_TEMPERAMENT, SET_POINT
        self.temperament = temperament or BASELINE_TEMPERAMENT
        self.state = self.temperament.set_point

    def tick(self, dt):
        """Time passes. Feelings fade — toward rest, not toward nothing."""
        self.state = relax(
            self.state, dt=dt,
            half_life=self.temperament.resilience,
            toward=self.temperament.set_point,
        )

    def perceive(self, event: Appraisal, weight=0.5):
        """Something happened. Appraise it, and let it move you."""
        felt = appraisal_to_affect(event)
        self.state = self.state.blend(felt, weight)

    def act(self):
        """What am I getting ready to do?"""
        return dominant_tendency(self.state)

    def wants(self):
        """What is out of balance, and how badly?"""
        return drive(self.state, self.temperament.set_point)
```
That's the whole architecture:

```
event -> appraisal -> affect -> (tendency, drive) -> behaviour
                        ^                              |
                        +------- relax toward rest ----+
```
## Temperament: why two agents differ

Two agents in identical circumstances feel differently because they fall toward
different points. That is what "disposition" means, mechanically.

```python
from emotion_algebra import Temperament, AffectState

cheerful = Temperament(
    set_point=AffectState(positivity=0.35, negativity=0.02,
                          potency=0.30, arousal=0.25),
    resilience=180.0,          # bounces back fast
)

anxious = Temperament(
    set_point=AffectState(positivity=0.10, negativity=0.15,
                          potency=-0.20, arousal=0.35,
                          unpredictability=0.40),
    resilience=900.0,          # dwells on things
    negativity_bias=2.0,       # and bad news lands twice as hard
)
```
An anxious agent rests at *lower control* and *higher expected surprise*. Give it
the same bad news as the cheerful one and it will land harder, and take five
times as long to come back. You didn't script that, it falls out of where its
attractor is.

## Bad news lands harder than good news

```python
from emotion_algebra import perturb, SET_POINT

perturb(SET_POINT, {"positivity": 0.3}).positivity - SET_POINT.positivity  # 0.30
perturb(SET_POINT, {"negativity": 0.3}).negativity - SET_POINT.negativity  # 0.45
```
The **negativity bias**, equal pushes, unequal landings. One insult outweighs one
compliment, and your agent should behave the same way, because people do.

## Reading the agent's chemistry

If your architecture wants a neuromodulator layer (for drives, learning rates,
exploration/exploitation), it's right there:

```python
from emotion_algebra import NeuroState

NeuroState.from_affect(agent.state)
# NeuroState(dopamine=0.27, noradrenaline=0.56, serotonin=0.13, cortisol=1.0, ...)
```
Modulators map to **computational roles**, dopamine as reward-prediction error,
noradrenaline as unexpected uncertainty, serotonin as patience and time-horizon.
Those roles are directly useful to a learning agent. See
[neurochemistry](neurochemistry.md).

## Practical notes

**Pick a half-life that matches your tick rate.** `resilience` is in seconds. A
game running at 60 fps and a chat agent replying every 30 seconds want very
different numbers.

**Blend, don't add.** `state.blend(felt, weight)` is how an event moves an agent.
There is no `+`, and there is no `-`. Weight is "how much did this land", in
`[0, 1]`.

**Use the distribution, not just the name.** `label(state)` gives you the full
picture and `entropy(state)` tells you when the agent is genuinely conflicted, which is often the most interesting thing to render.

**Ambivalence is a feature.** An agent can be genuinely happy *and* sad at once
(`positivity` and `negativity` are separate channels). That's the bittersweet
ending, the reluctant victory, the goodbye, and a signed-valence model cannot
represent any of them.

---

## Where next

- Why the axes are what they are → **[the model](theory.md)**
- The formal guarantees (including Banach) → **[the laws](core-laws.md)**
- Turning events into emotions → **[appraisal](appraisal.md)**

---
[← Evidence](evidence.md) · [Home](index.md) · [State & timeline →](state.md)
