# The model

Most emotion libraries pick a theory and implement it. This one implements
several, **grades them by evidence**, and routes every conversion through the one
that actually replicates.

## Why not just use Plutchik and the Hourglass?

Because they do not survive an evidence audit, and their defects are load-bearing.

Ask the Hourglass axes what the average of maximal rage and maximal terror is:

```
(rage + terror) / 2  ==  [0, 0, 0, 0]   ->  neutrality
```

Blend the two most intense negative states a person can have, and get **calm**.

That is not a rounding error. It follows necessarily from putting anger and fear
at opposite ends of one signed axis — and they are **not opposites**. Four
independent research programmes (Smith & Ellsworth 1985; Roseman 1996; Scherer's
SECs; Lerner & Keltner 2001, with *causal mediation*) find that anger and fear
are both unpleasant, both highly aroused, and separated by **control / coping
potential**. They are neighbours.

Two more symptoms of the same mistake:

- The Hourglass's own polarity formula is `(P + |At| − |S| + Ap) / 3`. Those
  absolute values are an **admission**: both poles of Sensitivity are
  unpleasant, so the axis is not hedonically bipolar — which contradicts the
  bipolar geometry the model otherwise assumes.
- PAD's Dominance had to be *reconstructed by fitted regression*, because
  collapsing anger and fear onto one axis had thrown the dimension away.

And the audit itself:

| Model | Grade | Why |
|---|---|---|
| Russell's circumplex | `ESTABLISHED` | Best-replicated structure in affective science; cross-cultural, cross-modal |
| Control separates anger/fear | `ESTABLISHED` | Four programmes converge; Lerner & Keltner show causal mediation |
| No discrete-emotion signatures | `ESTABLISHED` | Lindquist (2012) + Siegel (2018): two meta-analyses, two modalities, same null |
| Fontaine's GRID (4-D) | `SUPPORTED` | Derived from 144 componential features across cultures |
| Valence bipolarity | `CONTESTED` | Happy and sad **co-activate** (Larsen 2001). Unresolved — both readings shipped |
| Lövheim's cube | `SPECULATIVE` | **Never tested.** Venue did not practise external peer review |
| Plutchik's antipodal wheel | `METAPHOR` | Smith & Schneider (2009), >2,000 tests: *"no empirical support"* |
| Cambria's Hourglass | `METAPHOR` | Self-described derivative of the wheel; built for sentiment engineering |

Ask the library yourself:

```python
from emotion_algebra import evidence
print(evidence.report())
```

## The core

Fontaine, Scherer, Roesch & Ellsworth (2007), *The world of emotions is not
two-dimensional*, derived four dimensions empirically and cross-culturally.
That is the core — with one refinement.

```python
AffectState(
    positivity,        # [0, 1]   separable hedonic channels...
    negativity,        # [0, 1]   ...so bittersweet is representable
    potency,           # [-1, 1]  control/coping — THE anger/fear axis
    arousal,           # [0, 1]
    unpredictability,  # [0, 1]
)
```

`valence = positivity − negativity` is available as an exact projection, and
`ambivalence = min(positivity, negativity)` is what a signed axis destroys.

Now the crux case comes out right:

```python
>>> anger = prototype("anger")   # negative, aroused, potency +0.60
>>> fear  = prototype("fear")    # negative, aroused, potency -0.60
>>> dominant(anger.blend(fear, 0.5))
'distress'                       # not 'neutrality'
```

## It is not a vector space

There is no `__neg__`, no `__sub__`, no `__mul__`.

Sadness is **not** "minus joy". It has its own action readiness — withdraw,
help-seek — which is not "negative approach". The supported operations are
**convex mixture**, **intensification** (a positive scalar), **decay**
(contraction toward the set point) and **distance**: a convex cone with a metric.

It is not polar or conical either. Plutchik's cone needs a constant-radius circle
to separate quality (angle) from intensity (radius) — and the affect circumplex
is an **ellipse**, not a circle (Stanisławski, Cieciuch & Strus 2021). The cone
dies with the wheel.

## Neutrality is not the origin

**"Lack of emotion" is not a state.** Core affect is always on (Barrett &
Bliss-Moreau 2009) — you are never without valence and arousal, any more than you
are without a body temperature. The coordinate origin is a *mathematical
reference that nothing occupies.*

What an organism falls toward is a **set point**, and it is not at zero:

- **positivity offset** — at rest, positivity slightly exceeds negativity. This
  is why an organism at rest *explores* instead of freezing.
- **negativity bias** — negativity, once engaged, rises more steeply.
  "Bad is stronger than good" (Baumeister et al. 2001).

```python
>>> at_rest(ORIGIN)      # the origin is NOT rest
False
>>> at_rest(SET_POINT)   # this is
True
```

Which gives the dynamical system an agent actually needs:

| layer | what it is | timescale |
|---|---|---|
| **emotion** | the current displacement | seconds–minutes |
| **mood** | a slow-moving estimate | hours–days |
| **temperament** | the constitutional set point (the attractor) | stable, per-agent |

**A need deficit is a displacement from the set point**, and the emotion is the
felt signal of it. `drive(state)` returns the restoring vector — the thing a
needs-driven agent minimises:

```python
>>> relax(prototype("terror"), dt=..., half_life=300)
terror -> fear -> apprehension -> acceptance     # a recovery trajectory
```

## Names are a readout, not a basis

Lindquist (2012) and Siegel (2018) find no consistent signature for discrete
emotions. Cowen & Keltner (2017) find categories bridged by **continuous
gradients**. So the honest answer to "what emotion is this?" is a *distribution*:

```python
>>> label(mid, top_k=3)
{'distress': 0.81, 'fear': 0.10, 'loathing': 0.10}
>>> entropy(mid)
2.98   # it sits between names — and says so
```

`dominant()` is the argmax convenience. It throws away the runners-up, which is
where the gradient lives.

## Action readiness tracks potency, not valence

**Anger is negative in valence but approach-motivated** (Carver & Harmon-Jones
2009). That single fact refutes every "negative = avoid" sentiment model.

| | valence | potency | arousal | → |
|---|---|---|---|---|
| anger | − | **+** | high | move *against* |
| fear | − | **−** | high | move *away* |
| sadness | − | − | **low** | *withdraw* |

Valence tells you whether it is good. **Potency tells you what you will do about
it.** Arousal tells you whether you will do it now.

## Neurochemistry, done honestly

Lövheim's error was not "neurochemistry" — it was mapping three monoamines onto
eight **emotion names**, a claim nobody knows how to test. Map them to
**computational roles** instead and you stand on replicated ground:

| modulator | role | drives |
|---|---|---|
| dopamine | reward-prediction error, incentive salience, vigor (Schultz 1997) | potency, positivity |
| noradrenaline | arousal; **unexpected** uncertainty (Yu & Dayan 2005) | arousal, unpredictability |
| acetylcholine | **expected** uncertainty; attentional precision | unpredictability |
| serotonin | patience, inhibition, time-horizon (Doya 2002) | negativity(−), inhibition |
| cortisol | sustained threat under low coping | negativity, potency(−) |
| opioids | hedonic *liking* — the pleasure dopamine is not | positivity |
| testosterone | dominance, status seeking | potency |

Those roles land **directly on the core axes** — because both are describing the
same functional dimensions. A mapping to emotion *names* could never show that.

The payoff, in one test:

```python
# same threat. only the coping chemistry differs.
NeuroState(noradrenaline=.95, cortisol=.95, dopamine=.15).to_affect()   # potency -0.77 -> fear
NeuroState(noradrenaline=.90, dopamine=.85, testosterone=.9).to_affect() # potency +0.86 -> approach
```

Lövheim's three monoamines are a **subset** of these, so his cube stays reachable
as a coordinate drop. Nothing downstream breaks.

## Every model converts to every other

The theories are not mutually consistent, so they are not merged. They are
**registered as views on the core**, and conversion is routed through it:

```
   Hourglass ─┐                    ┌─ PAD
   Plutchik ──┼──►  AFFECT CORE  ◄─┼─ circumplex
   Lövheim ───┘                    └─ NeuroState
```

N models, 2N maps, not N². Every pair is reachable — **always** — and every map
says what it costs:

```python
>>> fidelity("hourglass", "pad")
<Fidelity.HEURISTIC>
>>> print(explain_loss("circumplex", "core"))
potency and unpredictability. This is why the circumplex cannot tell anger
from fear: they differ on potency, and it has no potency axis.
```

**PAD is now exact on its own axes.** Pleasure *is* valence, Arousal *is* arousal,
Dominance *is* potency. The fitted weights, the closed-form inverse, and the ~40%
unreachable region are all gone — they were only ever compensating for an axis the
Hourglass had destroyed.

A conversion that loses information is fine. A conversion that loses it
**silently** is not.
