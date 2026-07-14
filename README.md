# emotion-algebra

A Python library for representing, reasoning about, and computing with emotion —
built on the parts of affective science that actually replicate, and honest about
the parts that don't.

```python
from emotion_algebra import prototype, dominant

anger = prototype("anger")   # unpleasant, aroused, and IN CONTROL
fear  = prototype("fear")    # unpleasant, aroused, and NOT

dominant(anger.blend(fear, 0.5))
# 'distress'
```

Most emotion libraries would tell you that blend is **neutrality** — that anger
and fear, being "opposites", cancel out. They don't, and this one doesn't say
they do. Understanding why is most of what this library is about.

---

## Install

```bash
pip install emotion-algebra
```

| Extra | Adds |
| --- | --- |
| `emotion-algebra[viz]` | plots (matplotlib) |

Requires Python 3.10+.

---

## The 60-second version

An emotion is a point in a five-coordinate space:

```python
from emotion_algebra import AffectState

AffectState(
    positivity=0.8,        # how good it feels          [0, 1]
    negativity=0.6,        # how bad it feels           [0, 1]   (yes, both at once)
    potency=0.3,           # how in-control you feel    [-1, 1]
    arousal=0.7,           # how activated you are      [0, 1]
    unpredictability=0.2,  # how unexpected it is       [0, 1]
)
```

Two things about that are unusual, and both are deliberate.

**Positivity and negativity are separate.** People genuinely feel good and bad at
the same time — the classic case is graduation day. A single "valence" number
cannot represent that; two channels can.

```python
graduation = AffectState(positivity=0.8, negativity=0.6, arousal=0.7)
graduation.valence      # +0.2  -- "mildly happy", says a one-axis model
graduation.ambivalence  #  0.6  -- what the one-axis model destroyed
```

**Potency is the axis nobody ships.** Anger and fear are both unpleasant and both
highly aroused, so valence and arousal *cannot tell them apart*. What separates
them is your sense of control. Anger is what you feel when something is wrong and
you can act. Fear is what you feel when you can't.

That single axis is why this library exists, and it's the one thing you should
take away.

---

## What you can do with it

### Name a feeling — with honest uncertainty

Emotion names are **labels over regions**, not coordinates. So the answer to
"what emotion is this?" is a distribution, not a word.

```python
from emotion_algebra import prototype, label, dominant, entropy

mixed = prototype("rage").blend(prototype("terror"), 0.5)

label(mixed, top_k=3)
# {'distress': 0.66, 'distraction': 0.19, 'apprehension': 0.15}

entropy(mixed)   # 3.91 bits -- it sits BETWEEN names, and says so
dominant(mixed)  # 'distress'   (the convenient answer; throws away the rest)
```

### Work out what someone will *do*

Motivational direction tracks **potency**, not pleasantness. This is why "negative
= avoid" sentiment systems get anger wrong: anger is unpleasant *and*
approach-motivated.

```python
from emotion_algebra import dominant_tendency

dominant_tendency(prototype("anger"))    # 'antagonism'  -- move against it
dominant_tendency(prototype("fear"))     # 'avoidance'   -- move away
dominant_tendency(prototype("sadness"))  # 'withdrawal'  -- give up
dominant_tendency(prototype("joy"))      # 'affiliation' -- draw close
```

### Go from an event to an emotion

Emotions aren't triggered by events. They're triggered by your *appraisal* of
events — and the appraisal checks map almost one-to-one onto the core's axes.

```python
from emotion_algebra import Appraisal
from emotion_algebra.appraisal import appraisal_to_affect

# One obstructing event. Vary NOTHING but whether you can cope.
fight  = Appraisal(goal_relevance=0.9, goal_congruence=0.0, coping_potential=0.9)
flight = Appraisal(goal_relevance=0.9, goal_congruence=0.0, coping_potential=0.1)

dominant(appraisal_to_affect(fight))    # 'rage'
dominant(appraisal_to_affect(flight))   # 'fear'
```

Same event. Same unpleasantness. Coping decides whether you fight or flee.

### Build an agent that has moods

Emotion decays toward a **set point** — not toward zero. "No emotion" isn't a
state anything is ever in; resting is a mildly positive, calm, mildly-in-control
place. That's why a creature at rest *explores* rather than freezing.

```python
from emotion_algebra import SET_POINT, at_rest, relax, drive, ORIGIN

at_rest(ORIGIN)      # False -- the coordinate origin is NOT rest
at_rest(SET_POINT)   # True

# Recovery is a trajectory, not a switch:
#   terror -> fear -> apprehension -> acceptance
relax(prototype("terror"), dt=900, half_life=300)

drive(prototype("terror"))   # what must change to get home again
# {'negativity': -0.51, 'potency': +1.05, 'arousal': -0.47, ...}
```

`drive()` is the restoring force — the thing a needs-driven agent minimises. **A
need deficit *is* a displacement from the set point**, and the emotion is the felt
signal of it.

### Read the neurochemistry

```python
from emotion_algebra import NeuroState

# Same threat. Only the coping chemistry differs.
NeuroState(noradrenaline=.95, cortisol=.95, dopamine=.15).to_affect()   # potency -0.77 -> fear
NeuroState(noradrenaline=.90, dopamine=.85, testosterone=.9).to_affect() # potency +0.86 -> approach
```

Neuromodulators are mapped to **computational roles** — dopamine as
reward-prediction error, noradrenaline as unexpected uncertainty — not to emotion
names. That's what makes it testable.

### Convert to whatever your other tools speak

```python
from emotion_algebra import convert, fidelity, explain_loss

convert(prototype("anger"), "core", "pad")    # (-0.62, 0.62, 0.60)
convert((-0.6, 0.8, 0.6), "pad", "core")      # -> AffectState

fidelity("hourglass", "pad")                  # Fidelity.HEURISTIC
print(explain_loss("circumplex", "core"))
# potency and unpredictability. This is why the circumplex cannot tell
# anger from fear: they differ on potency, and it has no potency axis.
```

Every model converts to every other. **Every conversion tells you what it
destroys.** A conversion that loses information is fine; one that loses it
*silently* is not.

---

## The library grades its own claims

This is the feature we're proudest of, and we don't know of another library that
has it.

Affective science does not speak with one voice. Some of the models in here are
replicated across cultures and meta-analyses; one was published in a journal that
did not practise external peer review. A library that presents them all in the
same typeface is lying by omission.

So every construct carries a **grade** and its **citation**, in code:

```python
from emotion_algebra import evidence

evidence.grade_of("circumplex")          # Grade.ESTABLISHED
evidence.grade_of("grid")                # Grade.SUPPORTED
evidence.grade_of("valence.bipolarity")  # Grade.CONTESTED   <- both readings shipped
evidence.grade_of("lovheim.cube")        # Grade.SPECULATIVE
evidence.grade_of("plutchik.antipodal")  # Grade.METAPHOR

print(evidence.report())                 # the whole table, with citations
```

| Grade | Meaning |
| --- | --- |
| `ESTABLISHED` | Replicated, cross-cultural, meta-analytic. Build on it. |
| `SUPPORTED` | Good primary evidence, thin replication. |
| `CONTESTED` | A live scientific conflict — **both readings are implemented**. |
| `SPECULATIVE` | Proposed, plausible, never tested. Usable; not citable. |
| `METAPHOR` | A design device. Often the most convenient way to *talk* about emotion — which is why it ships. |

If you think a grade is wrong, the citation is right there to argue with.

---

## Is it still an algebra?

Yes — a better-specified one than it used to be.

The old claim was "vector space with negation": emotions add, scale, and every
emotion has an opposite. That claim is false, and it's what produced
`(rage + terror)/2 == calm`.

What's actually true:

- **`(S, blend)` is a barycentric algebra** — a convex space. By Stone's theorem
  its models are exactly the convex subsets of vector spaces, so no rigour is
  lost; we just say precisely *which* subset. Closure comes free: blending never
  needs clamping.
- **`(S, d)` is a metric space.**
- **`{relax_t}` is a contraction semigroup**, so by the Banach fixed-point theorem
  the set point is its **unique** attractor. Every state converges to rest,
  exponentially, from anywhere. That's a theorem, not a preference.

The supported operations are **mixture**, **intensification**, **decay**, and
**distance**. There is no `__neg__`, `__sub__`, `__add__` or `__mul__` — and tests
assert their absence.

**Sadness is not "minus joy."** It has its own pull — withdraw, seek help — and
that is not "negative approach."

Full laws, with the ones that *don't* hold: **[docs/core-laws.md](docs/core-laws.md)**

---

## Many models, honestly mapped

This library does not implement *an* emotion model. It implements **several** —
each faithfully, to its own author's specification — grades them by evidence, and
maps between them.

| Model | Author | Grade |
| --- | --- | --- |
| **Affect core** | Fontaine, Scherer, Roesch & Ellsworth (2007) | `SUPPORTED` |
| **Circumplex** | Russell (1980) | `ESTABLISHED` |
| **PAD / VAD** | Mehrabian & Russell (1974) | `SUPPORTED` |
| **Plutchik's wheel** | Plutchik (1980) | `METAPHOR` |
| **Hourglass** | Cambria, Livingstone & Hussain (2012) | `METAPHOR` |
| **Lövheim's cube** | Lövheim (2012) | `SPECULATIVE` |
| **Neuromodulators** | Schultz; Doya; Yu & Dayan | `SUPPORTED` |

So if you came for `joy + trust == love` and `-anger == fear`, they're here, and
they work:

```python
from emotion_algebra.emotions import get_emotion
from emotion_algebra.feelings import get_feeling_from_emotions

anger = get_emotion("anger")
anger + 1                                   # rage
-anger                                      # fear   (Plutchik's "opposite")
get_feeling_from_emotions("joy", "trust")   # 'love'
```

That arithmetic is correct **for Plutchik's model**. Plutchik's model is not
correct about people — anger and fear are neighbours, not opposites. Both things
are true, and the library tells you both: the wheel is graded `METAPHOR`, and the
core has no `__neg__`.

**Use the wheel to talk. Use the core to compute.** And convert between them with
a map that says what it costs.

**[docs/models.md](docs/models.md)** — the full catalogue.

## Documentation

**New here?** Read them in this order.

| | |
| --- | --- |
| **[Quickstart](docs/quickstart.md)** | Five minutes, hands-on. Start here. |
| **[The model](docs/theory.md)** | Why these axes, and not the others. The science. |
| **[Evidence](docs/evidence.md)** | Every construct, its grade, and its citation. |
| **[Building an agent](docs/agents.md)** | Set points, drives, moods, temperament. The pattern. |
| **[The laws](docs/core-laws.md)** | The algebra, formally — including what it refuses to do. |
| **[Appraisal](docs/appraisal.md)** | From events to emotions. |
| **[Neurochemistry](docs/neurochemistry.md)** | Neuromodulators as computational roles. |
| **[Interop](docs/interop.md)** | PAD, circumplex, and the conversion graph. |
| **[Text & emoji](docs/text_emoji.md)** | Getting emotion out of language. |
| **[The models](docs/models.md)** | Every model, its grade, and how they map. |
| **[CLI](docs/cli.md)** | `emotion-algebra` on the command line. |
| **[API reference](docs/api_reference.md)** | Every public symbol. |

---

## Validation

The claims above are tested, and the tests are in the repo — including the ones
that went against us.

| | |
| --- | --- |
| Anger/fear separable in **DeepMoji** (1.2B tweets, no theory of emotion) | **0.773** held out (baseline 0.598; permutation control 0.600) |
| …and the axis it uses to do it | **potency, r=+0.306** — 3× valence, arousal or unpredictability |
| **Lerner & Keltner (2001)** risk-judgement reproduction | anger patterns with *happiness*, not fear; fully mediated by control + certainty |
| Valence & arousal vs **human norms** (Warriner, 13,915 words) | taken directly from the data |
| Arousal *from text* | r=**0.35** — emoji carry valence, *punctuation* carries arousal |

Scripts in `scripts/validate/`. The honest limits of each are written into the
script that produces it.

---

## License

Apache-2.0.
