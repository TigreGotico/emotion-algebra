# Quickstart

Five minutes. By the end you'll be able to read an emotion out of text, work out
what someone will *do* about it, and give an agent a mood that persists.

```bash
pip install emotion-algebra
```
---

## 1. An emotion is a point, not a word

```python
from emotion_algebra import prototype

anger = prototype("anger")
fear  = prototype("fear")

anger.valence, anger.arousal, anger.potency    # -0.62, 0.62, +0.60
fear.valence,  fear.arousal,  fear.potency     # -0.52, 0.64, -0.60
```
Look carefully. Anger and fear have **nearly the same valence** and **nearly the
same arousal**. They differ almost entirely on `potency`, the sense of being able
to act.

This is why "negative sentiment, high arousal" is not a useful answer: it is true
of both, and they need opposite responses.

## 2. Blend them, and get something real

```python
from emotion_algebra import dominant

dominant(anger.blend(fear, 0.5))
# 'distress'
```
Unpleasant, activated, with the sense of control cancelled out. (Libraries that
treat anger and fear as opposites return **neutrality** here, they average to
zero. Blend maximal rage with maximal terror and get *calm*.)

`blend` is a **convex mixture**: `a.blend(b, w)` is `(1-w)·a + w·b`, and it can
never leave the space. There is no `+`, no `-`, and no unary minus. See
[the laws](core-laws.md) for why.

## 3. Names are distributions, not labels

Emotion names are regions, not coordinates, so the honest answer is a
distribution:

```python
from emotion_algebra import label, entropy

mixed = prototype("rage").blend(prototype("terror"), 0.5)

label(mixed, top_k=3)
# {'distress': 0.49, 'distraction': 0.29, 'apprehension': 0.22}

entropy(mixed)   # 4.25 bits -- this state sits BETWEEN names, and says so
dominant(mixed)  # 'distress'  -- the convenient answer, which throws the rest away
```
High entropy is not a failure to classify. It is the state genuinely being between
categories, which is a real thing for a state to do.

## 4. Work out what they'll do

Motivational direction tracks **potency**, not pleasantness:

```python
from emotion_algebra import dominant_tendency

dominant_tendency(prototype("anger"))    # 'antagonism'  -- move against it
dominant_tendency(prototype("fear"))     # 'avoidance'   -- move away
dominant_tendency(prototype("sadness"))  # 'withdrawal'  -- give up
dominant_tendency(prototype("disgust"))  # 'rejection'   -- push it away
dominant_tendency(prototype("joy"))      # 'affiliation' -- draw close
```
> **Trust the direction. Prefer the distribution.** "Anger approaches while being
> unpleasant" is reliable, it survives **100%** of perturbations of every guessed
> coefficient in the library. But *which* mode wins the argmax is not: `anger →
> antagonism` holds in only **55%**, `fear → avoidance` in **52%**, because
> approach and antagonism are neighbouring readings of the same drive. Use
> `action_readiness()` (the full distribution) when the answer matters, and treat
> `dominant_tendency()` as the convenience it is. See
> [the reliability report](evidence.md#reliability).


Anger is **unpleasant and approach-motivated**, which breaks every "negative =
avoid" model. Sadness and fear are both unpleasant and both low-control. What
separates them is *certainty*. Fear is an uncertain threat you can't handle, so
you run. Grief is a certain loss you can't handle, so you stop.

## 5. Read emotion out of text

```python
from emotion_algebra import affect_from_texts, dominant

angry, afraid = affect_from_texts([
    "This is the third time your app has lost my work. Fix it.",
    "I don't know if I'm doing this right and I'm scared I've broken something.",
])

angry.valence,  angry.potency    # -0.43, +0.16  -> 'disgust'
afraid.valence, afraid.potency   # -0.47, -0.43  -> 'apprehension'
```
Near-identical valence. **Opposite potency.** One will escalate. One will quietly
disappear.

> Held-out scores: valence **r=0.42**, arousal **r=0.35**. **Emoji tell you how
> someone feels. Punctuation tells you how loudly**, the probe reads emoji *and*
> typographic cues (`!!!`, CAPS, `sooo`), because emphasis is what carries
> arousal. See [text & emoji](text_emoji.md).

## 6. Go from an event to an emotion

Emotions aren't caused by events. They're caused by your **appraisal** of events.

```python
from emotion_algebra import Appraisal, dominant
from emotion_algebra.appraisal import appraisal_to_affect

# One obstructing event. Vary NOTHING but whether you can cope with it.
fight  = Appraisal(goal_relevance=0.9, goal_congruence=0.0, coping_potential=0.9)
flight = Appraisal(goal_relevance=0.9, goal_congruence=0.0, coping_potential=0.1)

dominant(appraisal_to_affect(fight))    # 'rage'
dominant(appraisal_to_affect(flight))   # 'fear'
```
Two of the core's axes *are* appraisal checks, `potency` **is** coping potential,
`unpredictability` **is** novelty, so most of that mapping is an identity, not a
fit. See [appraisal](appraisal.md).

## 7. Give an agent a mood

Emotion decays toward a **set point**, not toward zero. "No emotion" is not a
state anything is ever in.

```python
from emotion_algebra import SET_POINT, ORIGIN, at_rest, relax, drive

at_rest(ORIGIN)      # False  -- the coordinate origin is NOT rest
at_rest(SET_POINT)   # True   -- rest is mildly positive, calm, mildly in control

# Recovery is a trajectory:  terror -> fear -> apprehension -> acceptance
calmer = relax(prototype("terror"), dt=900, half_life=300)

drive(prototype("terror"))    # what must change to get home again
# {'negativity': -0.51, 'potency': +1.05, 'arousal': -0.47, ...}
```
`drive()` is the restoring force. **A need deficit *is* a displacement from the
set point.** That's the whole pattern for a needs-driven agent, see
[building an agent](agents.md).

## 8. Talk to your other tools

```python
from emotion_algebra import convert, explain_loss

convert(prototype("anger"), "core", "pad")   # (-0.62, 0.62, 0.60)
convert((-0.6, 0.8, 0.6), "pad", "core")     # -> AffectState

print(explain_loss("circumplex", "core"))
# potency and unpredictability. This is why the circumplex cannot tell
# anger from fear: they differ on potency, and it has no potency axis.
```
Every model converts to every other, and every conversion tells you what it
destroys. See [interop](interop.md).

## 9. Ask the library how much to trust itself

```python
from emotion_algebra import evidence

evidence.grade_of("circumplex")          # Grade.ESTABLISHED
evidence.grade_of("valence.bipolarity")  # Grade.CONTESTED    <- both readings shipped
evidence.grade_of("lovheim.cube")        # Grade.SPECULATIVE
evidence.grade_of("plutchik.antipodal")  # Grade.METAPHOR

print(evidence.report())                 # the whole table, with citations
```
See [evidence](evidence.md).

---

## Where next

- Building something that feels over time → **[agents](agents.md)**
- Want to know *why* these axes → **[the model](theory.md)**
- Want the formal algebra → **[the laws](core-laws.md)**
- Came here for `joy + trust == love` → **[the models](models.md)**

---
[Home](index.md) · [The model →](theory.md)
