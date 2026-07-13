# Neurochemistry

The library has a neuromodulator layer. It maps neurotransmitters onto
**computational roles** — not onto emotion names.

That distinction is the whole design, and it is the difference between a model
you can test and one you can't.

```python
from emotion_algebra import NeuroState

# Same threat. Only the coping chemistry differs.
NeuroState(noradrenaline=.95, cortisol=.95, dopamine=.15).to_affect()
# potency -0.77  ->  fear

NeuroState(noradrenaline=.90, dopamine=.85, testosterone=.9).to_affect()
# potency +0.86  ->  approach
```

## Why not Lövheim's cube

Lövheim (2012) maps three monoamines onto the eight Tomkins affects, one per cube
corner. It's an appealing picture and it turns up constantly in ML papers.

It has **never been empirically tested.** No study has measured monoamine levels
against discrete emotion reports in the same subjects. It appeared in *Medical
Hypotheses*, which by explicit editorial policy did not practise external peer
review — an Elsevier panel found the journal was publishing "baseless,
speculative, non-testable" material and removed the editor in 2010.

For scale: the far narrower serotonin-*depression* hypothesis, studied enormously
more, did not survive umbrella review (Moncrieff et al. 2022). A
3-monoamine → 8-discrete-emotion mapping is a categorically stronger claim resting
on far less.

**But the error is not "neurochemistry."** The error is mapping neuromodulators
directly onto **discrete emotion labels**, which is a claim nobody knows how to
test. What would even count as evidence?

## The roles

Map them onto *computational roles* instead, and you are standing on some of the
most replicated work in systems neuroscience.

| modulator | computational role | drives |
| --- | --- | --- |
| **dopamine** | reward-prediction error; incentive salience; vigor (Schultz, Dayan & Montague 1997) | potency, positivity |
| **noradrenaline** | arousal, adaptive gain, and **unexpected** uncertainty (Aston-Jones & Cohen 2005; Yu & Dayan 2005) | arousal, unpredictability |
| **acetylcholine** | **expected** uncertainty; attentional precision (Yu & Dayan 2005) | unpredictability |
| **serotonin** | patience, behavioural inhibition, aversive weighting, effective time horizon (Doya 2002) | negativity (−), potency |
| **cortisol** | sustained threat under low coping | negativity, potency (−) |
| **opioids / oxytocin** | hedonic *liking*; affiliation | positivity |
| **testosterone** | dominance and status seeking | potency |

Doya (2002), *Metalearning and neuromodulation*, is the canonical framing:
dopamine ≈ TD error, serotonin ≈ discount factor, noradrenaline ≈ inverse
temperature, acetylcholine ≈ learning rate.

Two things worth internalising:

**Dopamine is *wanting*, not *liking*.** Berridge & Robinson's distinction.
Dopamine drives pursuit; the hedonic *pleasure* lives with the opioids. Conflating
them is the classic error, and a lot of pop neuroscience is built on it.

**Noradrenaline and acetylcholine split uncertainty in two.** Noradrenaline
signals *unexpected* uncertainty — the world just broke its own rules.
Acetylcholine signals *expected* uncertainty — known, estimable noise. That is a
genuinely useful distinction for an agent, and it maps cleanly onto the core's
`unpredictability` axis.

## The payoff

Those roles land **directly on the core's axes** — because both are describing the
same functional dimensions. Noradrenaline drives arousal and unpredictability.
Dopamine drives potency and approach. Serotonin drives inhibition and aversive
weight.

**The neurochemistry and the empirical affect space agree**, which is exactly what
a mapping to emotion *names* could never have shown you.

```python
from emotion_algebra import NeuroState, dominant

cases = {
  "threat, cannot cope":  NeuroState(noradrenaline=0.95, cortisol=0.95, dopamine=0.15),
  "threat, CAN cope":     NeuroState(noradrenaline=0.90, dopamine=0.85, testosterone=0.9),
  "reward":               NeuroState(dopamine=0.90, opioids=0.90),
  "novelty, safe":        NeuroState(noradrenaline=0.80, acetylcholine=0.90),
  "depleted":             NeuroState(dopamine=0.15, opioids=0.10, serotonin=0.20),
}
for name, ns in cases.items():
    a = ns.to_affect()
    print(f"{name:22} potency={a.potency:+.2f}  ->  {dominant(a)}")

# threat, cannot cope    potency=-0.77  ->  fear
# threat, CAN cope       potency=+0.86  ->  ecstasy
# reward                 potency=+0.59  ->  joy
# novelty, safe          potency=+0.09  ->  surprise
# depleted               potency=-0.39  ->  sadness
```

## Both directions

```python
from emotion_algebra import NeuroState, prototype

NeuroState.from_affect(prototype("terror"))
# NeuroState(dopamine=0.14, noradrenaline=0.62, serotonin=0.31,
#            acetylcholine=1.0, cortisol=0.82, ...)
```

The inverse is **lossy, and says so**: seven modulators do not determine five axes,
nor the reverse. It returns the **minimum-norm** chemistry consistent with the
affect — the least dramatic explanation of the state, which is the honest default.

A round trip through `neuro` therefore *flattens unusual chemistry toward the
ordinary*. The forward direction (chemistry → affect) is the trustworthy one; it
rests on computational roles, not on emotion labels.

## Lövheim is still reachable

His three monoamines are a **subset** of ours, so his cube remains available as a
coordinate drop — for anyone with a downstream consumer that expects it:

```python
from emotion_algebra import convert, prototype, fidelity

convert(prototype("rage"), "core", "lovheim")   # -> LovheimPoint
fidelity("core", "lovheim")                     # Fidelity.HEURISTIC
```

It is graded `SPECULATIVE`. Convert to it if you need it. **Do not cite it.**

## Baseline is not zero

Every modulator sits in `[0, 1]` **relative to the individual's own baseline**, and
`0.5` is that baseline — not an absolute concentration.

```python
from emotion_algebra import NeuroState, at_rest

NeuroState().is_baseline               # True
at_rest(NeuroState().to_affect())      # True -- resting chemistry IS resting affect
```

Which is the same point the rest of the library keeps making: **rest is a place,
and it is not the origin.**
