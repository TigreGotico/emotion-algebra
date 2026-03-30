# emotion_algebra — Scientific Specification

**Version**: 1.0
**Status**: Normative
**Purpose**: This document is the single source of truth for the behaviour of every class
and operation in `emotion_algebra`.  All implementation decisions must be traceable to a
clause here.  Where the spec departs from common intuition or prior code, the departure is
justified with a citation or explicit argument.

---

## 0. Guiding principles

1. **One model, one truth** — representation is Cambria's Hourglass.  Plutchik's Wheel
   supplies *names* and *dyad rules*, nothing else.  Wherever the two models conflict,
   Cambria's integer-axis definition wins for arithmetic; Plutchik's naming wins for
   human-readable labels.
2. **No undocumented heuristics** — every classification property (`type`, `valence`,
   `arousal`) must be derivable from the numeric representation by a stated formula.
3. **Emotion ≠ affect** — this library models discrete labelled emotion states, not
   continuous affect.  The well-known valence–arousal plane (Russell 1980) is used for
   classification only; it is not the primary representation.
4. **Operators are vector operations** — all arithmetic has the semantics of operations
   on a signed-integer vector, not on arbitrary psychological constructs.

---

## 1. The representational basis: Cambria's Hourglass

### 1.1 The four axes

Source: Cambria, E., Livingstone, A., & Hussain, A. (2012). *The Hourglass of Emotions.*
Cognitive Behavioural Systems, LNCS 7403, 144–157.

An emotion state is a point in a 4-dimensional signed-integer space **H**:

```
H = Z₄,   axis labels: (P, A, S, Ap)
                         ↑  ↑  ↑   ↑
                         │  │  │   └── Aptitude    (capability, competence)
                         │  │  └────── Sensitivity (reactivity, arousal)
                         │  └───────── Attention   (engagement, focus)
                         └──────────── Pleasantness (hedonic tone)
```

Each axis runs from −3 (intense negative pole) to +3 (intense positive pole).  The value
0 denotes the absence of activation on that axis.

**Taxonomy** (Cambria 2012, Table 1, extended with Plutchik 1980 intensity levels):

| Axis | +3 | +2 | +1 | −1 | −2 | −3 |
|------|----|----|----|----|----|----|
| Pleasantness | ecstasy | joy | serenity | pensiveness | sadness | grief |
| Attention | vigilance | anticipation | interest | distraction | surprise | amazement |
| Sensitivity | rage | anger | annoyance | apprehension | fear | terror |
| Aptitude | admiration | trust | acceptance | boredom | disgust | loathing |

**Justification for axis direction on Sensitivity**: Cambria places *anger* on the positive
pole of Sensitivity (high reactivity) and *fear* on the negative pole (suppressed
reactivity / defensive withdrawal).  This follows Öhman (1986) "Face the Beast and Fear
the Face": anger is an approach response (high arousal, outward activation); fear is an
avoidance response (defensive, inward).  The positive direction therefore means *reactive/
aggressive*, the negative direction means *inhibited/defensive*.

### 1.2 Named emotions as lattice points

A **named emotion** is a point in H with exactly one non-zero axis.  The six points on
each axis correspond to the six intensity levels: intense positive (±3), mild (±2), basic
(±1), with 0 = Neutrality.

A **composite state** is a point in H with two or more non-zero axes.

---

## 2. Core classes

### 2.1 `Emotion`

Represents a named lattice point with exactly one non-zero axis.

| Attribute | Type | Definition |
|-----------|------|------------|
| `_name` | `str` | Canonical lowercase label (e.g. `"anger"`) |
| `_dimension` | `EmotionalDimension` | The axis this emotion occupies |
| `intensity_offset` | `int` | Extra hyper-intensity beyond ±3 |

**`emotional_flow`** (property, returns `int`)
The signed integer value at this emotion's position on its axis.  Derived from taxonomy:
+3 = intense positive, +2 = mild positive, +1 = basic positive, −1 basic negative,
−2 mild negative, −3 intense negative, 0 = Neutrality.

**`valence`** (property, returns `int`)
The contribution of this emotion to hedonic tone — i.e. the value on the **Pleasantness**
axis only.

```
valence(e) = e.P   (the Pleasantness component of e's H-vector)
```

For single-axis emotions:
- `valence(joy) = +2`    (joy is on Pleasantness at +2)
- `valence(sadness) = −2`
- `valence(anger) = 0`   (anger is on Sensitivity, zero Pleasantness component)
- `valence(fear) = 0`

**Justification**: Hedonic valence = "how pleasant or unpleasant is this experience" (Lang
1995, Bradley & Lang 2000).  In Cambria's Hourglass, Pleasantness is explicitly the
hedonic axis.  Anger is not unpleasant *in the hedonic sense*; it is high-reactivity.
Assigning anger `valence = −1` (as the old code did via axis heuristics) conflates arousal
with valence — a well-documented error in affective computing (Posner et al. 2005).  The
correct assignment is `valence(anger) = 0`.

**`arousal`** (property, returns `int`)
The intensity of activation regardless of direction or axis:

```
arousal(e) = |e.emotional_flow|
```

Arousal maps to the "arousal" dimension in Russell's Circumplex (Russell 1980) and to
Cambria's combined Sensitivity + Attention activation.  Range: 0–3 (or higher for hyper).

**`__bool__`** — `True` iff `emotional_flow != 0` (presence, not valence).

**`type`** (property, returns `str`)
Classification using the valence–arousal plane following Russell (1980):

```
Quadrant I   (valence > 0, arousal > 1): "excited positive"   — e.g. ecstasy, joy
Quadrant II  (valence < 0, arousal > 1): "excited negative"   — e.g. grief, rage, terror
Quadrant III (valence < 0, arousal ≤ 1): "calm negative"      — e.g. sadness, pensiveness
Quadrant IV  (valence > 0, arousal ≤ 1): "calm positive"      — e.g. serenity, acceptance
Neutral      (valence == 0, arousal > 0): "activated neutral"  — e.g. anger, fear, anticipation, surprise
Inert        (arousal == 0):              "neutral"            — Neutrality only
```

**Justification**: Russell's Circumplex is the most empirically validated 2D emotion
classification (Russell 1980, Barrett & Russell 1999).  We project onto it by reading
`valence = Pleasantness component` and `arousal = |emotional_flow|`.  Emotions on the
Sensitivity and Attention axes have `valence = 0` and map to "activated neutral" — this
is accurate: anger and fear are high-arousal states without an inherent pleasantness
polarity *in this model*.  This is not a limitation but a feature: it forces callers to
use the correct axis for hedonic reasoning.

**`__add__(int)`** — intensity arithmetic: shift along the same axis by `n` steps.
**`__add__(Emotion, same axis)`** — sum flows; resolve to named emotion via `emotion_from_flow`.
**`__add__(Emotion, different axis)`** — returns `CompositeEmotion` spanning both axes.
**`__neg__`** — flip sign on the same axis (opposite pole): `−anger = fear`. Defined as
`emotion_from_flow(−emotional_flow)` on the same axis.
**`__mul__(Emotion, different axis)`** — alias for `__add__` across axes (creates
`CompositeEmotion`).  `*` is kept as a convenience operator; its semantics are identical
to cross-axis `+`.

**Justification for `−anger = fear`**: In Cambria's model, anger (+2 on Sensitivity) and
fear (−2 on Sensitivity) are symmetric opposite-pole emotions on the same axis.  The sign
flip is exact and well-defined.  This also matches Plutchik's wheel where anger and fear
are in the same "sensitivity" dyad as opposites.

### 2.2 `Neutrality`

The additive identity: `e + Neutrality() == e` for all `e`.
`emotional_flow = 0`, `valence = 0`, `arousal = 0`, `__bool__ = False`.

### 2.3 `EmotionalDimension`

One axis of the Hourglass.  Holds references to the six named `Emotion` objects at its
six non-zero positions.

**`valence`** (property, returns `int`)
The *inherent hedonic sign* of the positive pole of this dimension:

| Axis | `valence` | Justification |
|------|-----------|---------------|
| Pleasantness | +1 | Positive pole is explicitly pleasant by definition |
| Aptitude | +1 | Positive pole (admiration, trust) is socially desirable |
| Sensitivity | 0 | Positive pole is reactive (anger); negative pole is suppressed (fear). Neither is inherently pleasant — reactivity is orthogonal to hedonics |
| Attention | 0 | Positive pole is attentive (vigilance); negative pole is inattentive (surprise). Engagement is orthogonal to hedonics |

**Justification**: Assigning Sensitivity `valence = −1` (old code) was wrong because it
treats "reactive" as "unpleasant", conflating two independent psychological dimensions.
Posner et al. (2005, "The circumplex model of affect") note that arousal and valence are
orthogonal in all validated models.  Sensitivity encodes arousal/reactivity; its sign
(approach vs avoidance) is a separate dimension, not a valence modifier.

### 2.4 `CompositeEmotion`

A point in H with two or more non-zero axes.  Subclasses `Emotion` for operator
inheritance only; conceptually it is a *state*, not a single emotion.

**`components`**: list of single-axis `Emotion` objects, one per active axis.

**`emotional_flow`** (returns `int`)
Signed sum of all component flows:
```
emotional_flow = Σ e.emotional_flow  for e in components
```
This represents *net activation* — positive states dominate negative ones.  It is NOT
the same as the single-axis `emotional_flow` on a simple emotion and should not be
compared across the two types without awareness of this distinction.

**Justification**: The sum is the only linear, signed, dimensionless scalar that captures
net activation from a vector.  Euclidean norm (L2) was used previously but is always
non-negative, making the type property's signed branches permanently dead code.  The L1
sum preserves sign and is additive under composition.

**`valence`** (returns `int`)
The Pleasantness component of the composite state (summed across all components on
the Pleasantness axis):
```
valence = Σ e.emotional_flow  for e in components where e._dimension.axis == "pleasantness"
```
If no component is on the Pleasantness axis, `valence = 0`.

**`arousal`** (returns `int`)
Maximum absolute activation across all components:
```
arousal = max(|e.emotional_flow|  for e in components)
```
This measures *peak intensity* in the composite state.

**`type`** — same Circumplex classification as `Emotion.type`, using the composite's
`valence` and `arousal` values.

### 2.5 `Feeling`

A human-language label for a named cross-axis pattern (Plutchik's dyads and extensions).

A `Feeling` wraps a list of `Emotion` objects (any number).  It is NOT a subclass of
`Emotion`.  It represents the *psychological gestalt* of a multi-axis state when that
state is culturally named (e.g. "love" = joy + trust).

**`valence`**: sum of `e.valence` for all component emotions (Pleasantness contributions).
**`arousal`**: max `e.arousal` across components.
**`type`**: same Circumplex classification as above.

---

## 3. Operator algebra

### 3.1 Single-axis intensity arithmetic

```
emotion + n  →  emotion_from_flow(flow + n)    (n is int)
emotion - n  →  emotion_from_flow(flow - n)
emotion / n  →  emotion_from_flow(flow / n)
emotion // n →  emotion_from_flow(flow // n)
```

Out-of-range flows (|flow| > 3) generate hyper-emotions with `intensity_offset > 0`.

### 3.2 Opposite (negation)

```
-emotion  →  emotion_from_flow(-emotional_flow)  on the same axis
```

Examples: `−anger = fear`, `−joy = sadness`, `−admiration = loathing`.

### 3.3 Cross-axis composition (addition / multiplication)

```
emotion₁ + emotion₂  (different axes)  →  CompositeEmotion([emotion₁, emotion₂])
emotion₁ * emotion₂  (different axes)  →  same as above  (alias)
```

If the resulting `CompositeEmotion` matches a named dyad in `COMPOSITE_EMOTIONS_NAMES`,
it is returned with that name populated.

### 3.4 Same-axis addition

```
emotion₁ + emotion₂  (same axis)  →  emotion_from_flow(flow₁ + flow₂)
```

### 3.5 Comparison

All comparison operators (`<`, `<=`, `>`, `>=`, `==`, `!=`) compare `emotional_flow`
values.  Two emotions from different axes can be compared numerically (their flows are
comparable integers) but the result has no psychological meaning unless they share an
axis — callers should be aware of this.

### 3.6 Shifting operators (`<<`, `>>`)

```
emotion << n  →  emotion_from_flow(flow - n)   (downgrade intensity)
emotion >> n  →  emotion_from_flow(flow + n)   (upgrade intensity)
```

For emotion operands (not int), these compute the flow *difference* between the two
emotions, requiring same-axis operands.

---

## 4. Named composites and feelings

### 4.1 `CompositeEmotion` names (Plutchik's secondary dyads)

Source: Plutchik (1980), "A general psychoevolutionary theory of emotion."

Named composites arise from *adjacent* emotions on Plutchik's wheel.  The 16 named
composites in `COMPOSITE_EMOTIONS_NAMES` are the standard Plutchik secondary and
tertiary dyads.

### 4.2 `Feeling` names (Plutchik's primary dyads + extensions)

`FEELING_NAMES` maps pairs of basic single-axis emotions to named feelings.  These are
Plutchik's primary dyads (e.g. joy + trust = love) plus additional common combinations.

### 4.3 Resolution order

When `emotion₁ + emotion₂` (cross-axis) is evaluated:
1. A `CompositeEmotion` with both as components is created.
2. If the pair appears in `COMPOSITE_EMOTIONS_NAMES`, the composite is named.
3. The result type is always `CompositeEmotion`.
4. A `Feeling` is only created explicitly (`Feeling("love")`) or via `get_feeling()`.

**Justification for the Feeling/CompositeEmotion separation**: `Feeling` is a cultural
label (what humans call the experience).  `CompositeEmotion` is the mathematical
representation (the algebraic state).  Conflating them — returning a `Feeling` from `+`
— obscures which layer is doing the work and makes the operator algebra non-uniform.
Under this spec, `+` always returns an `Emotion` or `CompositeEmotion`, never a `Feeling`
directly.

---

## 5. POSITIVE_EMOTIONS and NEGATIVE_EMOTIONS

```python
POSITIVE_EMOTIONS = [e for e in EMOTIONS if e.valence > 0]
NEGATIVE_EMOTIONS = [e for e in EMOTIONS if e.valence < 0]
```

Under this spec, only emotions on the **Pleasantness** axis have non-zero valence.  All
Sensitivity- and Attention-axis emotions have `valence = 0` and do not appear in either
list.  This is correct: whether anger is "positive" or "negative" depends on framing
(approach vs affect), and the library makes no such claim — only Pleasantness and Aptitude
axis emotions carry hedonic information.

Aptitude-axis emotions (admiration, trust, acceptance, boredom, disgust, loathing): their
valence is the aptitude flow, because Aptitude is a socially hedonic axis (capable vs
incapable = desirable vs aversive).  More precisely:

```
valence(e) = P_component + Ap_component * 0.5
```

However, this half-weight for Aptitude is itself a simplification (not in Cambria's
original scoring).  To keep the implementation honest, we adopt:

```
valence(e) = P_component   (Pleasantness axis only)
```

and document that Aptitude emotions are not classified as positive or negative.

---

## 6. Properties NOT derived from the model

The following exist in the codebase as convenience features, are NOT derived from Cambria
or Plutchik, and are marked accordingly:

- **`kind`** — maps emotion names to hand-curated psychological categories
  (event-related, social, etc.).  Source: `EMOTION_CONTRASTS` reference map.
  No formula; lookup only.  Not a scientific derivation.
- **Lexicon** (`lexicons.py`) — a word→emotion CSV mapping from external annotation.
  Not part of the algebraic model.

---

## 7. What this library does NOT model

| Feature | Reason not included |
|---------|----------------------|
| PAD (Pleasure–Arousal–Dominance) | 3-axis continuous model; incompatible with Hourglass integer lattice without redesign |
| Cognitive appraisal (Lazarus 1991) | Requires structured world-state; out of scope |
| Temporal dynamics (emotion decay/buildup) | No time dimension in current model |
| Mixed valence / ambivalence | e.g. bittersweet; requires probability distributions over states |
| Individual / cultural variation | Model is universal by assumption |
