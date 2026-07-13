> **Plutchik / Hourglass model.** This page documents the laws of the
> *Hourglass* algebra — one of the [models](models.md) this library
> implements. For the affect core's laws, see [core-laws](core-laws.md).

# The laws

emotion-algebra calls itself an algebra, so the laws it obeys are written down
here and **machine-checked** in `test/test_laws.py` with
[hypothesis](https://hypothesis.readthedocs.io/) (deterministic in CI:
`derandomize=True`, 200 examples per law).

Where a law has a boundary, the boundary is part of the law.

## Identity

`Neutrality` is the additive identity.

```
e + neutral == e
neutral + e == e        for every EmotionBase
```

`Neutrality + e` returns a **copy** of `e`, not `e` itself. The named emotions
are shared singletons in the `EMOTIONS` registry, so returning the operand would
expose the registry to mutation through it.

## Negation

Negation is an involution, and opposites are exact negatives.

```
-(-e) == e
e + (-e) == neutral
terror == -rage
amazement == -vigilance
```

That last pair is what makes the Lövheim bridge non-injective — see
[lovheim.md](lovheim.md).

## Addition

Same-axis addition is commutative, and associative **as long as no intermediate
saturates**.

```
a + b == b + a
(a + b) + c == a + (b + c)      only when no partial sum saturates
```

The saturation caveat is load-bearing. Flow is clamped to the hyper range, so
`anger + anger` already saturates, and once an *intermediate* result clamps, the
information needed to re-associate is gone. The property test therefore scopes its
`assume()` to every pairwise partial sum, not only the final one.

Cross-axis `+` and `*` build a `CompositeEmotion`; both are symmetric, and
`CompositeEmotion` equality is component-set equality (order-independent).

## Subtraction

```
a - b == -(b - a)
```

## Intensity shifts

`<<` and `>>` are inverse shifts along an axis:

```
(a >> n) << n == a      within range
```

## Metric

`emotion_distance` is a true metric on the vector representation:

```
d(a, b) >= 0
d(a, b) == 0   iff  a and b have the same vector
d(a, b) == d(b, a)
d(a, c) <= d(a, b) + d(b, c)
```

## Similarity

`emotion_similarity(a, b)` is in `[0, 1]`, with `1.0` for identical vectors.
Two metrics are offered — `"distance"` (normalized inverse distance, the default)
and `"cosine"` — and an unknown metric raises `ValueError` rather than silently
picking one. A zero vector has no direction, so cosine similarity against it is
defined as `0.5` rather than `NaN`.

**An opposite is not always the least similar thing.** Negation mirrors at the
same intensity tier, so `loathing` (−3) is *further* from `acceptance` (+1) than
`acceptance`'s own opposite `boredom` (−1) is. The law that holds is: the
**extreme of the opposite pole** is least similar *on that axis*.

## Polarity vs valence

These are deliberately different quantities:

* **`valence`** is the Pleasantness axis alone — the hedonic axis.
* **`polarity`** is Cambria's published four-axis formula:

  ```
  polarity = (Pleasantness + |Attention| − |Sensitivity| + Aptitude) / 3
  ```

  on axes normalized to `[−1, 1]`, clamped to `[−1, 1]`.

Sign-flip under negation therefore holds for `polarity` **only on the hedonic
axes**. Because Attention enters as a *magnitude*, both of its poles
(anticipation and surprise) score positive — negating a purely attention-axis
emotion leaves polarity unchanged. That is a consequence of the published
formula, and it is locked by `test_both_attention_poles_score_positive`.

`_circumplex_type` classifies on `polarity`, not `valence`.

## Conversion coherence

Round-tripping through the float space preserves identity:

```
closest_emotion(FloatEmotion.from_emotion(e)) == e
```

for all 24 named emotions and all 40 feelings.

## Lövheim

```
corner.closest_affect()                  names that corner            (bijective)
corner.affect_blend()[its own name]      == 1.0
sum(affect_blend().values())             == 1.0
aptitude(F(p))                           <= 0     for every p in the cube
F is NOT injective                       (a theorem — see lovheim.md)
```

Raising serotonin never moves the result toward a low-serotonin corner
(monotonicity). No symmetry beyond what the paper states is assumed or tested.

## PAD

`from_pad` is an **exact right inverse of `to_pad` on pleasure and dominance** —
machine precision, not tolerance — wherever the target is reachable. Arousal is
the exception by construction: `to_pad` reads it as a *max* over the axes, and a
max is not invertible, so `from_pad` honours it as an activation floor. Roughly
40% of the raw PAD cube is unreachable in the bounded Hourglass space (e.g.
"maximally pleasant, zero arousal"); those targets saturate rather than raise.

## Serialization

Every type round-trips through `to_dict`/`from_dict` and `to_json`/`from_json`
with its type and vector intact, and stateful types (`EmotionalState`,
`EmotionTimeline`) additionally preserve mood and history. Payloads are stamped
with `schema: 1`; an unknown schema version raises `UnsupportedSchemaError` (a
`ValueError` subclass, so callers that only catch `ValueError` stay safe).
