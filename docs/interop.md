# Interop: PAD / VAD

Most of the affective-computing field speaks **PAD** (Pleasure-Arousal-Dominance;
Mehrabian & Russell 1974, Mehrabian 1996), also called **VAD** when Pleasure is
named Valence. `emotion_algebra.pad` maps between PAD and the library's 4-axis
Hourglass space so external resources — the NRC-VAD lexicon, PAD-annotated
corpora — plug straight in.

```python
from emotion_algebra import to_pad, from_pad, get_emotion

to_pad(get_emotion("rage"))     # PAD(pleasure=-0.33, arousal=1.0, dominance=0.6)
to_pad(get_emotion("terror"))   # PAD(pleasure=-0.33, arousal=1.0, dominance=-0.6)

from_pad(pleasure=0.6, arousal=0.7, dominance=0.4)   # -> FloatEmotion
```

`PAD` is a `NamedTuple`; `.valence` is an alias for `.pleasure`.

## Forward: Hourglass → PAD

| PAD component | Derived from |
|---|---|
| **Pleasure** | `polarity` — Cambria's four-axis score, already in `[−1, 1]` |
| **Arousal** | peak axis magnitude, normalized to `[0, 1]` |
| **Dominance** | `0.60·Sensitivity + 0.30·Aptitude + 0.10·Attention` |

The dominance weights are the only calibrated numbers here, and they are chosen
around the single most-replicated fact in the PAD literature: **anger is dominant,
fear is submissive**. Those two emotions are near-identical in pleasure and
arousal and differ almost entirely on dominance — it is *why* PAD needs a third
axis at all. In the Hourglass, anger and fear are exactly the two poles of
Sensitivity, so Sensitivity carries dominance and is the only signed term.
Aptitude adds felt competence (trust reads as potent, disgust as impotent);
Attention contributes weakly, since engagement is mildly empowering while its
negative pole (surprise) is a loss of control.

## Inverse: PAD → Hourglass

Three numbers cannot pin down four axes, so the lift is under-determined and is
closed by one stated choice: **the hedonic load splits evenly between Pleasantness
and Aptitude.** That is not arbitrary — Cambria's polarity formula weights the two
identically, so it gives no reason to prefer either.

With that choice the system is square and solves in closed form, and the result is
an **exact right inverse on pleasure and dominance**:

```python
p, a, d = 0.6, 0.7, 0.4
back = to_pad(from_pad(p, a, d))
back.pleasure == p     # to machine precision
back.dominance == d    # to machine precision
```

Two honest caveats, both tested:

* **Arousal is a floor, not an equality.** `to_pad` reads arousal as a *max* over
  the axes, and a max destroys the information needed to undo it. Where a
  stronger hedonic or dominance demand pushes an axis above the requested
  arousal, the returned arousal is higher than asked.
* **Not every PAD triple is reachable.** The Hourglass axes are bounded, so
  "maximally pleasant, zero arousal" has no pre-image. Those targets saturate at
  the cube face and round-trip only approximately — roughly 40% of the raw PAD
  cube. That is a property of the two spaces, not a defect of the map.

## Distance in PAD space

`pad_distance(a, b)` compares two emotions **in PAD**, which is what you want when
benchmarking against PAD-native resources:

```python
from emotion_algebra import pad_distance, get_emotion

pad_distance(get_emotion("rage"), get_emotion("terror"))   # large — dominance separates them
```

In Hourglass space anger and fear sit on one axis and are close; in PAD they are
far apart. Neither is wrong — they are different questions.
