# Interop

Every model converts to every other, and **every conversion tells you what it
destroys**.

```python
from emotion_algebra import convert, fidelity, explain_loss, conversion_views

conversion_views()
# ['core', 'circumplex', 'hourglass', 'lovheim', 'neuro', 'pad', 'plutchik']

convert(prototype("anger"), "core", "pad")    # (-0.62, 0.62, 0.60)
convert((-0.6, 0.8, 0.6), "pad", "core")      # -> AffectState
```

## Hub and spokes

Conversions route through the core. N models need 2N maps, not N².

```
   Hourglass ─┐                    ┌─ PAD
   Plutchik ──┼──►  AFFECT CORE  ◄─┼─ circumplex
   Lövheim ───┘                    └─ NeuroState
```

Every pair is reachable, **always**. A test asserts the graph is total.

## Fidelity

Each view declares how much survives the trip.

| Fidelity | Meaning |
| --- | --- |
| `EXACT` | Bijective on its subspace. Round-trips to machine precision. |
| `LOSSY` | Information is provably discarded — and `explain_loss` says what. |
| `HEURISTIC` | Calibrated rather than derived. The numbers are a judgement call, and the target model may not be well-evidenced at all. |

```python
fidelity("pad", "core")          # Fidelity.LOSSY
fidelity("hourglass", "pad")     # Fidelity.HEURISTIC  -- the weakest leg wins

print(explain_loss("circumplex", "core"))
# potency and unpredictability. This is why the circumplex cannot tell
# anger from fear: they differ on potency, and it has no potency axis.
```

A conversion that loses information is fine. One that loses it **silently** is not.

## PAD / VAD

Most of the field speaks **PAD** — Pleasure, Arousal, Dominance (Mehrabian &
Russell 1974), also called **VAD** when Pleasure is named Valence. It is what the
NRC-VAD lexicon and most dimensional emotion regressors emit.

The mapping is a **coordinate drop**:

| PAD | core |
| --- | --- |
| Pleasure | `valence` |
| Arousal | `arousal` |
| Dominance | `potency` |

That's it. No fitted weights, no regression, no unreachable region — because the
core *has* a potency axis, and PAD's dominance is what it looks like from
outside.

```python
for name in ("anger", "fear"):
    p, a, d = convert(prototype(name), "core", "pad")
    print(f"{name}: P={p:+.2f} A={a:.2f} D={d:+.2f}")

# anger: P=-0.62 A=0.62 D=+0.60
# fear:  P=-0.52 A=0.64 D=-0.60
```

Look at what PAD's third axis is *for*: anger and fear are nearly identical in
Pleasure and Arousal and differ almost entirely on Dominance. That is the single
most-replicated fact about dominance in the literature, and it is why PAD needs a
third axis at all.

**What PAD loses:** `unpredictability` (it has no such axis), and **ambivalence** —
a single signed Pleasure cannot represent positivity and negativity co-active, so
bittersweet reads as mild.

> **A caveat worth knowing.** The core's `potency` means *appraised coping* —
> "can I act on this?" — while PAD's dominance, as humans rate it, means how
> in-control you *feel* while in the grip of the state. They correlate at r≈0.46,
> and they are not the same construct: people rate `rage` as *less* dominant than
> `anger`, because being enraged is not being in control. The ordering that
> matters (anger above fear) holds in both. The magnitudes do not transfer. See
> [evidence](evidence.md).

## Russell's circumplex

An even simpler drop: `(valence, arousal)`.

```python
convert(prototype("anger"), "core", "circumplex")   # (-0.62, 0.62)
convert(prototype("fear"),  "core", "circumplex")   # (-0.52, 0.64)
```

Note those two are **almost the same point**. The circumplex cannot tell anger
from fear — which is the honest, structural limit of every valence/arousal model,
and it is why the library reports it rather than papering over it.

## The legacy models

`plutchik`, `hourglass` and `lovheim` are all registered views. See
[legacy views](legacy.md) for what they are for and what they cannot do.

```python
fidelity("core", "lovheim")    # Fidelity.HEURISTIC
print(explain_loss("hourglass", "core"))
# Sensitivity conflates negative valence with potency: it is unpleasant at
# BOTH poles (anger and fear alike)... That conflation cannot be undone.
```

## Registering your own

```python
from emotion_algebra.projection import register_view, Fidelity

register_view(
    "my_model",
    to_core=lambda x: ...,      # -> AffectState
    from_core=lambda s: ...,    # -> your type
    fidelity=Fidelity.LOSSY,
    loses="what your model cannot represent",   # required, if not EXACT
)
```

A lossy view that does not declare its loss raises `ValueError`. That is
deliberate.
