> **Lövheim's model.** A faithful implementation of Lövheim (2012),
> graded `SPECULATIVE`, see [evidence](evidence.md). For the
> library's own neurochemical layer, see
> [neurochemistry](neurochemistry.md).

# The Lövheim cube

`emotion_algebra.lovheim` implements the neurochemical model of
Lövheim (2012), *A new three-dimensional model for emotions and monoamine
neurotransmitters*, Medical Hypotheses 78(2):341-8.

Three monoamine axes, **serotonin**, **dopamine**, **noradrenaline**, each
normalised to `[0, 1]` against the individual's own baseline of `0.5`, span a
cube whose eight corners are Tomkins' eight basic affects.

```python
from emotion_algebra import LovheimPoint

LovheimPoint(1.0, 1.0, 0.0).closest_affect()   # 'enjoyment/joy'
LovheimPoint(0.5, 0.5, 0.5).is_baseline        # True
```
## The corners

The corner assignment is the paper's, not ours:

| (S, D, N) | Tomkins affect | Hourglass anchor (Sen, Att, Pls, Apt) |
|---|---|---|
| (0,0,0) | shame/humiliation | (0, 0, −3, −3) |
| (0,0,1) | distress/anguish | (0, 0, −3, 0) |
| (0,1,0) | fear/terror | (−3, 0, 0, 0) |
| (0,1,1) | anger/rage | (+3, 0, 0, 0) |
| (1,0,0) | contempt/disgust | (0, 0, 0, −3) |
| (1,0,1) | surprise | (0, −3, 0, 0) |
| (1,1,0) | enjoyment/joy | (0, 0, +3, 0) |
| (1,1,1) | interest/excitement | (0, +3, 0, 0) |

The **anchor** column is ours: it is how each Tomkins affect is expressed in the
library's 4-axis Hourglass space, and it is what makes the two models
interoperable. Seven corners map onto a Plutchik primary at full intensity.
Shame is the exception, Plutchik has no shame primary, so it is anchored to the
**grief + loathing** dyad (sadness ⊕ disgust), which is the standard Plutchik
reading of shame/remorse.

Interior points are then defined by **trilinear interpolation** over these
anchors. On a cube that is exact, not an approximation: a corner's weight is the
volume of the sub-box opposite the point, and the eight weights are non-negative
and sum to 1, a genuine distribution over the basic affects.

```python
LovheimPoint(0.5, 0.5, 0.5).affect_blend()   # all eight at 0.125
```
## Two things the cube cannot do

Both are **proven properties of the model**, not defects, and both are locked by
tests.

### 1. It cannot express positive Aptitude

Over the *entire* cube, `aptitude ∈ [−3, 0]`.

Tomkins' eight affects contain no trust/acceptance/admiration affect, so no
corner anchors a positive Aptitude. Because interior points are a **convex**
blend of the corners, no interior point can produce one either. The cube
therefore cannot represent trust at all.

This is a genuine expressiveness limit of the model. Extending it with a fourth
axis belongs in a **separate**, separately-cited module: Lövheim's model is
explicitly about the three monoamines, and any further neurotransmitter is not
part of it. Locked by `test_cube_never_reaches_positive_aptitude`.

### 2. The map to Hourglass space is not injective

Distinct cube points can share an image. For example `F(0, 1, 0.5)` and
`F(1, 0.5, 1)` both land on the origin.

The reason is structural, and it follows from combining the two models:

1. Plutchik/Hourglass makes opposites **exact negatives**, `terror = −rage`,
   `amazement = −vigilance`.
2. Lövheim places those same pairs at **adjacent** corners: fear (0,1,0) and
   anger (0,1,1) differ in a single bit (noradrenaline).
3. The bridge is a weighted blend of corner anchors.

Any blend crossing an edge whose two endpoints are exact negatives *must* output
zero on that axis. So an entire family of interior points collapses onto the same
Hourglass vector. No choice of anchors avoids this while keeping both models
intact, and **adding dimensions does not help**, fear and anger stay adjacent and
antipodal at any dimension count.

Consequences, all deliberate:


* `from_float_emotion()` finds a **least-squares pre-image** by Levenberg, Marquardt
  from nine seeds (the eight corners plus baseline), with an analytic Jacobian.
* Where several pre-images are equally exact, it prefers the one **closest to
  baseline**. Of two equally faithful readings of the same affect, the less
  extreme one is the honest choice. Seed order breaks any remaining tie, so the
  result is deterministic.
* A neutral input short-circuits to baseline rather than to an arbitrary exact
  pre-image (without this, a neutral emotion projects to "surprise").

Locked by `test_forward_map_is_not_injective`.

## The cube's centre is not neutral

`LovheimPoint(0.5, 0.5, 0.5).to_float_emotion()` is `(0, 0, −0.375, −0.75)`, mildly
*negative*, not the Hourglass origin.

Five of Tomkins' eight affects are negative (shame, distress, fear, anger,
contempt), so the centroid of the corner anchors is hedonically negative. This is
a property of the affect set, not a calibration error. The inverse direction
still maps neutral back to baseline, so the round trip is deliberately asymmetric
here.

## Bridging to the rest of the library

```python
from emotion_algebra import EmotionalState, get_emotion

state = EmotionalState()
state.apply(get_emotion("ecstasy"))

state.to_lovheim()                 # live neurochemical readout
state.to_lovheim().affect_blend()  # distribution over Tomkins' affects
```
`deltas_from_baseline()` returns signed `(dopamine, serotonin, adrenaline)`
displacements, the ordering downstream neurotransmitter consumers expect.

> **`adrenaline` is an alias for `noradrenaline`.** Lövheim's third axis is
> noradrenaline (norepinephrine). Downstream consumers that say "adrenaline"
> mean this axis.

## Appraisal → cube

`appraisal_to_lovheim()` runs Scherer's appraisal checks straight through to
neurochemistry, which is the full agent pattern:

```
needs → appraisal → Hourglass → cube → reaction
```
`float_emotion_to_lovheim_deltas()` gives the signed version of the same readout.
The older `float_emotion_to_neuro_deltas()` is **deprecated but unchanged**, it
keeps its original non-negative contract, because the engine that consumes it
depends on that sign convention. The replacement returns *signed* deltas and is
not a drop-in substitution.

---
[← Neurochemistry](neurochemistry.md) · [Home](index.md) · [Interop →](interop.md)
