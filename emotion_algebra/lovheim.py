"""Lövheim's cube of emotion — the neurochemical model.

Lövheim (2012) proposes that the eight *basic affects* of Tomkins' affect
theory sit at the eight corners of a cube whose axes are the three monoamine
neurotransmitters:

* **serotonin** — satisfaction / self-confidence pole
* **dopamine** — reward-seeking / approach pole
* **noradrenaline** — arousal / alertness pole

Each axis is normalised to ``[0, 1]``: 0 is "low" and 1 is "high" relative to
that individual's own baseline, which the paper places at the cube's centre
``(0.5, 0.5, 0.5)``.  Every corner is a (low/high)³ combination and carries a
named Tomkins affect — see :data:`CORNERS`.

This module gives the cube a first-class implementation and a *bidirectional*
bridge to the Hourglass space used by the rest of the library:

* :meth:`LovheimPoint.closest_affect` — nearest corner (deterministic).
* :meth:`LovheimPoint.affect_blend` — trilinear weights over all 8 corners.
  Because the model is literally a cube, trilinear interpolation is exact, not
  an approximation: the weights are the volumes of the 8 sub-boxes the point
  cuts the cube into, and they always sum to 1.
* :meth:`LovheimPoint.to_float_emotion` / :meth:`LovheimPoint.from_float_emotion` —
  the Hourglass bridge, defined once by the :data:`CORNER_ANCHORS` table so the
  two directions cannot drift apart.

Where the two models disagree
-----------------------------
The bridge is exact going *out* of the cube and lossy coming *back*.  That
asymmetry is not an artefact of the anchor table — it is forced by the source
theories, and it is worth stating exactly, because it is a real finding about
the two models rather than an implementation wart.

**1. The cube cannot reach positive Aptitude.**
Tomkins' eight affects contain no *trust/admiration* — there is no basic affect
for the positive Aptitude pole.  Consequently ``aptitude ∈ [-3, 0]`` over the
*entire* cube: half of the Hourglass's Aptitude axis has no pre-image at all.
Five corners are negative (shame, distress, fear, anger, contempt) against two
positive (joy, interest) and one neutral (surprise).  This skew is also why the
cube's own centre, :data:`BASELINE`, maps to a mildly *negative* Hourglass
vector rather than to neutral.

**2. The map cannot be made injective.**
Three facts, none of them ours to choose, collide:

a. In Plutchik/Hourglass, opposite emotions are **exact negatives**:
   ``terror = -rage`` and ``amazement = -vigilance`` as 4-vectors.
b. In Lövheim's cube, those very pairs sit on **adjacent corners** —
   *fear/terror* ``(0,1,0)`` and *anger/rage* ``(0,1,1)`` differ only in
   noradrenaline; *surprise* ``(1,0,1)`` and *interest* ``(1,1,1)`` differ only
   in dopamine.
c. The bridge is a **weighted blend** — forced, if corners are to map onto
   their anchors exactly with anything continuous in between.

A blend crossing the midpoint of an edge whose two endpoints are exact
negatives must output zero on that axis.  So the Sensitivity axis necessarily
collapses along ``noradrenaline = 0.5, dopamine = 1``, and Attention along
``dopamine = 0.5, serotonin = 1``.  Both families contain the Hourglass origin,
which therefore has (at least) two exact pre-images.

Injectivity could only be bought by breaking (a) — e.g. giving fear and anger
anchors that are *not* exact negatives, by loading both with the negative
Pleasantness they plainly share.  That would be fudging the theory for
mathematical convenience, and it would break the far more valuable property
that each corner maps onto its named emotion's real vector.  We keep the
theory and document the consequence.

The deeper point: **Lövheim and Plutchik genuinely disagree about the geometry
of fear and anger.** Lövheim makes them neighbours separated by one
neurotransmitter; Plutchik makes them polar opposites.  The degeneracy above is
that disagreement, made arithmetic.

:meth:`LovheimPoint.from_float_emotion` is therefore a documented *projection*
with a deterministic tie-break, not an inverse.  Do not read it as one.

The neurotransmitter naming used by downstream consumers (which call the
third axis *adrenaline*) is the same axis: Lövheim's paper works with
**noradrenaline** (norepinephrine), the CNS arousal monoamine.  The
:attr:`LovheimPoint.adrenaline` alias exists for exactly that reason.

References
----------
Lövheim, H. (2012). A new three-dimensional model for emotions and monoamine
neurotransmitters.  *Medical Hypotheses*, 78(2), 341–348.
https://doi.org/10.1016/j.mehy.2011.11.016

Tomkins, S. S. (1962). *Affect Imagery Consciousness, Vol. I: The Positive
Affects*.  Springer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    from emotion_algebra.float_emotion import FloatEmotion

#: The cube's centre — every axis at its individual baseline (Lövheim 2012).
BASELINE: Tuple[float, float, float] = (0.5, 0.5, 0.5)

#: The eight basic affects at the cube's corners, keyed by
#: ``(serotonin, dopamine, noradrenaline)`` as low(0)/high(1) triples.
#: Reproduced exactly from Lövheim (2012), Table 1 / Fig. 1.
CORNERS: Dict[Tuple[int, int, int], str] = {
    (0, 0, 0): "shame/humiliation",
    (0, 0, 1): "distress/anguish",
    (0, 1, 0): "fear/terror",
    (0, 1, 1): "anger/rage",
    (1, 0, 0): "contempt/disgust",
    (1, 0, 1): "surprise",
    (1, 1, 0): "enjoyment/joy",
    (1, 1, 1): "interest/excitement",
}

#: Hourglass anchor vector for each corner affect, in axis order
#: ``[sensitivity, attention, pleasantness, aptitude]``.
#:
#: This table is the *single* definition of the cube↔Hourglass bridge; both
#: directions of the mapping are derived from it, so they cannot drift apart.
#: Each anchor is the Plutchik/Hourglass emotion (or dyad) that names the same
#: affect, at tertiary intensity (±3) where a direct counterpart exists:
#:
#: ===========================  ====================================================
#: Lövheim corner               Hourglass anchor and rationale
#: ===========================  ====================================================
#: ``anger/rage``               ``rage`` — Sensitivity +3, the axis's own name.
#: ``fear/terror``              ``terror`` — Sensitivity −3, the opposite pole.
#: ``enjoyment/joy``            ``ecstasy`` — Pleasantness +3.
#: ``distress/anguish``         ``grief`` — Pleasantness −3.
#: ``contempt/disgust``         ``loathing`` — Aptitude −3.
#: ``surprise``                 ``amazement`` — Attention −3 (the surprise pole).
#: ``interest/excitement``      ``vigilance`` — Attention +3 (the interest pole).
#: ``shame/humiliation``        Plutchik has **no shame primary**.  Shame is the
#:                              dyad ``grief + loathing`` (Pleasantness −3,
#:                              Aptitude −3) — exactly ``FEELING_NAMES["shame"]``,
#:                              so the anchor is that feeling's own vector rather
#:                              than an invented one.
#: ===========================  ====================================================
CORNER_ANCHORS: Dict[str, Tuple[float, float, float, float]] = {
    "shame/humiliation":   (0.0, 0.0, -3.0, -3.0),   # grief + loathing (Plutchik dyad "shame")
    "distress/anguish":    (0.0, 0.0, -3.0, 0.0),    # grief
    "fear/terror":         (-3.0, 0.0, 0.0, 0.0),    # terror
    "anger/rage":          (3.0, 0.0, 0.0, 0.0),     # rage
    "contempt/disgust":    (0.0, 0.0, 0.0, -3.0),    # loathing
    "surprise":            (0.0, -3.0, 0.0, 0.0),    # amazement
    "enjoyment/joy":       (0.0, 0.0, 3.0, 0.0),     # ecstasy
    "interest/excitement": (0.0, 3.0, 0.0, 0.0),     # vigilance
}


def _corner_order() -> list:
    """Corners in a fixed, deterministic iteration order."""
    return sorted(CORNERS.keys())


#: ``(8, 3)`` array of corner coordinates and the matching ``(8, 4)`` array of
#: their Hourglass anchors, both in :func:`_corner_order` order.  Built once so
#: the forward map can be evaluated on whole grids of points at a time.
_CORNER_COORDS = np.array(_corner_order(), dtype=float)
_ANCHOR_MATRIX = np.array(
    [CORNER_ANCHORS[CORNERS[c]] for c in _corner_order()], dtype=float
)


def _trilinear_weights(coords: np.ndarray) -> np.ndarray:
    """Trilinear corner weights for one or many cube points.

    Parameters
    ----------
    coords:
        Array of shape ``(..., 3)`` — cube coordinates in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Shape ``(..., 8)``, non-negative, summing to 1 along the last axis.
    """
    pts = np.asarray(coords, dtype=float)
    # For each corner c and axis i: weight factor is x_i when c_i == 1, else 1 - x_i.
    hi = pts[..., None, :]                       # (..., 1, 3)
    corner = _CORNER_COORDS                      # (8, 3)
    factors = np.where(corner == 1, hi, 1.0 - hi)  # (..., 8, 3)
    return np.prod(factors, axis=-1)             # (..., 8)


def _forward(coords: np.ndarray) -> np.ndarray:
    """Map cube points to Hourglass vectors — the closed form of the anchor blend.

    Substituting :data:`CORNER_ANCHORS` into the trilinear weights collapses to::

        sensitivity  =  3·(1-s)·d·(2n-1)
        attention    =  3·s·n·(2d-1)
        pleasantness = -3·(1-s)·(1-d) + 3·s·d·(1-n)
        aptitude     = -3·(1-d)·(1-n)

    which is what this computes (via the weights, so the two can never drift).

    Parameters
    ----------
    coords:
        Array of shape ``(..., 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(..., 4)`` in Hourglass axis order.
    """
    return _trilinear_weights(coords) @ _ANCHOR_MATRIX


def _forward_jacobian(coords: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of :func:`_forward`, shape ``(..., 4, 3)``.

    ``∂w_c/∂x_i`` is ``±1`` (positive when corner *c* is high on axis *i*)
    times the product of the other two axes' weight factors, so the whole
    Jacobian follows from the same factor array the forward map uses.
    """
    pts = np.asarray(coords, dtype=float)
    hi = pts[..., None, :]                          # (..., 1, 3)
    corner = _CORNER_COORDS                         # (8, 3)
    factors = np.where(corner == 1, hi, 1.0 - hi)   # (..., 8, 3)
    signs = np.where(corner == 1, 1.0, -1.0)        # (8, 3)

    # Leave-one-out products: for axis i, multiply the factors of the other two.
    dw = np.empty_like(factors)                     # (..., 8, 3)
    for i in range(3):
        others = [j for j in range(3) if j != i]
        dw[..., i] = signs[:, i] * factors[..., others[0]] * factors[..., others[1]]

    # (..., 8, 3) → (..., 4, 3):  J[:, k, i] = Σ_c dw[c, i] · anchor[c, k]
    return np.einsum("...ci,ck->...ki", dw, _ANCHOR_MATRIX)


#: Seeds and iteration budget for :func:`_project_to_cube`.
#:
#: The residual surface ``‖_forward(p) − target‖²`` is multimodal — the forward
#: map is multilinear, not convex — so a single descent lands in whichever
#: basin it started in.  The projection therefore runs a damped Gauss-Newton
#: (Levenberg-Marquardt) descent from every corner plus the baseline, all in
#: one vectorised batch, and keeps the best.  Nine seeds cover every basin of
#: the cube; LM converges quadratically, so the iteration budget is small.
_PROJECTION_SEEDS = np.vstack([_CORNER_COORDS, np.array([BASELINE])])
_PROJECTION_ITERS = 60

#: Residuals and baseline distances are rounded to this many decimals before
#: the tie-break compares them, so float noise never decides which of two
#: equally-exact pre-images wins.
_PROJECTION_TIE_DECIMALS = 9


def _project_to_cube(target: np.ndarray) -> np.ndarray:
    """Cube point whose :func:`_forward` image is nearest to *target*.

    Returns the least-squares projection — the cube point minimising
    ``‖_forward(p) − target‖``.  Two facts make this a projection rather than
    an inverse:

    * The forward map is **not surjective**.  It sends a 3-dimensional cube
      into 4-dimensional Hourglass space, so a generic Hourglass vector has no
      exact pre-image at all.
    * The forward map is **not injective** either.  Distinct cube points can
      share an image — the Hourglass origin, for instance, is hit exactly by
      both ``(0, 1, 0.5)`` (halfway between fear and anger, whose anchors
      cancel) and ``(1, 0.5, 1)`` (halfway between surprise and interest).

    So this returns *a* best pre-image, chosen deterministically: lowest
    residual, then nearest to :data:`BASELINE`, then seed order.  Fixed seeds,
    fixed damping schedule, no randomness, no solver dependency.  Anything that
    *is* uniquely on the image — every corner — is recovered to float precision.
    """
    target = np.asarray(target, dtype=float).ravel()[:4]

    pts = _PROJECTION_SEEDS.copy()                       # (K, 3)
    lam = np.full(len(pts), 1e-3)
    err = np.linalg.norm(_forward(pts) - target, axis=-1)

    eye = np.eye(3)
    for _ in range(_PROJECTION_ITERS):  # noqa: B007 - fixed budget, LM converges well inside it
        resid = _forward(pts) - target                   # (K, 4)
        jac = _forward_jacobian(pts)                     # (K, 4, 3)
        jtj = np.einsum("kai,kaj->kij", jac, jac)        # (K, 3, 3)
        jtr = np.einsum("kai,ka->ki", jac, resid)        # (K, 3)

        damped = jtj + lam[:, None, None] * eye
        try:
            # solve() needs an explicit column vector to batch unambiguously.
            step = np.linalg.solve(damped, -jtr[..., None])[..., 0]
        except np.linalg.LinAlgError:  # pragma: no cover - damping keeps this away
            step = -jtr

        candidate = np.clip(pts + step, 0.0, 1.0)
        cand_err = np.linalg.norm(_forward(candidate) - target, axis=-1)

        improved = cand_err < err
        pts = np.where(improved[:, None], candidate, pts)
        err = np.where(improved, cand_err, err)
        # Classic LM damping: trust the linear model more when it pays off.
        lam = np.where(improved, np.maximum(lam / 3.0, 1e-12), np.minimum(lam * 3.0, 1e12))

    # The forward map is NOT injective (see the module docstring), so several
    # seeds can converge to genuinely different, equally-exact pre-images.
    # Rank by residual first, then by distance to baseline — of two equally
    # faithful readings of the same affect, the less extreme one is the honest
    # choice. Seed order breaks any remaining tie, so the result is stable.
    resid = np.round(err, _PROJECTION_TIE_DECIMALS)
    to_baseline = np.round(
        np.linalg.norm(pts - np.array(BASELINE), axis=-1), _PROJECTION_TIE_DECIMALS
    )
    order = np.lexsort((to_baseline, resid))  # primary key last
    return pts[int(order[0])]


@dataclass(frozen=True)
class LovheimPoint:
    """A point in Lövheim's neurochemical cube.

    Parameters
    ----------
    serotonin, dopamine, noradrenaline:
        Monoamine levels, each in ``[0, 1]`` relative to the individual's own
        baseline (0.5).  Values outside the cube raise :class:`ValueError` —
        the model is only defined inside it.

    Examples
    --------
    >>> LovheimPoint(1.0, 1.0, 0.0).closest_affect()
    'enjoyment/joy'
    >>> LovheimPoint(*BASELINE).is_baseline
    True
    """

    serotonin: float = 0.5
    dopamine: float = 0.5
    noradrenaline: float = 0.5

    def __post_init__(self) -> None:
        for axis in ("serotonin", "dopamine", "noradrenaline"):
            v = getattr(self, axis)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise TypeError(f"{axis} must be a number, got {type(v).__name__}")
            if not np.isfinite(v):
                raise ValueError(f"{axis} must be finite, got {v!r}")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{axis} must be in [0, 1], got {v!r}")

    # ------------------------------------------------------------------
    # Aliases and basics
    # ------------------------------------------------------------------

    @property
    def adrenaline(self) -> float:
        """Alias for :attr:`noradrenaline`.

        Lövheim's third axis is noradrenaline (norepinephrine); downstream
        consumers that call the arousal monoamine "adrenaline" mean this axis.
        """
        return self.noradrenaline

    @property
    def as_array(self) -> np.ndarray:
        """``[serotonin, dopamine, noradrenaline]`` as a float array."""
        return np.array(
            [self.serotonin, self.dopamine, self.noradrenaline], dtype=float
        )

    @property
    def is_baseline(self) -> bool:
        """``True`` when every axis sits at the individual baseline (0.5)."""
        return bool(np.allclose(self.as_array, BASELINE))

    @property
    def name(self) -> str:
        """The nearest corner affect — see :meth:`closest_affect`."""
        return self.closest_affect()

    # ------------------------------------------------------------------
    # Cube geometry
    # ------------------------------------------------------------------

    def closest_affect(self) -> str:
        """Return the name of the nearest corner affect.

        Ranked by Euclidean distance, then by corner coordinate — so a point
        exactly equidistant from several corners (the cube's centre is
        equidistant from all eight) always resolves the same way, never by
        float epsilon or dict order.

        Returns
        -------
        str
            One of the eight values of :data:`CORNERS`.
        """
        vec = self.as_array
        best_key = None
        best_name = ""
        for corner in _corner_order():
            d = round(float(np.linalg.norm(np.array(corner, dtype=float) - vec)), 9)
            key = (d, corner)
            if best_key is None or key < best_key:
                best_key = key
                best_name = CORNERS[corner]
        return best_name

    def affect_blend(self) -> Dict[str, float]:
        """Return the trilinear weight of every corner affect.

        On a cube, trilinear interpolation is exact: the weight of a corner is
        the volume of the opposite sub-box the point carves out.  Weights are
        non-negative and sum to 1, so this is a genuine distribution over the
        eight basic affects rather than a heuristic similarity score.

        Returns
        -------
        dict
            Maps each of the eight affect names to its weight in ``[0, 1]``.

        Examples
        --------
        >>> blend = LovheimPoint(1.0, 1.0, 0.0).affect_blend()
        >>> blend["enjoyment/joy"]
        1.0
        >>> centre = LovheimPoint(0.5, 0.5, 0.5).affect_blend()
        >>> all(round(w, 6) == 0.125 for w in centre.values())
        True
        """
        s, d, n = self.serotonin, self.dopamine, self.noradrenaline
        weights: Dict[str, float] = {}
        for corner in _corner_order():
            cs, cd, cn = corner
            w = (
                (s if cs else 1.0 - s)
                * (d if cd else 1.0 - d)
                * (n if cn else 1.0 - n)
            )
            weights[CORNERS[corner]] = float(w)
        return weights

    def distance(self, other: "LovheimPoint") -> float:
        """Euclidean distance to *other* inside the cube."""
        return float(np.linalg.norm(self.as_array - other.as_array))

    def deltas_from_baseline(self) -> Tuple[float, float, float]:
        """Signed displacement from baseline as ``(dopamine, serotonin, adrenaline)``.

        Ordered to match the convention of downstream neurotransmitter
        consumers of this readout.  Each value is in
        ``[-0.5, +0.5]``.
        """
        return (
            float(self.dopamine - BASELINE[1]),
            float(self.serotonin - BASELINE[0]),
            float(self.noradrenaline - BASELINE[2]),
        )

    # ------------------------------------------------------------------
    # Hourglass bridge
    # ------------------------------------------------------------------

    def to_float_emotion(self) -> "FloatEmotion":
        """Project this cube point into the 4-axis Hourglass space.

        The point's :meth:`affect_blend` weights are applied to the corner
        anchor vectors in :data:`CORNER_ANCHORS`.  A corner therefore maps
        exactly onto its anchor, and interior points interpolate smoothly
        between them.

        Returns
        -------
        FloatEmotion

        Examples
        --------
        >>> fe = LovheimPoint(0.0, 1.0, 1.0).to_float_emotion()  # anger/rage
        >>> float(fe.as_array[0])
        3.0
        """
        from emotion_algebra.float_emotion import FloatEmotion

        return FloatEmotion(*_forward(self.as_array))

    @classmethod
    def from_float_emotion(cls, fe: "FloatEmotion") -> "LovheimPoint":
        """Map a Hourglass vector into the cube — the **projection**, not an inverse.

        A faithful bridge between these two models is lossy in this direction,
        and it is worth being precise about why:

        * Tomkins' eight affects do not span the Hourglass.  There is no
          *trust/admiration* affect in Lövheim's cube, so the positive-Aptitude
          region has no corner to anchor to, and no cube point reproduces it.
        * The map is not injective.  The Hourglass origin, for instance, has two
          exact pre-images (see :func:`_project_to_cube`).

        So this returns the cube point whose image is nearest to *fe*, chosen
        deterministically among equally-good candidates: nearest to
        :data:`BASELINE` wins.  A vector inside
        :data:`~emotion_algebra.distance.NEUTRAL_RADIUS` of the origin
        short-circuits to :data:`BASELINE` outright — an inert state means *no*
        monoamine displacement, which is the one reading the least-squares
        projection would not reliably give you (the origin's exact pre-images
        sit at cube corners' midpoints, not at the centre).

        Corners still round-trip exactly.

        Parameters
        ----------
        fe:
            Any :class:`~emotion_algebra.float_emotion.FloatEmotion`.

        Returns
        -------
        LovheimPoint

        Examples
        --------
        >>> from emotion_algebra.emotions import get_emotion
        >>> from emotion_algebra.float_emotion import FloatEmotion
        >>> p = LovheimPoint.from_float_emotion(FloatEmotion.from_emotion(get_emotion("rage")))
        >>> p.closest_affect()
        'anger/rage'
        >>> LovheimPoint.from_float_emotion(FloatEmotion()).is_baseline  # neutral → no displacement
        True
        """
        from emotion_algebra.distance import NEUTRAL_RADIUS

        target = np.asarray(fe.as_array, dtype=float).ravel()[:4]
        if float(np.linalg.norm(target)) < NEUTRAL_RADIUS:
            return cls(*BASELINE)
        coords = _project_to_cube(target)
        return cls(*(float(min(1.0, max(0.0, c))) for c in coords))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "lovheim_point",
            "serotonin": self.serotonin,
            "dopamine": self.dopamine,
            "noradrenaline": self.noradrenaline,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LovheimPoint":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            serotonin=data.get("serotonin", 0.5),
            dopamine=data.get("dopamine", 0.5),
            noradrenaline=data.get("noradrenaline", 0.5),
        )

    def to_json(self, **kwargs) -> str:
        """Serialize to a versioned JSON string."""
        from emotion_algebra.serialization import to_json
        return to_json(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"LovheimPoint(serotonin={self.serotonin:.3g}, "
            f"dopamine={self.dopamine:.3g}, "
            f"noradrenaline={self.noradrenaline:.3g})"
        )

    def __str__(self) -> str:
        return self.closest_affect()


#: The eight corners as ready-made :class:`LovheimPoint` constants,
#: keyed by affect name.
CORNER_POINTS: Dict[str, LovheimPoint] = {
    name: LovheimPoint(*(float(v) for v in coords))
    for coords, name in CORNERS.items()
}


def closest_affect(serotonin: float, dopamine: float, noradrenaline: float) -> str:
    """Convenience wrapper: name the affect nearest to a raw monoamine triple."""
    return LovheimPoint(serotonin, dopamine, noradrenaline).closest_affect()
