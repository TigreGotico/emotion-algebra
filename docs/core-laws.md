# The laws of the affect core

**Is it still an algebra?** Yes — a better-specified one.

The old claim was that emotions form a *vector space*: they add, they scale, and
every emotion has a negation. That claim is false, and it was load-bearing: it is
what made the average of maximal rage and maximal terror come out as **calm**.

The true structure is three things at once, and each has laws that hold *exactly*
— machine-checked in `test/test_core_laws.py` with `hypothesis`.

```
(S, blend)   is a barycentric algebra   (a convex space)
(S, d)       is a compact metric space
{relax_t}    is a contraction semigroup with a unique fixed point
```

---

## 1. `(S, blend)` is a barycentric algebra

A **barycentric algebra** (equivalently: a *convex space*) is a real, named
algebraic structure with exactly four axioms. The core satisfies all four.

```
idempotence            blend(a, a, w) = a                       for all w
unit                   blend(a, b, 0) = a    blend(a, b, 1) = b
skew-commutativity     blend(a, b, w) = blend(b, a, 1-w)
barycentric assoc.     blend(blend(a,b,p), c, q) = blend(a, blend(b,c,r), s)
                           where  s = 1 - (1-p)(1-q)   and   r = q/s
```

**Stone's theorem** (1949) says the models of this theory are exactly the convex
subsets of real vector spaces. So we have not abandoned rigour by giving up the
vector space — we have said *precisely which subset of one* we live in.

Two consequences worth stating:

- **Closure is free.** A convex combination of points in the core is *always* in
  the core. `blend` never clamps, never saturates, never needs a guard. The old
  additive algebra had to clamp constantly, and clamping is what broke
  associativity there.
- **Naive associativity is false**, and that is not a defect. A weight is a share
  of *what remains*, not a share of the whole, so the weights must be
  reparametrized — that is what the fourth axiom says. The library ships a
  concrete counterexample as a test, so nobody "fixes" it.

## 2. `(S, d)` is a metric space

```
d(a, b) >= 0
d(a, a) = 0
d(a, b) = d(b, a)
d(a, c) <= d(a, b) + d(b, c)
```

And a law that ties the metric to the algebra: a blend **lies between its
endpoints** — `d(a, blend(a,b,w)) <= d(a,b)`.

> Honest caveat: no study comparing Euclidean against angular distance on human
> similarity judgements was located. Euclidean is a reasonable default, not a
> validated choice.

## 3. `{relax_t}` is a contraction semigroup — and this is a *theorem*

This is the part that answers "what is the gravitational attractor?".

```
identity      relax(a, 0)                = a
semigroup     relax(relax(a, t1), t2)    = relax(a, t1 + t2)        EXACTLY
fixed point   relax(SET_POINT, t)        = SET_POINT
contraction   d(relax(a,t), relax(b,t))  = 2^(-t/h) · d(a, b)
```

The contraction factor is `2^(−t/h) < 1` for all `t > 0`. The core is a complete
metric space. So by the **Banach fixed-point theorem**, `SET_POINT` is the
**unique** fixed point, and *every* state converges to it, exponentially, from
anywhere.

That is not a design preference. It is a theorem, and the library tests it:

```python
relax(any_state, dt=100_000, half_life=300)  ==  SET_POINT     # always
```

**And the attractor is not the origin.** Core affect is always on (Barrett &
Bliss-Moreau 2009) — "lack of emotion" is not a state anything occupies. Rest is
a mildly positive, low-arousal *set point* (the positivity offset; Cacioppo &
Berntson). A test asserts that the state everything converges to has
`valence > 0` and is **not** the origin.

This is what a needs-driven agent needs: the set point is the attractor, `drive()`
is the restoring force, and a need deficit *is* a displacement from rest.

## 4. `intensify` is a partial monoid action

An action of `(ℝ≥₀, ×)` on the core:

```
unit            intensify(a, 1)                = a
composition     intensify(intensify(a,j), k)   = intensify(a, j·k)      [1]
annihilator     intensify(a, 0)                = ORIGIN
distributivity  intensify(blend(a,b,w), k)     = blend(intensify(a,k), intensify(b,k), w)   [1]
```

`[1]` holds where nothing clamps — the action is *partial*, because the core is
bounded. This is stated, not hidden.

Note the annihilator: scaling to zero lands on the **origin**, which is exactly
why the origin is a coordinate fact rather than a psychological state. Nothing
*relaxes* there; you can only get there by multiplying by zero.

Negative factors raise `ValueError`. There is no negation, and there will be no
negation by the back door.

---

## The laws that do not hold

An algebra is defined as much by what it refuses to do. These are tested too.

| | why |
|---|---|
| **no `__neg__`** | Sadness is not "minus joy". It has its own action readiness — withdraw, help-seek — which is not "negative approach". |
| **no `__sub__`** | Subtraction presupposes an additive inverse. There isn't one. |
| **no `__add__`** | Emotions do not sum; they *mix*. Addition leaves the space; convex mixture cannot. |
| **no `__mul__`** | "Cross-axis product" was never an operation with a meaning. |

The one thing the old algebra was proudest of — `-anger == fear` — is the one
thing the evidence most clearly refutes. Anger and fear are **neighbours**: both
unpleasant, both aroused, separated by *control*. It survives in the library as a
lexical fact about **Plutchik's wheel** (`grade: METAPHOR`), where it belongs,
and not as a law of the core.
