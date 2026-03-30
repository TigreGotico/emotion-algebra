# FAQ — emotion_data

## What is emotion_data?

A Python library implementing **emotion algebra** based on Plutchik's Wheel of Emotions and the Hourglass of Emotions. Emotions are first-class mathematical objects with operator overloading.

## What are the minimum dependencies?

Core package requires only `numpy`. All other dependencies are optional extras.

## How do I install just the core?

```bash
pip install emotion_data
```

## What optional extras are available?

| Extra | Deps | Enables |
|---|---|---|
| `[lexicon]` | pandas | `lexicons.py` — word→emotion CSV lookup |
| `[deepmoji]` | torch, torchMoji | `deepmoji.py` — emoji-based emotion tagging |
| `[tagging]` | paralleldots | `tag.py` — ParallelDots API tagging |

## How does intensity arithmetic work?

Emotions sit on a signed integer axis per dimension. Adding an integer moves up the axis:

```
annoyance (1) → anger (2) → rage (3) → mega rage (offset 1) → hyper rage (offset 3+)
```

## How do I get the opposite emotion?

Use negation: `-anger == fear`, `-joy == sadness`.

## How does joy + trust produce love?

When two emotions from **different** Hourglass dimensions are added, a `Feeling` dyad is created. The pair is looked up in `FEELING_NAMES` and the named feeling is returned.

## What Python versions are supported?

Python 3.10 and above.

## Where are the tests?

`test/test_algebra.py` — covers all core algebra operators, composition, vectors, valence, and comparison.

## What is the emotional_flow property?

Signed integer intensity: ±1 primary, ±2 secondary, ±3 tertiary, 0 neutral. Used as the scalar operand in all arithmetic.

## What is Neutrality?

The identity element: `emotion + Neutrality() == emotion`. Also returned when arithmetic reduces flow to zero.
