> **This page documents a [legacy view](legacy.md).** It describes the
> Plutchik/Hourglass layer, which is kept as a *vocabulary* and graded
> `METAPHOR`. For the model the library actually computes with, see
> [the core](theory.md) and [its laws](core-laws.md).

# CLI Reference

`python -m emotion_algebra` provides three modes: info lookup, expression evaluation, and an interactive REPL.

---

## Info mode

```bash
python -m emotion_algebra info <name>
```

`<name>` can be an emotion, feeling, or dimension.

```
$ python -m emotion_algebra info anger

anger
─────
  type      : Emotion
  axis      : sensitivity
  flow      : +2
  valence   : +0
  arousal   : 2
  affect    : activated neutral
  opposite  : fear
  base      : anger
```

```
$ python -m emotion_algebra info love

love
────
  type      : Feeling
  valence   : +2
  arousal   : 2
  affect    : excited positive
  components: joy, trust
```

```
$ python -m emotion_algebra info pleasantness

pleasantness
────────────
  type      : dimension
  axis      : pleasantness
  valence   : +1
  kind      : hedonic
  emotions  : grief ←0→ ecstasy
```

---

## Expression mode

```bash
python -m emotion_algebra <expression>
```

Supported operators: `+`, `-`, `*`, `//`, `>>`, `<<`.

```bash
python -m emotion_algebra "joy + trust"
# joy + trust  →  love  [Feeling](valence=+2, arousal=2, type='excited positive')

python -m emotion_algebra "rage - 1"
# rage - 1  →  anger  [Emotion](valence=+0, arousal=2, type='activated neutral')

python -m emotion_algebra "anger >> 2"
# anger >> 2  →  hyper anger  [Emotion](...)
```

Operands are resolved via `get_emotion()` or parsed as integers.

---

## Emoji shortcut

A single argument consisting entirely of emoji characters in `EMOJI_EMOTION_MAP` is scored directly:

```bash
python -m emotion_algebra 😊
# 😊  →  serenity  [Emotion](valence=+1, arousal=1, type='calm positive')

python -m emotion_algebra 😄🎉
# 😄🎉  →  ecstasy  [Emotion](valence=+3, arousal=3, type='excited positive')
```

---

## Interactive REPL

```bash
python -m emotion_algebra
```

Launches a Python REPL with all 24 named emotions pre-loaded as variables, plus `get_emotion`, `get_feeling`, `get_dimension`, `Neutrality`, `EMOTIONS`, `FEELINGS`.

```python
>>> joy + trust
FeelingObject:love
>>> rage - 1
anger
>>> anger.arousal
2
>>> joy.as_array
array([0, 0, 2, 0])
>>> exit()
```
