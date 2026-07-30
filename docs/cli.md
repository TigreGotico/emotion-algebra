# CLI Reference

`python -m emotion_algebra` runs several subcommands.

- The `info` subcommand looks up a single emotion, feeling, or dimension.
- Plain expression mode evaluates an arithmetic expression against the library's emotion algebra.
- The `analyze` subcommand reads the affect core out of a line of text.
- The `distance` subcommand compares two named emotions.
- The `wheel` subcommand renders Plutchik's wheel to a PNG.
- With no arguments at all, the command starts an interactive REPL.

Two flags apply across subcommands. The `--json` flag switches the output of `analyze` and `distance` to a single JSON object, for scripting. The `--lang` flag sets the language that `analyze` reads text in. It defaults to `en`.

---

## Info mode

Info mode prints the full record for one name: its type, its axes, and its relationships to other emotions.

```bash
python -m emotion_algebra info <name>
```
The `<name>` argument can be an emotion, feeling, or dimension.

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
The same command works for a feeling.

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
It also works for a dimension.

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
That covers info mode. Expression mode is the next step up: it evaluates arithmetic on named emotions.

## Expression mode

Expression mode evaluates a single expression on the command line and prints the resulting emotion or feeling.

```bash
python -m emotion_algebra <expression>
```
Supported operators: `+`, `-`, `*`, `//`, `>>`, `<<`. Each one below shows a full command and the line it prints.

```bash
python -m emotion_algebra "joy + trust"
# joy + trust  →  love  [Feeling](valence=+2, arousal=2, type='excited positive')

python -m emotion_algebra "rage - 1"
# rage - 1  →  anger  [Emotion](valence=+0, arousal=2, type='activated neutral')

python -m emotion_algebra "anger >> 2"
# anger >> 2  →  hyper anger  [Emotion](...)
```
Each operand resolves through `get_emotion()`, or parses as an integer when it is numeric. When a name does not resolve to a known emotion, or an operator is not supported, the command prints an error to standard error and exits with status 1.

## Emoji shortcut

A single argument consisting entirely of emoji characters in `EMOJI_EMOTION_MAP` scores directly, with no expression syntax needed.

```bash
python -m emotion_algebra 😊
# 😊  →  serenity  [Emotion](valence=+1, arousal=1, type='calm positive')

python -m emotion_algebra 😄🎉
# 😄🎉  →  ecstasy  [Emotion](valence=+3, arousal=3, type='excited positive')
```
## Analyze mode

The `analyze` subcommand reads a line of text and prints the affect core it detects, including potency, the axis most sentiment tools drop. English text goes through the DeepMoji-based encoder. Portuguese and Arabic go through an experimental multilingual probe. A language that has no evaluated encoder is refused rather than approximated.

```bash
python -m emotion_algebra analyze "nobody has replied to me"
python -m emotion_algebra analyze "ninguém me respondeu" --lang pt
```
Plain text output prints valence, potency, arousal, and unpredictability, plus ambivalence when it is above a small threshold, followed by the dominant emotion, the dominant tendency, and the top labels with their weights. Add `--json` to get the same fields, plus the dominant emotion, the dominant tendency, and the top-k label distribution, as one JSON object:

```bash
python -m emotion_algebra analyze "nobody has replied" --json
```
## Distance mode

The `distance` subcommand reports the metric distance between two named emotions, plus two normalized similarity scores: a default one and a cosine one.

```bash
python -m emotion_algebra distance joy grief
python -m emotion_algebra distance joy grief --json
```
## Wheel mode

The `wheel` subcommand renders Plutchik's wheel to a PNG, centered on the named emotion. This subcommand needs the `emotion-algebra[viz]` extra.

```bash
python -m emotion_algebra wheel joy
# wrote joy_wheel.png
```
## Interactive REPL

The REPL loads the whole named-emotion set into scope, so you can chain expressions without re-importing anything.

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
---
[← Hourglass laws](laws.md) · [Home](index.md) · [API reference →](api_reference.md)
