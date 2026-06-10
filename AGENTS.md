# emotion-algebra — agent guide

Signed-integer arithmetic library over a 4-axis affective space (Cambria Hourglass) grounded in Plutchik's Wheel. Emotions are first-class objects supporting operator overloading (`+ - * << >>`), with text/emoji scoring, cognitive appraisal, need-deficit mapping, and a continuous `FloatEmotion` space.

## Setup

```bash
pip install -e .                 # core: numpy, deepmoji-onnx
pip install -e .[lexicon]        # adds pandas for word→emotion CSV lookup
pip install -e .[fast]           # adds ahocorasick-ner phrase backend
pip install -e .[test]           # CI install_extras for build-tests
```

`deepmoji-onnx` is a core dependency; neural text→emoji→emotion scoring (`DeepMojiONNXAdapter`) is always available. The DeepMoji model downloads from HuggingFace on first use.

## Test

```bash
pytest test/
```

Coverage (matches CI config in `.coveragerc`, which omits `deepmoji.py` and `tag.py`):

```bash
pytest test/ --cov=emotion_algebra
```

## Lint

```bash
ruff check .
```

CI runs `ruff` via the shared lint workflow (`pre_commit: false`).

## Layout

- `emotion_algebra/plutchik.py` — core `Emotion`, `EmotionalDimension`, `Neutrality`; operator algebra; PASA axis definitions.
- `emotion_algebra/feelings.py` — Plutchik dyads (`Feeling`, `get_feeling`).
- `emotion_algebra/composite_emotions.py` — multi-axis `CompositeEmotion` (cross-axis `+`/`*`).
- `emotion_algebra/float_emotion.py` — continuous `FloatEmotion`, `from_embedding`, `blend`.
- `emotion_algebra/state.py` — `EmotionalState`, `EmotionTimeline` (weighted accumulation + decay).
- `emotion_algebra/text.py` — `from_text`, `score_text`, `score_mixed`, `from_mixed` (lexicon + emoji).
- `emotion_algebra/emoji.py` — emoji→emotion map, `DeepMojiAdapter`, `register_emoji`.
- `emotion_algebra/deepmoji.py` — `DeepMojiONNXAdapter` wrapping the `deepmoji-onnx` package.
- `emotion_algebra/appraisal.py` — Scherer CPM: `Appraisal`, `appraisal_to_emotion`, continuous v2 → neuro deltas (dopamine/serotonin/adrenaline, Lövheim).
- `emotion_algebra/needs.py` — Max-Neef / Murray / CIA-drive need-deficit → emotion.
- `emotion_algebra/distance.py` — `emotion_distance`, `closest_emotion`, `emotion_clusters` (4D geometry).
- `emotion_algebra/lexicons.py` — word→emotion/sentiment/color/AFINN/Hourglass CSV lookups; ships `word_emotion_lexicon.csv`.
- `emotion_algebra/__init__.py` — public API + `EmotionAnalyzer` facade.
- `emotion_algebra/__main__.py` — CLI (`emotion-algebra` console_script / `python -m emotion_algebra`).
- `scripts/build_lexicon.py` — lexicon CSV build helper.
- `test/` — pytest suite (one file per module). `examples/` — runnable usage scripts. `docs/` — module docs.

Entry points: one `console_scripts` entry (`emotion-algebra`). This is a plain Python library — no OVOS/OPM plugin or skill entry-point group.

## Conventions

- Branches: `dev` (work) / `master` (stable). NEVER `main`.
- Never edit `emotion_algebra/version.py`; gh-automations bumps semver from conventional-commit prefixes (`feat:` / `fix:` / `feat!:`).
- New repos private by default.
- Commit identity: JarbasAi <jarbasai@mailfence.com>.
- Reference `OpenVoiceOS/gh-automations` reusable workflows at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary (no history, dates, or design-mistake narration); describe current state only.
- CI is provided by `OpenVoiceOS/gh-automations`.

## Gotchas

- `TODO.md` and `ROADMAP.md` are local planning files — gitignored, never committed.
- `valence` is the Pleasantness axis only; `anger.valence == 0` by design (valence ⊥ arousal). Don't treat all negative emotions as negative valence.
- `deepmoji.py` and `tag.py` are excluded from coverage; the referenced `tag.py` is not present in `emotion_algebra/` (only `examples/tag.py`).
- Cross-axis `Emotion + Emotion` returns a `CompositeEmotion`, not an `Emotion`.
