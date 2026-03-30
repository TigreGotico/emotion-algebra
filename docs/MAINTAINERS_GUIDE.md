# Maintainers Guide

## Repository layout

```
emotion-algebra/
├── emotion_algebra/         # Package source
│   ├── plutchik.py          # Core Emotion / EmotionalDimension / Neutrality
│   ├── feelings.py          # Feeling (named dyads)
│   ├── composite_emotions.py# CompositeEmotion / CompositeDimension
│   ├── base.py              # EmotionBase ABC
│   ├── emotions.py          # EMOTIONS registry, get_emotion, constants
│   ├── behaviour.py         # Behaviour / BehavioralReaction
│   ├── lexicons.py          # Word→emotion CSV loader
│   ├── state.py             # EmotionalState, EmotionTimeline
│   ├── distance.py          # emotion_distance, closest_emotion, emotion_clusters
│   ├── text.py              # from_text, score_text, score_mixed, HFEmotionAdapter
│   ├── emoji.py             # EMOJI_EMOTION_MAP, DeepMojiAdapter, register_emoji
│   ├── appraisal.py         # Appraisal dataclass, appraisal_to_emotion
│   ├── float_emotion.py     # FloatEmotion (continuous space)
│   ├── reference_maps.py    # Raw reference data (Parrott, HUMAINE, etc.)
│   ├── __init__.py          # Public API + EmotionAnalyzer facade
│   ├── __main__.py          # CLI entry point
│   └── word_emotion_lexicon.csv
├── test/                    # pytest test suite
├── docs/                    # This documentation
├── examples/                # Runnable example scripts
├── pyproject.toml
├── README.md
├── FAQ.md
├── AUDIT.md
├── MAINTENANCE_REPORT.md
├── ROADMAP.md
└── SPECIFICATION.md
```

---

## Development setup

```bash
git clone <repo>
cd emotion-algebra
uv pip install -e ".[lexicon]"
uv run pytest test/ -v --cov=emotion_algebra --cov-config=.coveragerc
```

Python 3.10+ required. Core dependency: `numpy` only.

---

## Running tests

```bash
# Default (no optional extras)
uv run pytest test/ -v

# With coverage
uv run pytest test/ --cov=emotion_algebra --cov-report=term-missing

# Single module
uv run pytest test/test_emoji.py -v
```

Coverage targets: ≥94% overall; individual modules should not drop below 90%.

---

## Branching model

| Branch | Purpose |
|--------|---------|
| `master` | Stable releases (tagged) |
| `dev` | Integration branch — all PRs target `dev` |
| `feat/*` | Feature branches |
| `fix/*` | Bug-fix branches |

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/).

---

## Release process

1. All tests pass on `dev`: `uv run pytest test/ -q`
2. Update `emotion_algebra/version.py` (semver)
3. Update `pyproject.toml` `version` field to match
4. Update `CHANGELOG.md` (or `MAINTENANCE_REPORT.md`) with release notes
5. Merge `dev` → `master` via PR — no direct commits to `master`
6. Tag the release: `git tag vX.Y.Z`
7. Build: `uv build`
8. Publish: `uv publish` (human action — never automated)

---

## Adding a new module

1. Create `emotion_algebra/<module>.py` with full type hints and docstrings.
2. Re-export public symbols from `emotion_algebra/__init__.py`.
3. Add `EmotionAnalyzer` static methods where appropriate.
4. Write `test/test_<module>.py` — coverage must not drop overall.
5. Add a section to `docs/api_reference.md`.
6. Update `docs/index.md` Key classes table.
7. Update `FAQ.md` with a usage example.
8. Log in `MAINTENANCE_REPORT.md`.

---

## Extending the emoji map

The canonical `EMOJI_EMOTION_MAP` (`emoji.py`) is immutable. To add entries permanently:

1. Add to `_EMOJI_EMOTION_MAP_RAW` in `emoji.py` with a comment citing the scientific basis.
2. The value must be a name returned by `get_emotion()`.
3. Update `test/test_emoji.py` — `test_not_empty` and `test_all_values_are_valid_emotion_names` will catch errors automatically.

For per-process overrides without modifying the source, use `register_emoji()`.

---

## Optional extras policy

| Extra | Trigger | Test strategy |
|-------|---------|--------------|
| `[lexicon]` | `pandas` import in `lexicons.py` | CSV loaded at import; no mock needed |
| `[transformers]` | `HFEmotionAdapter.__init__` lazy import | Mock `transformers` in `test_text.py` |

Never add a hard dependency to the core package. All optional imports must be inside the function/method body with a clear `ImportError` message pointing to the correct extra.

---

## AI usage policy

All AI-generated changes must be logged in `MAINTENANCE_REPORT.md` with:
- Model name
- Intent
- Impact
- Test results

Every commit message must include the model name and state that the change is AI-generated (see existing commit history for the format).
