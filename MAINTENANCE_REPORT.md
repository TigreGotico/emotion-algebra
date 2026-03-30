# Maintenance Report — emotion_data

## 2026-03-30 — Phase 1 Revival

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-reviewed task specification; AI executed all changes.

### Actions Taken

1. **Migrated `setup.py` → `pyproject.toml`**
   - `python_requires = ">=3.10"`
   - Core dependency: `numpy` only
   - Optional extras: `[lexicon]` (pandas + CSV), `[deepmoji]` (torch + torchMoji), `[tagging]` (paralleldots)
   - File: `pyproject.toml`

2. **Added unit tests for emotion algebra** (`test/test_algebra.py`)
   - Intensity arithmetic: `annoyance+1==anger`, `anger+1==rage`, `serenity-1` → neutrality
   - Negation: `-anger==fear`, `-joy==sadness`, `-rage==terror`
   - Hyperintensity offsets: `rage+1` → mega prefix, `rage+3` → hyper prefix
   - Composition dyads: `joy+trust==Feeling("love")`, `joy+surprise==Feeling("delight")`
   - 4D vector shape and content assertions
   - Valence property checks
   - Comparison operators (`rage > anger`)
   - Neutrality identity element

3. **Added type hints and docstrings** to core classes
   - `Emotion`, `Neutrality`, `EmotionalDimension` — `emotion_data/plutchik.py`
   - `Feeling` — `emotion_data/feelings.py`
   - `CompositeEmotion`, `CompositeDimension` — `emotion_data/composite_emotions.py`
   - `Behaviour`, `BehavioralReaction` — `emotion_data/behaviour.py`

4. **Created `docs/index.md`**
   - Overview, algebra examples, emotion taxonomy table, installation instructions
   - Source citations for all key classes

### Files Modified
- `setup.py` → replaced by `pyproject.toml` (setup.py retained for legacy compatibility)
- `emotion_data/plutchik.py` — type hints + docstrings
- `emotion_data/feelings.py` — type hints + docstrings
- `emotion_data/composite_emotions.py` — type hints + docstrings
- `emotion_data/behaviour.py` — type hints + docstrings

### Files Created
- `pyproject.toml`
- `test/__init__.py`
- `test/test_algebra.py`
- `docs/index.md`
- `MAINTENANCE_REPORT.md` (this file)

---

## 2026-03-30 — Phase 2 Production Pass (90% coverage)

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-directed; AI executed all changes.

### Bug Fixes

| File | Line | Bug | Fix |
|------|------|-----|-----|
| `plutchik.py` | 497-500 | `Emotion.__bool__` returned `numpy.bool_` (TypeError in Python 3.11+) | Return `bool(...)` explicitly |
| `behaviour.py` | 139 | `if e:` silently skips negative-flow emotions (fear, terror) in `_get_behaviours()` | `if e is not None:` |
| `emotions.py` | 70 | `if emotion:` same negative-flow skip bug in `emotion_to_dimension()` | `if emotion is not None:` |
| `behaviour.py` | 172 | `from_data({})` raises `KeyError` on missing `'behaviour'` key | Graceful fallback when key absent |

### Test Suite Added

374 tests across 5 new files; all pass. Core coverage: 90% total.

| File | Tests | Coverage |
|------|-------|----------|
| `test/test_plutchik.py` | ~100 | 90% (`plutchik.py`) |
| `test/test_feelings.py` | ~80 | 88% (`feelings.py`) |
| `test/test_composite.py` | ~120 | 90% (`composite_emotions.py`) |
| `test/test_behaviour.py` | ~30 | 93% (`behaviour.py`) |
| `test/test_emotions_module.py` | ~44 | 87% (`__init__.py`), 82% (`emotions.py`), 86% (`lexicons.py`) |

Excluded from coverage: `deepmoji.py`, `tag.py` (optional external integrations) — configured in `.coveragerc`.

### Files Created
- `.coveragerc`
- `test/test_plutchik.py`
- `test/test_feelings.py`
- `test/test_composite.py`
- `test/test_behaviour.py`
- `test/test_emotions_module.py`

### Files Modified
- `emotion_data/plutchik.py` — `__bool__` bug fix
- `emotion_data/behaviour.py` — `if e is not None:` + `from_data()` guard
- `emotion_data/emotions.py` — `if emotion is not None:`
- `FAQ.md` — documented bugs, tests, coverage

---

## 2026-03-30 — Phase 3: Architectural & Scientific Remediation

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-directed; AI executed all changes.

### Changes

| Area | Change |
|---|---|
| `lexicons.py` | `get_emotion` → `get_word_emotion` — resolved import shadow in `__init__.py` |
| `__init__.py` | Rewritten — lazy deepmoji/tag imports; `EmotionAnalyzer` methods corrected |
| `composite_emotions.py` | `emotional_flow` changed from `np.linalg.norm` to signed sum; `type` property now has live branches |
| `plutchik.py`, `feelings.py`, `composite_emotions.py` | `copy(self)` → `deepcopy(self)` in all operator methods |
| `plutchik.py` | `Emotion.__bool__` reflects presence (non-zero), not valence |
| `plutchik.py` | `Emotion.valence` returns `int` (+1/−1/0) not `bool`; `POSITIVE_EMOTIONS` filter fixed |
| `plutchik.py` | `as_matrix` uses `np.ndarray` (deprecated `np.matrix` removed) |
| `plutchik.py` | Dead `self.name - other` branch in `__sub__` → `NotImplemented` |
| `feelings.py` | Removed duplicate `string_to_emotion`; `acknowledgement` spelling normalised; `FEELINGS` wrapped in `MappingProxyType` |
| `emotions.py` | `EMOTIONS` wrapped in `MappingProxyType`; `POSITIVE_EMOTIONS` filter fixed |
| `behaviour.py` | `BEHAVIOURS`, `REACTIONS`, `REACTION_TO_EMOTION_MAP` wrapped in `MappingProxyType` |
| `plutchik.py`, `composite_emotions.py` | Module docstrings document Plutchik vs Cambria provenance and known limitations |

### Post-fix coverage: 90% (375 tests, 0 failures)
