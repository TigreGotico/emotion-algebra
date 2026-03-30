# Maintenance Report — emotion_algebra

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
   - `Emotion`, `Neutrality`, `EmotionalDimension` — `emotion_algebra/plutchik.py`
   - `Feeling` — `emotion_algebra/feelings.py`
   - `CompositeEmotion`, `CompositeDimension` — `emotion_algebra/composite_emotions.py`
   - `Behaviour`, `BehavioralReaction` — `emotion_algebra/behaviour.py`

4. **Created `docs/index.md`**
   - Overview, algebra examples, emotion taxonomy table, installation instructions
   - Source citations for all key classes

### Files Modified
- `setup.py` → replaced by `pyproject.toml` (setup.py retained for legacy compatibility)
- `emotion_algebra/plutchik.py` — type hints + docstrings
- `emotion_algebra/feelings.py` — type hints + docstrings
- `emotion_algebra/composite_emotions.py` — type hints + docstrings
- `emotion_algebra/behaviour.py` — type hints + docstrings

### Files Created
- `pyproject.toml`
- `test/__init__.py`
- `test/test_algebra.py`
- `docs/index.md`
- `MAINTENANCE_REPORT.md` (this file)

---

## 2026-03-30 — Phase 2 Architectural Refactor

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-specified step-by-step plan; AI executed all changes.

### Actions Taken

1. **Created `emotion_algebra/base.py`** — `EmotionBase` ABC
   - Abstract properties: `name`, `emotional_flow`, `valence`, `arousal`, `type`, `emotion_vector`
   - Concrete shared implementations: `as_array`, `as_matrix`, `__bool__`, `__int__`, `__float__`, `__lt__`, `__le__`, `__gt__`, `__ge__`

2. **`Emotion(EmotionBase)`** — `emotion_algebra/plutchik.py`
   - Removed methods now provided by `EmotionBase`: `as_array`, `as_matrix`, `__bool__`, `__lt__`, `__le__`, `__gt__`, `__ge__`
   - Retained `__int__` (includes `intensity_offset`), `__float__`, `__eq__`, `__ne__`
   - Removed dead TODO comment block

3. **`CompositeEmotion(EmotionBase)`** — `emotion_algebra/composite_emotions.py`
   - Broke inheritance from `Emotion` — `CompositeEmotion` is now a peer, not a subclass
   - `__init__` no longer calls `Emotion.__init__`; no `intensity_offset`
   - Added `string_to_emotion` static method (was previously inherited)
   - Removed duplicate `__lt__`, `__le__`, `__gt__`, `__ge__` (now from `EmotionBase`)
   - Removed commented-out print and dead TODO block in `kind` property

4. **`Feeling(EmotionBase)`** — `emotion_algebra/feelings.py`
   - Added `EmotionBase` as base class
   - Removed debug `print(self.emotions)` in `secondary_name`
   - Removed duplicate `__int__`, `__float__`, `__lt__`, `__le__`, `__gt__`, `__ge__`
   - Removed large commented-out dead operator code block
   - Removed `if __name__ == '__main__'` block

5. **`emotion_algebra/__init__.py`** — exported `EmotionBase`

### Coverage
- Before: 87% total (1,350 stmts, 169 missed)
- After: 90% total (1,314 stmts, 126 missed — code deletion reduced total)
- `emotion_algebra/base.py`: 100%
- All 395 tests pass

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
- `emotion_algebra/plutchik.py` — `__bool__` bug fix
- `emotion_algebra/behaviour.py` — `if e is not None:` + `from_data()` guard
- `emotion_algebra/emotions.py` — `if emotion is not None:`
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

---

## 2026-03-30 — Phase 4: Scientific Specification Implementation

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-directed; AI executed all changes.

### Changes

| Area | Change |
|---|---|
| `SPECIFICATION.md` | Created — normative scientific contract for the library (Cambria + Plutchik + Russell) |
| `plutchik.py` | `Emotion.valence` now returns Pleasantness component only (anger → 0, joy → +2) |
| `plutchik.py` | `Emotion.arousal` new property: `abs(emotional_flow)` |
| `plutchik.py` | `Emotion.type` rewritten: Russell (1980) Circumplex (6 categories, not heuristic) |
| `plutchik.py` | `_circumplex_type()` module-level helper; shared by all `type` properties |
| `plutchik.py` | `EmotionalDimension.valence`: Sensitivity=0, Attention=0 (reactivity ⊥ hedonics) |
| `plutchik.py` | `EmotionalDimension.kind`: "hedonic" / "activation" (replaces "positive"/"negative") |
| `plutchik.py` | `Emotion.__add__` cross-axis: returns `CompositeEmotion` (not `Feeling`) |
| `composite_emotions.py` | `CompositeEmotion.valence`: sum of Pleasantness components |
| `composite_emotions.py` | `CompositeEmotion.arousal`: max `|e.emotional_flow|` across components |
| `composite_emotions.py` | `CompositeEmotion.type`: Russell Circumplex |
| `feelings.py` | `Feeling.valence`: sum of `e.valence` (Pleasantness only) |
| `feelings.py` | `Feeling.arousal`: new property — max `e.arousal` across components |
| `feelings.py` | `Feeling.type`: new property — Russell Circumplex |
| Tests | Updated 8 tests to reflect spec; added 13 new Circumplex/arousal tests |

### Scientific justification
- Valence = Pleasantness only: Posner et al. (2005) — arousal and valence are orthogonal in all validated models
- Arousal = |flow|: Russell (1980) Circumplex arousal dimension
- Type = Circumplex quadrants: Russell (1980), Barrett & Russell (1999)
- Sensitivity=0 valence: Öhman (1986) — anger is approach, fear is avoidance; neither is inherently pleasant

### Post-fix: 395 tests, 0 failures

---

## 2026-03-30 — v1.0.0 Pre-release: Rename & Audit

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-directed; AI executed all changes.

### Actions Taken

1. **Renamed package** `emotion_data` → `emotion-algebra` (`emotion_algebra`)
   - Package directory renamed: `emotion_data/` → `emotion_algebra/`
   - All imports updated throughout `emotion_algebra/`, `test/`, `examples/`
   - `pyproject.toml`: `name = "emotion-algebra"`, `version = "1.0.0"`, `license = "Apache-2.0"`, `readme = "README.md"`
   - `.coveragerc` updated to omit `emotion_algebra/deepmoji.py` and `emotion_algebra/tag.py`

2. **Wrote new `README.md`**
   - Installation, model overview (axes table, valence/arousal/type), quick reference
   - Scientific references: Plutchik 1980, Cambria 2012, Russell 1980, Posner 2005
   - Replaced old `readme.md` (donation badges, outdated 2018 blog link)

3. **Fresh pre-release `AUDIT.md`**
   - 6 open issues (A-001 to A-006), 17 resolved issues (R-001 to R-017)
   - All Phase 1–4 resolutions consolidated into resolved table
   - Security section updated with accurate optional-dep call paths

4. **`TODO.md`** updated: A-002, A-003, A-005 marked resolved.

### Test results
441 tests, 0 failures, 94% coverage (core algebraic modules 94–100%).

---

## 2026-03-30 — v1.1–v2.0 Feature Implementation

**AI Model**: claude-sonnet-4-6
**Oversight**: Human-directed ("do it all"); AI executed all changes.

### New modules

| Module | Milestone | Summary |
|---|---|---|
| `emotion_algebra/distance.py` | v1.3 | `emotion_distance`, `closest_emotion`, `emotion_clusters` |
| `emotion_algebra/state.py` | v1.1 | `EmotionalState` (mutable, weighted, decaying), `EmotionTimeline` |
| `emotion_algebra/text.py` | v1.2 | `from_text`, `score_text`, `HFEmotionAdapter` |
| `emotion_algebra/appraisal.py` | v1.5 | `Appraisal` dataclass, `appraisal_to_emotion` (Scherer CPM) |
| `emotion_algebra/float_emotion.py` | v2.0 | `FloatEmotion`, `from_embedding`, `from_emotion` |
| `emotion_algebra/__main__.py` | v1.4 | CLI: `info`, binary expressions, interactive REPL |

### Modified

- `plutchik.py`, `feelings.py`, `composite_emotions.py`: `to_dict`/`from_dict` serialization on all three concrete types
- `__init__.py`: all new public symbols + new `EmotionAnalyzer` methods (`analyze_text`, `score_text`, `distance`, `appraise`)
- `pyproject.toml`: `[transformers]` extra; `emotion-algebra` CLI entry point

### Test results
596 tests, 0 failures, 93% coverage.
