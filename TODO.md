# TODO — emotion_data

Issues are grouped by severity. Each item maps to one commit.

---

## Critical (correctness)

- [x] `Emotion.__bool__` returned `numpy.bool_` — TypeError in Python 3.11+  *(fixed: Phase 2)*
- [x] `_get_behaviours()` silently drops negative-flow emotions via `if e:`  *(fixed: Phase 2)*
- [x] `emotion_to_dimension()` same guard bug  *(fixed: Phase 2)*
- [x] `BehavioralReaction.from_data({})` KeyError on missing `behaviour` key  *(fixed: Phase 2)*
- [x] **`__init__.py` shadows `get_emotion`** — `lexicons.get_emotion` (returns string label) overwrites `emotions.get_emotion` (returns `Emotion`). Every method in `EmotionAnalyzer` that calls `get_emotion` silently returns a string instead of an object. Fix: rename to `get_word_emotion` in `lexicons.py` and update all callers.
- [x] **`CompositeEmotion.emotional_flow` is always ≥ 0** — uses `np.linalg.norm`, which is never negative. The `type` property has `if self.emotional_flow < 0` and `== 0` branches that are permanently dead. Fix: compute signed flow as `sum(component.emotional_flow for component in self.components)` to reflect net valence.
- [x] **`copy()` produces aliased mutable state** — all three classes (`Emotion`, `Feeling`, `CompositeEmotion`) use `copy()` in operator implementations. Since `emotions`/`components` are lists, the copy shares the same list; mutations through one alias corrupt the original. Fix: use `deepcopy` in all operator methods that return a modified copy.

---

## Scientific (model correctness)

- [x] **`Emotion.__bool__` semantics are wrong** — currently `False` for zero-flow and `True` only for positive-flow. Negative emotions (fear, sadness) are falsy. An emotion's truthiness should reflect *presence* (non-zero), not *valence*. Fix: `return self.emotional_flow != 0`.
- [x] **`Feeling` and `CompositeEmotion` are the same concept** — both represent multi-emotion composites, both have `emotion_vector`, `emotional_flow`, dimension lookups, and a full operator set. `CompositeEmotion` inherits `Emotion` (wrong — it is not a simple emotion) while `Feeling` is standalone (correct). Fix: extract a shared `_CompositeBase` mixin with the duplicated arithmetic logic; have both classes use it.
- [x] **`EmotionalDimension.valence` assigns fixed valence to dimensions** — sensitivity is always −1, attention always +1, pleasantness/aptitude are 0. This is eye-balled and contradicts Cambria's original scoring. Add a docstring calling this out explicitly as a simplification, not the model.
- [x] **`Emotion.valence` returns `bool`** — the property docstring says "True when non-zero" but scientific valence is a signed scalar (positive/negative/neutral). Fix type annotation and docstring; the property should return `int` (positive/zero/negative) not `bool`.
- [x] **`type` property uses undocumented heuristics throughout** — both `Emotion.type` and `CompositeEmotion.type` contain `# TODO science this`. Add module-level docstrings clearly labelling these as heuristic approximations, not derivations from Plutchik or Cambria.
- [x] **Model provenance is not documented** — the code silently mixes Plutchik's Wheel (dyad names, cone geometry) with Cambria's Hourglass (4-axis signed integer scale). Add a module-level docstring to `plutchik.py` and `composite_emotions.py` explicitly stating which constructs come from which paper, and where the library departs from both.

---

## Architectural (code quality)

- [x] **`string_to_emotion` is duplicated** — defined on `Emotion` as a `@staticmethod`, then copied verbatim into `CompositeEmotion` and `Feeling`. Move to a module-level function in `emotions.py` and import from there.
- [x] **`deepmoji` and `tag` imported unconditionally at module top** — `__init__.py` line 1 imports from `deepmoji.py` which requires torch/torchMoji. If those aren't installed, `import emotion_data` raises `ImportError`. Fix: lazy-import inside the methods that use them.
- [x] **Global dicts are mutable and tests depend on ordering** — `EMOTIONS`, `FEELINGS`, `BEHAVIOURS`, `REACTIONS` are plain `dict`s mutated freely in tests. Wrap with `types.MappingProxyType` after construction to make accidental mutation a `TypeError`.
- [x] **`Emotion.__sub__` has an unreachable string branch** — `if isinstance(other, str): return self.name - other` would raise `TypeError` because string subtraction is invalid Python. The string is already resolved to an emotion object earlier in the same method. Dead code; remove it.
- [x] **`np.matrix` is deprecated** — `as_matrix` uses `np.matrix(...)`. NumPy has deprecated `np.matrix` since 1.15. Replace with a 2×2 `np.ndarray`.
- [x] **`FEELING_NAMES` key `"acknowledgement"` vs `OPPOSITE_FEELINGS_NAMES` key `"acknowledgment"`** — one has a double-e, one doesn't. Inconsistent spelling means the opposite lookup silently fails. Fix: normalise to `"acknowledgement"` throughout.

---

## Documentation

- [x] Update `docs/index.md` to reflect model provenance (Plutchik vs Cambria) and the known limitations (valence oversimplification, `type` heuristics).
- [x] Update `FAQ.md` with all changes from this TODO pass.
- [x] Update `MAINTENANCE_REPORT.md`.

---

## Phase 4 fixes (done)

- [x] `Feeling.emotional_flow` used `np.linalg.norm` — always ≥ 0, same bug as CompositeEmotion *(fixed: Phase 4)*
- [x] `Feeling.__float__` returned `numpy.float64` — DeprecationWarning in Python 3.11+ *(fixed: Phase 4)*
- [x] `numpy` import removed from `feelings.py` (no longer used) *(fixed: Phase 4)*
- [x] `AUDIT.md` created — evidence-based debt register with 7 open issues *(Phase 4)*

## Open (from AUDIT.md)

- [ ] **A-002** `CompositeEmotion` inherits `Emotion` — conceptually wrong hierarchy; defer to next major version
- [ ] **A-005** `CompositeEmotion.__truediv__` / `__floordiv__` return `None` instead of `NotImplemented` for Emotion operands
- [ ] **A-003** `emotions.py` `__main__` debug print block — remove or convert to CLI entry point
- [ ] **A-006** `Feeling` operator suite undocumented and partially untested

---

## Won't Fix (by design)

- **Single valence axis** — `Emotion.valence` is 1D. Full PAD (Pleasure–Arousal–Dominance) would require a redesign of the entire numeric representation. Out of scope.
- **Appraisal theory layer** — behaviours are mapped directly from emotions, skipping cognitive appraisal. Correct implementation requires structured appraisal data that doesn't exist in the codebase.
- **Feeling/CompositeEmotion merge** — a full merge would break the public API. The mixin approach (above) is the pragmatic fix.
