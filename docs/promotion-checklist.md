# Promotion Checklist

Use this checklist before moving any module or feature from `experimental/` into the stable `tensor_cpu` path.

## 1. Scope

- The feature has a clearly defined stable user story.
- The feature does not require parallel IR worlds or duplicate public APIs.
- Any unsupported subcases are documented explicitly.

## 2. Architecture

- The feature is integrated into the existing stable execution path rather than added as a second path.
- Core abstractions are shared with stable modules instead of duplicated.
- Internal ownership is clear: frontend, IR, passes, backend, runtime.

## 3. API

- Stable import location is chosen and documented.
- Experimental-only names are not silently re-exported from `tensor_cpu`.
- Error messages for unsupported cases are explicit and tested.

## 4. Correctness

- Unit tests cover expected behavior and key edge cases.
- Regression tests exist for every previously known bug in this area.
- If the feature affects code generation or runtime behavior, numerical reference tests are included.

## 5. Performance

- A benchmark exists for the promoted path.
- The benchmark reports compile latency and runtime statistics.
- Performance claims are compared against a saved baseline.

## 6. Runtime Constraints

- ABI/runtime assumptions are documented.
- Shape, dtype, thread, and memory constraints are documented.
- Fallback behavior is explicit when the fast path is unavailable.

## 7. Tooling

- Root-level `pytest` covers the promoted feature.
- CI runs at least one smoke test for the promoted feature.
- Docs are updated in `architecture.md`, `execution-path.md`, and `stable-api.md` when needed.

## 8. Exit Rule

Do not promote a feature if any of these remain true:

- It depends on a separate graph or control-flow universe.
- It requires hidden feature flags to be safe.
- It lacks correctness regression coverage.
- It is not connected to the default stable runtime path.
