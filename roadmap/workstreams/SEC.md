# SEC: Security and privacy

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: nothing specific to a contributor's machine or accounts in the repository or the published
packages (SECURITY.md, "Keeping personal and secret information out of the project").

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| SEC-01 | Bring the current files in line with the hygiene policy | done | P0 | S | 0.3 | — | `9054584c` |
| SEC-02 | Hygiene check (home paths, credentials, tool files, tracked-but-ignored files) in CI and the pre-push hook; strategy in SECURITY.md | done | P0 | S | 0.3 | — | `9054584c` |
| SEC-03 | Check built wheels, sdists and conda packages with `check_repo_hygiene.py --artifacts` before upload | todo | P0 | S | 0.3 | SEC-02 | — |
| SEC-04 | Enable GitHub secret scanning and push protection for the repository | done | P1 | S | 0.3 | — | repository settings, 2026-11-20 |
| SEC-05 | Wheels and conda packages must not ship the source tree as data files (~600 files installed into the environment prefix) | todo | P0 | M | 0.3 | — | — |
| SEC-06 | Release 0.2.1 with the packaging fix (SEC-05); yank 0.2.0rc3 | todo | P0 | S | 0.3 | SEC-01, SEC-03, SEC-05 | — |

## Notes

- **SEC-02**: `scripts/check_repo_hygiene.py`. Tracked files that match an ignore rule are reported,
  which also enforces a contributor's own `.git/info/exclude` without naming its entries in public
  files. `docs/Makefile` matched the CMake `Makefile` ignore rule; an explicit exception was added.
  The pre-push hook's vcpkg check looked for a file that recent vcpkg versions no longer have.
- **SEC-05**: `include_package_data=True` with the `graft` rules of `MANIFEST.in` (meant for the sdist)
  makes scikit-build put `docs/`, `cpp/`, `tests/`, `cmake/` and the pybind11 submodule into the
  wheel's data directory. Fix the wheel contents and compare them before and after.
