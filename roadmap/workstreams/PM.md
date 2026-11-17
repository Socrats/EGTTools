# PM: Project management & community

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: make it easy and predictable for others to report issues and contribute.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| PM-01 | Create `roadmap/` tracking folder and schema | done | P0 | S | 0.2 | — | — |
| PM-02 | Replace markdown issue templates with YAML issue forms + `config.yml` | todo | P1 | S | 0.2 | — | — |
| PM-03 | Add pull request template with checklist | todo | P1 | S | 0.2 | — | — |
| PM-04 | Add `CODEOWNERS`, `SECURITY.md`, `SUPPORT.md` | todo | P1 | S | 0.2 | — | — |
| PM-05 | Rewrite `CONTRIBUTING.md` (dev setup, build, tests, style, sign-off) | todo | P1 | M | 0.2 | REL-01 | — |
| PM-06 | Add `RELEASING.md` (versioning, tagging, PyPI, conda) | todo | P1 | S | 0.2 | — | — |
| PM-07 | Label scheme on GitHub (`type:`, `area:`, `priority:`, `status:`) | ready | P1 | S | 0.2 | — | — |
| PM-08 | Update the 6 outdated GitHub issues (#67, #68, #72, #81, #82, #84) and link them to workstreams | ready | P1 | S | 0.2 | PM-07 | — |
| PM-09 | Branch protection on `master` with required checks | todo | P2 | S | 0.2 | REL-07 | — |
| PM-10 | One tracking issue per workstream, linking its roadmap file | ready | P1 | S | 0.2 | PM-07 | — |

## Notes

- **PM-02**: forms for bug report (version, OS, install method, minimal
  reproducer), feature request, performance problem, and documentation. Send
  usage questions to GitHub Discussions (already enabled) via
  `ISSUE_TEMPLATE/config.yml`.
- **PM-03**: the checklist covers tests added, docs/docstrings, a CHANGELOG
  entry under *Unreleased*, regenerated stubs if the bindings changed, and the
  roadmap task ID.
- **PM-07..10**: there is no GitHub Project board (see `DECISIONS.md`). This
  folder is the source of truth. GitHub issues mirror it at workstream level
  and give the community somewhere to discuss. A task's `Ref` column holds its
  issue number once one exists.
- **PM-09**: only once REL-07's test workflow exists. Earlier, required checks
  would block every merge.
