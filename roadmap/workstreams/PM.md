# PM: Project management & community

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: make it easy and predictable for others to report issues and contribute.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| PM-01 | Create `roadmap/` tracking folder and schema | done | P0 | S | 0.2 | — | merge `8a06d9e` |
| PM-02 | Replace markdown issue templates with YAML issue forms + `config.yml` | done | P1 | S | 0.2 | — | branch `pm-community-files` |
| PM-03 | Add pull request template with checklist | done | P1 | S | 0.2 | — | branch `pm-community-files` |
| PM-04 | Add `CODEOWNERS`, `SECURITY.md`, `SUPPORT.md` | done | P1 | S | 0.2 | — | branch `pm-community-files` |
| PM-05 | Rewrite `CONTRIBUTING.md` (dev setup, build, tests, style, sign-off) | done | P1 | M | 0.2 | REL-01 | branch `pm-community-files` |
| PM-06 | Add `RELEASING.md` (versioning, tagging, PyPI, conda) | done | P1 | S | 0.2 | — | branch `pm-community-files` |
| PM-07 | Label scheme on GitHub (`type:`, `area:`, `priority:`, `status:`) + milestones v0.2.0–v0.4.0 | done | P1 | S | 0.2 | — | — |
| PM-08 | Update the 6 outdated GitHub issues (#67, #68, #72, #81, #82, #84) and link them to workstreams | done | P1 | S | 0.2 | PM-07 | — |
| PM-09 | Branch protection on `master` with required checks | dropped | P2 | S | 0.2 | REL-07 | — |
| PM-10 | One tracking issue per workstream, linking its roadmap file | dropped | P1 | S | 0.2 | PM-07 | — |

## Notes

- **PM-02**: forms for bug report (version, OS, install method, minimal
  reproducer), feature request, performance problem, and documentation. Send
  usage questions to GitHub Discussions (already enabled) via
  `ISSUE_TEMPLATE/config.yml`.
- **PM-03**: the checklist covers tests added, docs/docstrings, a CHANGELOG
  entry under *Unreleased*, regenerated stubs if the bindings changed, and the
  roadmap task ID.
- **PM-07..10**: this folder is the source of truth. There is no GitHub Project
  board and there are no tracking issues (see `DECISIONS.md`). Existing GitHub
  issues are linked from the `Ref` column of the task they belong to.
- **PM-08**: #81, #82 and #67 were closed as already resolved. #84 maps to the
  DOC workstream, #72 to CI-04 and #68 to CI-02; all three are in milestone v0.4.0.
- **PM-09**: dropped. Work is merged directly into `master` without PRs, so
  required PR checks would only get in the way. CI on push to `master`
  (REL-07) is the safety net.
- **PM-03**: still useful for future external contributors, even though the
  maintainer doesn't use PRs.
