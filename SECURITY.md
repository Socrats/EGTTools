# Security policy

## Supported versions

Security fixes are made on the `master` branch and released in the next version. Only the
latest release on PyPI and conda receives fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/Socrats/EGTTools/security/advisories/new)
(Security tab → *Report a vulnerability*). Do not open a public issue.

Include a description of the problem, the affected versions, and steps or code to reproduce
it. You will receive an acknowledgement, and the fix will be coordinated with you before any
public disclosure.

EGTtools is a scientific library that runs locally on data you provide. Reports are most
relevant when they concern crashes or memory errors that can be triggered through the public
API, problems in the published packages, or issues in the build and release process.

## Keeping personal and secret information out of the project

The repository and the published packages must contain nothing specific to a contributor's
machine or accounts: no absolute home-directory paths (`/Users/<name>/…`, `/home/<name>/…`), no
credentials (tokens, keys, passwords), and no personal tool or environment files (virtual
environments, IDE settings, assistant or editor configuration, caches, OS metadata).

How this is enforced:

1. **Ignore rules.** `.gitignore` excludes generic tool and environment files. Keep files that are
   specific to your own setup in your local `.git/info/exclude` instead, so the public repository
   does not list them.
2. **Automatic check.** `scripts/check_repo_hygiene.py` fails when a tracked file matches an ignore
   rule (including your local excludes), lives in a tool or environment directory, contains an
   absolute home path, or contains a credential. It runs in CI on every push (job *Repository
   hygiene*) and in the pre-push hook (`scripts/install_hooks.sh`). With `--artifacts` it checks
   built wheels, sdists and conda packages.
3. **Notebook outputs.** Clear or check outputs before committing a notebook: outputs saved from a
   local run can include local paths.
4. **Builds outside the source tree.** Build in a separate directory, never in the source tree.
5. **Credentials** live only in GitHub Actions secrets or trusted publishing, never in files.
