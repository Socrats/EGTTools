# Releasing EGTtools

This checklist describes how a version of EGTtools is released. It is written for maintainers.

## Versions

- Versions follow `MAJOR.MINOR.PATCH` and stay at `0.x` while the API may still change.
- The version is derived from git tags by setuptools-scm: a tag `vX.Y.Z` releases `X.Y.Z`, and a
  tag `vX.Y.ZrcN` a release candidate. Commits after a tag build as `X.Y.(Z+1).devN`.
- Historical tags of the form `vX.Y.Z-patchN` are read as `X.Y.Z`.

## Before the release

1. Make sure the **Tests** and **Wheels** workflows are green on `master`.
2. In `CHANGELOG.md`, rename *Unreleased* to `[X.Y.Z] - YYYY-MM-DD`, check that every
   user-visible change, breaking change and change in numerical results is listed, and add
   migration notes for breaking changes.
3. Check that the documentation builds and the examples run.
4. For larger releases, tag a release candidate first (`vX.Y.Zrc1`) and test the wheels from it.

## Release

1. Tag the release commit on `master` and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "EGTtools X.Y.Z"
   git push origin vX.Y.Z
   ```

2. The tag triggers the **Wheels** workflow, which builds the sdist and the wheels for every
   platform and creates a **draft** GitHub release with them attached. The **Conda Build**
   workflow also runs for the tag.
3. Review the draft release (version, files, release notes taken from the CHANGELOG) and publish it.
4. Run the **Publish to Pypi** workflow manually: first with `testpypi`, check that
   `pip install -i https://test.pypi.org/simple/ egttools==X.Y.Z` works, then with `pypi`.
5. Upload the conda packages by running **Conda Build** manually with *upload* enabled.
6. Check that Read the Docs built the documentation for the new version.

## After the release

1. Add a new empty *Unreleased* section at the top of `CHANGELOG.md`.
2. Update `CITATION.cff` if the citation details changed.
