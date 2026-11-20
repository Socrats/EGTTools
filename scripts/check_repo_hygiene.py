#!/usr/bin/env python
"""Check that the repository and built packages contain no personal information.

    python scripts/check_repo_hygiene.py                  # tracked files of the repository
    python scripts/check_repo_hygiene.py --artifacts dist/*.whl dist/*.tar.gz pkgs/*.conda

Tracked files must not:
- match an ignore rule (.gitignore, or a contributor's own .git/info/exclude: files kept out of the
  repository on purpose must not be committed anyway);
- live in tool or environment directories (virtual environments, IDE settings, caches, OS metadata);
- contain absolute home-directory paths (e.g. ``/Users/<name>/``, ``/home/<name>/``,
  ``C:\\Users\\<name>\\``), which identify a contributor's machine.

Built packages (wheels, sdists, conda packages, directories) are checked for home paths. Paths of
CI runners and documentation placeholders are allowed. Credentials are left to the repository's
secret scanning and push protection.

Exit status 1 lists every finding; 0 means clean. Used by CI and by scripts/hooks/pre-push.
Findings give the file, the line and the kind of match, never the matched text: the output can
end up in public CI logs, which must not repeat what the check is meant to keep out.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# Tool and environment directories/files that are personal and never part of the project.
FORBIDDEN_PATH = re.compile(
    r"(^|/)(\.venv|venv|\.idea|\.vscode|__pycache__|\.ipynb_checkpoints|\.mypy_cache|\.pytest_cache)(/|$)"
    r"|(^|/)\.DS_Store$|\.pyc$|(^|/)\.env$")

# Home directories, except CI runners and placeholders used in documentation.
_ALLOWED_NAMES = r"(?:runner|runneradmin|user|username|USER|USERNAME|you|me|name|\.\.\.|<[^>/]*>|\$USER|\$\{USER\}|~)"
HOME_PATH = re.compile(
    rb"(?:/Users|/home)/(?!" + _ALLOWED_NAMES.encode() + rb"/)[A-Za-z0-9._-]+/"
    rb"|[A-Za-z]:\\{1,2}Users\\{1,2}(?!(?:runneradmin|Public|Default|USERNAME|<[^>\\]*>)\\)[A-Za-z0-9._-]+\\")

MAX_BYTES = 20 * 1024 * 1024  # skip content checks for very large blobs


def scan_bytes(data: bytes):
    """Yield (kind, line number) for every home path in ``data``.

    Only the position is returned, never the matched text, so that it cannot be printed.
    """
    for match in HOME_PATH.finditer(data):
        yield "home path", data.count(b"\n", 0, match.start()) + 1


def tracked_files(root: Path):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [name for name in out.decode().split("\0") if name]


def tracked_but_ignored(root: Path):
    out = subprocess.run(["git", "ls-files", "-z", "-ci", "--exclude-standard"], cwd=root,
                         capture_output=True, check=True).stdout
    return [name for name in out.decode().split("\0") if name]


def check_repository(root: Path):
    findings = []
    for name in tracked_but_ignored(root):
        findings.append((name, "tracked but matches an ignore rule"))
    for name in tracked_files(root):
        path = root / name
        if FORBIDDEN_PATH.search(name):
            findings.append((name, "tool or environment file"))
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
            continue
        for kind, lineno in scan_bytes(path.read_bytes()):
            findings.append((f"{name}:{lineno}", kind))
    return findings


def _members(path: Path):
    """Yield (member name, bytes) for files inside a wheel, sdist, conda package or directory."""
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.stat().st_size <= MAX_BYTES:
                yield str(file.relative_to(path)), file.read_bytes()
    elif path.suffix in (".whl", ".zip"):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                yield name, archive.read(name)
    elif path.suffix == ".conda":  # zip of zstd-compressed tarballs
        with zipfile.ZipFile(path) as archive:
            for inner in archive.namelist():
                if inner.endswith(".tar.zst"):
                    yield from _tar_members(_unzstd(archive.read(inner)), prefix=inner + ":")
                else:
                    yield inner, archive.read(inner)
    elif path.name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(path) as tar:
            yield from _tar_members(tar)
    else:
        yield path.name, path.read_bytes()


def _tar_members(tar_or_bytes, prefix=""):
    tar = tarfile.open(fileobj=io.BytesIO(tar_or_bytes)) if isinstance(tar_or_bytes, bytes) else tar_or_bytes
    with tar:
        for member in tar.getmembers():
            if member.isfile() and member.size <= MAX_BYTES:
                yield prefix + member.name, tar.extractfile(member).read()


def _unzstd(data: bytes) -> bytes:
    try:
        import zstandard
        return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    except ImportError:
        if shutil.which("zstd") is None:
            raise SystemExit("checking .conda packages needs the 'zstandard' module or the 'zstd' command")
        return subprocess.run(["zstd", "-dc"], input=data, capture_output=True, check=True).stdout


def check_artifacts(paths):
    findings = []
    for path in map(Path, paths):
        for member, data in _members(path):
            for kind, lineno in scan_bytes(data):
                findings.append((f"{path.name}:{member}:{lineno}", kind))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifacts", nargs="+", metavar="PATH", help="check built packages instead of the repository")
    args = parser.parse_args()
    if args.artifacts:
        findings = check_artifacts(args.artifacts)
    else:
        root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                                   text=True, check=True).stdout.strip())
        findings = check_repository(root)
    for where, kind in findings:
        print(f"{where}: {kind}")
    if findings:
        print(f"\n{len(findings)} finding(s). Personal paths and tool files must not be committed or "
              f"published; see SECURITY.md.", file=sys.stderr)
        return 1
    print("repository hygiene: clean" if not args.artifacts else "artifacts: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
