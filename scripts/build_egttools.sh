#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  rebuild_egttools.sh [--python /path/to/python] [--clone-if-missing] /path/to/EGTTools [git-ref]

Arguments:
  /path/to/EGTTools   Path to the local EGTTools source tree.
  [git-ref]           Optional git branch, tag, or commit to check out before rebuilding.

Options:
  --python PATH       Python executable to use for all checks and installation.
  --clone-if-missing  Clone EGTTools into the given path if it does not exist yet.
  -h, --help          Show this help message.

What this script does:
  - checks required external tools
  - checks Python package requirements declared by EGTTools
  - checks that vcpkg exists inside EGTTools
  - checks for OpenMP references and warns about ambiguous hard-coded library paths
  - checks the local git state before attempting any fetch/pull
  - updates the git checkout and submodules when it is safe to do so
  - bootstraps vcpkg if available
  - builds and installs EGTTools into the selected Python interpreter
  - prints where the package is being installed

What this script does NOT do:
  - it does not install missing system tools
  - it does not install missing Python requirements listed in requirements.txt
USAGE
}

prompt_yes_no() {
  local prompt="$1"
  while true; do
    printf '%s [Yes/No]: ' "$prompt"
    read -r answer
    case "$answer" in
    Yes | yes | Y | y)
      return 0
      ;;
    No | no | N | n | "")
      return 1
      ;;
    *)
      echo "Please answer Yes or No."
      ;;
    esac
  done
}

warn_merge_then_exit() {
  echo "WARNING: There appear to be merge or checkout issues in the repository."
  echo "Please merge the new changes before continuing."
  exit 1
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "MISSING TOOL: $cmd"
    return 1
  fi
  return 0
}

check_python_requirements() {
  local python_exe="$1"
  local req_file="$2"
  if [[ ! -f "$req_file" ]]; then
    return 0
  fi

  "$python_exe" - "$req_file" <<'PY'
import re
import sys
from pathlib import Path

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore

req_path = Path(sys.argv[1])
missing = []
ignored_prefixes = ("-r", "--", "git+", "http://", "https://", "-e ")

for raw in req_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith(ignored_prefixes):
        continue
    line = line.split("#", 1)[0].strip()
    if not line:
        continue

    name = re.split(r"[<>=!~;\\[]", line, maxsplit=1)[0].strip()
    if not name:
        continue

    try:
        importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        missing.append(line)

if missing:
    print("MISSING PYTHON PACKAGES:")
    for pkg in missing:
        print(f"  - {pkg}")
    sys.exit(2)
PY
}

check_openmp_references() {
  local root="$1"
  local files_found
  local suspicious
  local generic

  files_found=$(grep -RIlE 'OpenMP|libomp|libgomp|omp\.h|find_package\(OpenMP' \
    "$root/CMakeLists.txt" "$root/cmake" "$root/src" "$root/cpp" 2>/dev/null || true)

  suspicious=$(grep -RInE '/opt/homebrew|/usr/local/opt/libomp|/opt/local|libomp\.dylib|libgomp\.so' \
    "$root/CMakeLists.txt" "$root/cmake" "$root/src" "$root/cpp" 2>/dev/null || true)

  generic=$(grep -RInE 'find_package\(OpenMP|OpenMP::OpenMP|ENABLE_OPENMP|openmp' \
    "$root/CMakeLists.txt" "$root/cmake" "$root/src" "$root/cpp" 2>/dev/null || true)

  if [[ -z "$files_found" ]]; then
    echo "OPENMP CHECK: no OpenMP-related references were found in common build files."
    return 2
  fi

  if [[ -n "$suspicious" ]]; then
    echo "OPENMP CHECK: suspicious or machine-specific OpenMP library references were found:"
    echo "$suspicious"
    return 3
  fi

  if [[ -z "$generic" ]]; then
    echo "OPENMP CHECK: OpenMP references were found, but not a clear generic CMake/OpenMP integration."
    echo "$files_found"
    return 4
  fi

  echo "OPENMP CHECK: generic OpenMP references were found."
  return 0
}

PYTHON_EXE=""
CLONE_IF_MISSING=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
  --python)
    if [[ $# -lt 2 ]]; then
      echo "ERROR: --python requires a path argument." >&2
      usage
      exit 1
    fi
    PYTHON_EXE="$2"
    shift 2
    ;;
  --clone-if-missing)
    CLONE_IF_MISSING=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --)
    shift
    while [[ $# -gt 0 ]]; do
      POSITIONAL+=("$1")
      shift
    done
    ;;
  -*)
    echo "ERROR: unknown option: $1" >&2
    usage
    exit 1
    ;;
  *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 1 || ${#POSITIONAL[@]} -gt 2 ]]; then
  usage
  exit 1
fi

TARGET_PATH="${POSITIONAL[0]}"
GIT_REF="${POSITIONAL[1]:-}"

missing_any=0
for cmd in git cmake; do
  if ! require_cmd "$cmd"; then
    missing_any=1
  fi
done

if [[ -z "$PYTHON_EXE" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
  else
    echo "MISSING TOOL: python3"
    missing_any=1
  fi
fi

if [[ -n "$PYTHON_EXE" && ! -x "$PYTHON_EXE" ]]; then
  echo "ERROR: Python executable is not executable: $PYTHON_EXE" >&2
  exit 1
fi

if [[ -n "$PYTHON_EXE" ]] && ! "$PYTHON_EXE" -m pip --version >/dev/null 2>&1; then
  echo "MISSING TOOL: $PYTHON_EXE -m pip"
  missing_any=1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "WARNING: ninja not found. Build may still work with another CMake generator, but ninja is recommended."
fi

if [[ "$missing_any" -ne 0 ]]; then
  echo ""
  echo "Aborting because one or more required external tools are missing."
  exit 1
fi

if [[ ! -e "$TARGET_PATH" ]]; then
  if [[ "$CLONE_IF_MISSING" -eq 1 ]]; then
    echo "Target path does not exist. Cloning EGTTools into: $TARGET_PATH"
    git clone https://github.com/Socrats/EGTTools.git "$TARGET_PATH"
  else
    echo "ERROR: EGTTools source directory does not exist: $TARGET_PATH" >&2
    echo "Use --clone-if-missing to clone it automatically."
    exit 1
  fi
fi

EGTTOOLS_SRC_DIR="$(cd "$TARGET_PATH" && pwd)"

if [[ ! -f "$EGTTOOLS_SRC_DIR/CMakeLists.txt" ]]; then
  echo "ERROR: This does not look like an EGTTools source tree (missing CMakeLists.txt): $EGTTOOLS_SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$EGTTOOLS_SRC_DIR/.git" ]]; then
  echo "ERROR: The source directory is not a git repository: $EGTTOOLS_SRC_DIR" >&2
  exit 1
fi

PIP_CMD=("$PYTHON_EXE" -m pip)

"$PYTHON_EXE" - <<'PY'
import site
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version   : {sys.version.split()[0]}")
print(f"User site        : {site.getusersitepackages()}")
print(f"Prefix           : {sys.prefix}")
PY

if [[ -f "$EGTTOOLS_SRC_DIR/requirements.txt" ]]; then
  echo "Checking Python requirements declared in requirements.txt"
  if ! check_python_requirements "$PYTHON_EXE" "$EGTTOOLS_SRC_DIR/requirements.txt"; then
    echo ""
    echo "Aborting because some Python requirements declared by EGTTools are not installed."
    echo "They were not installed automatically."
    exit 1
  fi
else
  echo "WARNING: No requirements.txt found in $EGTTOOLS_SRC_DIR"
fi

VCPKG_DIR="$EGTTOOLS_SRC_DIR/vcpkg"
VCPKG_BOOTSTRAP_SH="$VCPKG_DIR/bootstrap-vcpkg.sh"
VCPKG_BOOTSTRAP_BAT="$VCPKG_DIR/bootstrap-vcpkg.bat"

if [[ ! -d "$VCPKG_DIR" ]]; then
  echo "MISSING VCPKG: expected directory not found at $VCPKG_DIR"
  exit 1
fi

if [[ ! -f "$VCPKG_BOOTSTRAP_SH" && ! -f "$VCPKG_BOOTSTRAP_BAT" ]]; then
  echo "MISSING VCPKG BOOTSTRAP: expected bootstrap-vcpkg.sh or bootstrap-vcpkg.bat inside $VCPKG_DIR"
  exit 1
fi

if ! check_openmp_references "$EGTTOOLS_SRC_DIR"; then
  rc=$?
  echo ""
  case "$rc" in
  2)
    echo "No OpenMP reference was found, so the build configuration may be incomplete."
    ;;
  3 | 4)
    echo "Since that path to the OpenMP library is ambiguous, we recommend you set it with export EGTTOOLS_EXTRA_CMAKE_ARGS=\"-DLIBOMP_DIR='/path/to/openmplib/'\" before continuing."
    ;;
  *)
    echo "The OpenMP check reported a warning."
    ;;
  esac

  if ! prompt_yes_no "Would you like to continue?"; then
    echo "Stopping at user request."
    exit 1
  fi
fi

cd "$EGTTOOLS_SRC_DIR"

SKIP_FETCH_AND_PULL=0
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STATUS_PORCELAIN="$(git status --porcelain || true)"
HAS_UNCOMMITTED=0
if [[ -n "$STATUS_PORCELAIN" ]]; then
  HAS_UNCOMMITTED=1
fi

UPSTREAM_REF=""
HAS_UPSTREAM=0
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  UPSTREAM_REF="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}')"
  HAS_UPSTREAM=1
fi

AHEAD_COUNT=0
BEHIND_COUNT=0
if [[ "$HAS_UPSTREAM" -eq 1 ]]; then
  counts="$(git rev-list --left-right --count HEAD...@{u} 2>/dev/null || echo '0 0')"
  AHEAD_COUNT="$(awk '{print $1}' <<<"$counts")"
  BEHIND_COUNT="$(awk '{print $2}' <<<"$counts")"
fi

if [[ "$HAS_UNCOMMITTED" -eq 1 || "$HAS_UPSTREAM" -eq 0 || "$AHEAD_COUNT" -gt 0 ]]; then
  echo "WARNING: local git state suggests that updating from the remote may not be safe."
  if [[ "$HAS_UNCOMMITTED" -eq 1 ]]; then
    echo "- The repository has uncommitted changes."
  fi
  if [[ "$HAS_UPSTREAM" -eq 0 ]]; then
    echo "- The current branch has no configured upstream."
  fi
  if [[ "$AHEAD_COUNT" -gt 0 ]]; then
    echo "- The current branch is ahead of its upstream by $AHEAD_COUNT commit(s)."
  fi
  echo "The script can continue building from the current local checkout without fetching."
  if prompt_yes_no "Would you like to continue?"; then
    SKIP_FETCH_AND_PULL=1
  else
    echo "Stopping at user request."
    exit 1
  fi
fi

if [[ "$SKIP_FETCH_AND_PULL" -eq 0 ]]; then
  echo "Updating repository"
  if ! git fetch --all --tags; then
    warn_merge_then_exit
  fi

  if [[ -n "$GIT_REF" ]]; then
    echo "Checking out requested ref: $GIT_REF"
    if ! git checkout "$GIT_REF"; then
      warn_merge_then_exit
    fi
  else
    if [[ "$CURRENT_BRANCH" != "HEAD" ]]; then
      echo "Pulling latest changes for current branch: $CURRENT_BRANCH"
      if ! git pull --ff-only; then
        warn_merge_then_exit
      fi
    else
      echo "Detached HEAD detected. No pull performed."
    fi
  fi
else
  echo "Skipping fetch and pull. Building from the current local checkout."
  if [[ -n "$GIT_REF" ]]; then
    echo "Requested git ref '$GIT_REF' was ignored because fetch/pull was skipped for safety."
    echo "Check it out manually if needed before rebuilding."
  fi
fi

echo "Updating submodules"
if ! git submodule update --init --recursive; then
  warn_merge_then_exit
fi

if [[ -f "$VCPKG_BOOTSTRAP_SH" ]]; then
  echo "Bootstrapping vcpkg"
  "$VCPKG_BOOTSTRAP_SH" --disableMetrics
elif [[ -f "$VCPKG_BOOTSTRAP_BAT" ]]; then
  echo "ERROR: Windows bootstrap script detected, but this shell script is running in a POSIX shell."
  echo "Run bootstrap-vcpkg.bat manually on Windows or adapt this script to PowerShell/cmd."
  exit 1
fi

if [[ -x "$VCPKG_DIR/vcpkg" ]]; then
  echo "Running vcpkg install"
  "$VCPKG_DIR/vcpkg" install --x-binarycaching
else
  echo "ERROR: vcpkg executable not found after bootstrap: $VCPKG_DIR/vcpkg"
  exit 1
fi

export VCPKG_PATH="$EGTTOOLS_SRC_DIR"

echo "Building and installing EGTTools into the selected Python environment"
"${PIP_CMD[@]}" install --upgrade pip setuptools wheel
"${PIP_CMD[@]}" install --upgrade .

echo ""
echo "EGTTools install finished."
"$PYTHON_EXE" - <<'PY'
import site
import sys
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore

print(f"Installed with Python : {sys.executable}")
print(f"Python prefix         : {sys.prefix}")
print(f"User site-packages    : {site.getusersitepackages()}")
try:
    dist = importlib_metadata.distribution("egttools")
    print("Installed package path:")
    for p in dist.files or []:
        if str(p).endswith("METADATA"):
            print(f"  {dist.locate_file(p)}")
            break
except Exception:
    print("Installed package path: could not be resolved from importlib metadata.")
PY
