#!/usr/bin/env bash
# Runs the EGTtools tests under Valgrind memcheck.
# Usage: run.sh [cpp|python]
set -u

mode="${1:-cpp}"
valgrind_args=(--leak-check=full --show-leak-kinds=definite --track-origins=yes)

case "$mode" in
  cpp)
    # Each C++ test program on its own; fail if Valgrind reports an error in any of them.
    status=0
    for program in build/tests/test*; do
      [ -f "$program" ] && [ -x "$program" ] || continue
      echo "=== ${program}"
      valgrind "${valgrind_args[@]}" --error-exitcode=1 "$program" > /dev/null || status=1
    done
    exit "$status"
    ;;
  python)
    # CPython's own allocator confuses Valgrind, so use the system malloc. Python itself
    # produces reports too, so this mode writes a log for inspection instead of failing.
    mkdir -p out
    PYTHONMALLOC=malloc valgrind "${valgrind_args[@]}" --log-file=out/valgrind-python.txt \
      python -m pytest tests -m "not slow" -q
    echo "Valgrind log: out/valgrind-python.txt"
    ;;
  *)
    echo "usage: $0 [cpp|python]" >&2
    exit 2
    ;;
esac
