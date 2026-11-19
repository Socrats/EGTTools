#!/usr/bin/env python
"""Run a command and kill it if its memory use exceeds a limit.

    python scripts/memguard.py [--limit-gb 4] [--interval 0.2] -- <command> [args...]

Example:

    python scripts/memguard.py --limit-gb 4 -- python -m pytest tests/test_state_counts.py

The guard polls the resident memory (RSS) of the command and all its child processes. When
the total exceeds the limit, it kills the whole process tree, reports the peak, and exits with
code 99. Otherwise it exits with the command's own exit code.

Use it whenever a run may allocate without bound, for example when tests exercise code with a
known size bug. A watchdog inside the process does not work for this: a large allocation in C++
code that holds the GIL blocks every Python thread, and macOS does not enforce RLIMIT_AS.

The limit can also be set with the environment variable MEMGUARD_LIMIT_GB.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

GUARD_EXIT_CODE = 99

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for environments without psutil
    psutil = None


def tree_rss_bytes(pid):
    """Total resident memory of a process and all its descendants, in bytes."""
    if psutil is not None:
        try:
            root = psutil.Process(pid)
            processes = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            return 0
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    # Fallback: parse `ps` (POSIX only); RSS is reported in KiB.
    output = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children, rss = {}, {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        child, parent, kib = (int(field) for field in fields)
        children.setdefault(parent, []).append(child)
        rss[child] = kib * 1024
    total, stack = 0, [pid]
    while stack:
        current = stack.pop()
        total += rss.get(current, 0)
        stack.extend(children.get(current, []))
    return total


def kill_tree(process):
    """Kill the process group started for the command (the command and its children)."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            if psutil is not None:
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit-gb", type=float, default=float(os.environ.get("MEMGUARD_LIMIT_GB", 4)),
                        help="memory limit in GB for the command and its children (default: 4)")
    parser.add_argument("--interval", type=float, default=0.2, help="polling interval in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; use: memguard.py [--limit-gb N] -- <command>")

    limit = int(args.limit_gb * 1024 ** 3)
    # A new session puts the command in its own process group, so it can be killed as a whole.
    process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
    peak = 0
    try:
        while process.poll() is None:
            current = tree_rss_bytes(process.pid)
            peak = max(peak, current)
            if current > limit:
                kill_tree(process)
                process.wait()
                print(f"\nmemguard: killed `{' '.join(command)}`: memory {current / 1024 ** 3:.2f} GB "
                      f"exceeded the limit of {args.limit_gb:g} GB", file=sys.stderr)
                return GUARD_EXIT_CODE
            time.sleep(args.interval)
    except KeyboardInterrupt:
        kill_tree(process)
        process.wait()
        raise
    print(f"memguard: peak memory {peak / 1024 ** 3:.2f} GB (limit {args.limit_gb:g} GB)", file=sys.stderr)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
