import argparse
import collections
import datetime
import os
import pathlib
import pwd
import resource
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

from tqdm import tqdm

from bench.cli.benchmark import TOTAL_CORE_BUDGET
from bench.cli.split_script import plan

PACIFIC = ZoneInfo("America/Los_Angeles")
BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total} [{elapsed}<{remaining}{postfix}]"
CAUGHT = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)


def describe_sender(info):
    if info.si_pid == 0:
        return "kernel" + (" (terminal hung up)" if info.si_signo == signal.SIGHUP else "")
    try:
        user = pwd.getpwuid(info.si_uid).pw_name
    except KeyError:
        user = str(info.si_uid)
    chain, pid = [], info.si_pid
    while pid > 1 and len(chain) < 4:
        try:
            cmd = " ".join(pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").split())[:200]
            ppid = int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            chain.append(f"pid {pid} (gone)")
            break
        chain.append(f"pid {pid} [{cmd}]")
        pid = ppid
    return f"user {user}: " + " <- ".join(chain)


def oom_kills():
    try:
        text = pathlib.Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/memory.events").read_text()
    except OSError:
        return "unknown"
    return next((line.split()[1] for line in text.splitlines() if line.startswith("oom_kill ")), "unknown")


def read_progress(path):
    try:
        done, total = map(int, path.read_text().split())
    except (OSError, ValueError):
        return 0.0
    return done / total if total else 0.0


def main(argv=None):
    signal.pthread_sigmask(signal.SIG_BLOCK, CAUGHT)
    ap = argparse.ArgumentParser(description="Run bench/script.sh's split runs, one progress bar per script line, output in per-run logs.")
    ap.add_argument("script", nargs="?", default="bench/script.sh")
    ap.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("logs/batch"))
    args = ap.parse_args(argv)

    parallel, lines, planned = plan(args.script)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "runs.tsv").write_text("".join(f"{name}\t{cmd}\n" for _, name, cmd, _ in planned))
    progress_dir = args.log_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    jobserver = args.log_dir / "jobserver"
    jobserver.unlink(missing_ok=True)
    os.mkfifo(jobserver)
    jobserver_fd = os.open(jobserver, os.O_RDWR)
    os.write(jobserver_fd, b"+" * TOTAL_CORE_BUDGET)

    print(f"{len(planned)} runs from {len(lines)} line(s), {parallel} at a time, "
          f"{TOTAL_CORE_BUDGET} cores shared -- logs in {args.log_dir}/")
    per_line = collections.Counter(line for line, _, _, _ in planned)
    width = min(60, max(len(text) for text in lines))
    bars = {
        line: tqdm(total=per_line[line], desc=f"{line:02d} {lines[line - 1][:width]:<{width}}",
                   position=pos, unit="run", leave=True, dynamic_ncols=True, bar_format=BAR_FORMAT)
        for pos, line in enumerate(sorted(per_line))
    }
    running = collections.Counter()
    finished = collections.Counter()
    failed = collections.Counter()
    failures = []
    lock = threading.Lock()
    acquire_lock = threading.Lock()
    stop = threading.Event()
    started = datetime.datetime.now(PACIFIC)
    state = {name: "pending" for _, name, _, _ in planned}
    summary_path = args.log_dir / "00-summary.txt"
    killed = []
    oom_at_start = oom_kills()

    def write_summary(done):
        counts = collections.Counter(state.values())
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        head = [f"batch summary ({'finished' if done else 'killed' if killed else 'in progress'})",
                f"script: {args.script}",
                f"started: {started:%Y-%m-%d %H:%M:%S %Z}",
                f"updated: {datetime.datetime.now(PACIFIC):%Y-%m-%d %H:%M:%S %Z}",
                f"cpu: {usage.ru_utime + usage.ru_stime:.0f} s",
                f"oom kills at start: {oom_at_start}",
                f"runs: {len(planned)} total, {counts['done']} done, {counts['failed']} failed, "
                f"{counts['running']} running, {counts['pending']} pending", *killed, ""]
        if done:
            body = ["commands run:"] + [f"[{state[name]}] {name}\n    {cmd}" for _, name, cmd, _ in planned]
        else:
            body = [f"[running] {name}\n    {cmd}" for _, name, cmd, _ in planned if state[name] == "running"]
            body = ["running:"] + body if body else []
            failed_now = [f"[failed] {name}" for _, name, _, _ in planned if state[name] == "failed"]
            body += (["", "failed so far:"] + failed_now) if failed_now else []
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text("\n".join(head + body) + "\n")
        os.replace(tmp, summary_path)

    write_summary(False)

    def wait_for_signal():
        info = signal.sigwaitinfo(CAUGHT)
        killed.append(f"killed: {signal.Signals(info.si_signo).name} at "
                      f"{datetime.datetime.now(PACIFIC):%Y-%m-%d %H:%M:%S %Z} by {describe_sender(info)}")
        got = lock.acquire(timeout=5)
        try:
            write_summary(False)
        finally:
            if got:
                lock.release()
        os._exit(128 + info.si_signo)

    threading.Thread(target=wait_for_signal, daemon=True).start()

    def refresh(line):
        bars[line].set_postfix_str(f"runs={finished[line]}/{per_line[line]} running={running[line]} "
                                   f"failed={failed[line]}", refresh=False)
        frac = sum(1.0 if state[name] in ("done", "failed") else
                   read_progress(progress_dir / name) if state[name] == "running" else 0.0
                   for l, name, _, _ in planned if l == line)
        bars[line].update(frac - bars[line].n)
        bars[line].refresh()

    def poll():
        while not stop.wait(2):
            with lock:
                for line in bars:
                    refresh(line)

    def run(item):
        line, name, cmd, tokens = item
        with acquire_lock:
            got = 0
            while got < tokens:
                got += len(os.read(jobserver_fd, tokens - got))
        with lock:
            running[line] += 1
            state[name] = "running"
            write_summary(False)
            refresh(line)
        log_path = args.log_dir / f"{name}.log"
        env = {**os.environ, "PHLAG_JOBSERVER": str(jobserver), "PHLAG_PROGRESS_FILE": str(progress_dir / name)}
        try:
            with open(log_path, "w") as log:
                code = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=log,
                                      stderr=subprocess.STDOUT, env=env,
                                      preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, CAUGHT)).returncode
        finally:
            if tokens:
                os.write(jobserver_fd, b"+" * tokens)
        with lock:
            running[line] -= 1
            finished[line] += 1
            state[name] = "failed" if code else "done"
            if code:
                failed[line] += 1
                failures.append((name, code, log_path))
            write_summary(False)
            refresh(line)

    for line in bars:
        refresh(line)
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(run, planned))
    stop.set()
    poller.join()
    write_summary(True)
    for bar in bars.values():
        bar.close()
    os.close(jobserver_fd)
    jobserver.unlink(missing_ok=True)

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for name, code, log_path in sorted(failures):
            print(f"  {name} (exit {code}): {log_path}")
    else:
        print(f"\nAll {len(planned)} runs succeeded.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
