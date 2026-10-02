import argparse
import collections
import datetime
import os
import pathlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from bench.cli.split_script import plan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run bench/script.sh's split runs, one progress bar per script line, output in per-run logs.")
    ap.add_argument("script", nargs="?", default="bench/script.sh")
    ap.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("logs/batch"))
    args = ap.parse_args(argv)

    parallel, lines, planned = plan(args.script)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "runs.tsv").write_text("".join(f"{name}\t{cmd}\n" for _, name, cmd in planned))

    print(f"{len(planned)} runs from {len(lines)} line(s), {parallel} at a time -- logs in {args.log_dir}/")
    per_line = collections.Counter(line for line, _, _ in planned)
    width = min(60, max(len(text) for text in lines))
    bars = {
        line: tqdm(total=per_line[line], desc=f"{line:02d} {lines[line - 1][:width]:<{width}}",
                   position=pos, unit="run", leave=True, dynamic_ncols=True)
        for pos, line in enumerate(sorted(per_line))
    }
    running = collections.Counter()
    failed = collections.Counter()
    failures = []
    lock = threading.Lock()
    started = datetime.datetime.now()
    state = {name: "pending" for _, name, _ in planned}
    summary_path = args.log_dir / "00-summary.txt"

    def write_summary(done):
        counts = collections.Counter(state.values())
        head = [f"batch summary ({'finished' if done else 'in progress'})",
                f"script: {args.script}",
                f"started: {started:%Y-%m-%d %H:%M:%S}",
                f"updated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                f"runs: {len(planned)} total, {counts['done']} done, {counts['failed']} failed, "
                f"{counts['running']} running, {counts['pending']} pending", ""]
        if done:
            body = ["commands run:"] + [f"[{state[name]}] {name}\n    {cmd}" for _, name, cmd in planned]
        else:
            body = [f"[running] {name}\n    {cmd}" for _, name, cmd in planned if state[name] == "running"]
            body = ["running:"] + body if body else []
            failed_now = [f"[failed] {name}" for _, name, _ in planned if state[name] == "failed"]
            body += (["", "failed so far:"] + failed_now) if failed_now else []
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text("\n".join(head + body) + "\n")
        os.replace(tmp, summary_path)

    write_summary(False)

    def refresh(line):
        bars[line].set_postfix_str(f"running={running[line]} failed={failed[line]}", refresh=True)

    def run(item):
        line, name, cmd = item
        with lock:
            running[line] += 1
            state[name] = "running"
            write_summary(False)
            refresh(line)
        log_path = args.log_dir / f"{name}.log"
        with open(log_path, "w") as log:
            code = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=log, stderr=subprocess.STDOUT).returncode
        with lock:
            running[line] -= 1
            state[name] = "failed" if code else "done"
            if code:
                failed[line] += 1
                failures.append((name, code, log_path))
            bars[line].update(1)
            write_summary(False)
            refresh(line)

    for line in bars:
        refresh(line)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(run, planned))
    write_summary(True)
    for bar in bars.values():
        bar.close()

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for name, code, log_path in sorted(failures):
            print(f"  {name} (exit {code}): {log_path}")
    else:
        print(f"\nAll {len(planned)} runs succeeded.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
