import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import sys
import argparse
import datetime
import pathlib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from phlag.caster import int_or_abbrev, format_val


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a phlagster -w/--rho/--beta-prime sweep with one (pattern, window, rho, beta') config per core, "
                    "skipping configs whose report.tsv was written at/after --since."
    )
    parser.add_argument("specs", nargs="+")
    parser.add_argument("-w", dest="windows", type=int_or_abbrev, nargs="+", required=True)
    parser.add_argument("--rho", type=float, nargs="+", required=True)
    parser.add_argument("--beta-prime", dest="beta_prime", type=float, nargs="+", required=True)
    parser.add_argument("-d", "--dist-type", dest="dist_type", default="gaussian")
    parser.add_argument("--since", type=datetime.datetime.fromisoformat, default=None,
                        help="Reports older than this are redone (default: only missing reports run).")
    parser.add_argument("-j", "--jobs", type=int, default=64)
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("logs/phlagster_parallel"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_caster(spec, window, dist_type):
    from phlag import caster
    result = caster.main([spec, "-d", dist_type, "-w", str(window), "--plot", "scatter"])
    return pathlib.Path(result[0] if isinstance(result, list) else result)


def expected_scores(spec, window):
    from phlag.utils import get_out_root
    return pathlib.Path(get_out_root()) / spec / f"w{format_val(window)}_s{format_val(window)}" / "scores.tsv"


def report_done(report, since):
    return report.exists() and (since is None or report.stat().st_mtime >= since.timestamp())


def run_phlag(scores, rho, beta_prime, dist_type, log_path):
    cmd = [str(pathlib.Path(sys.executable).parent / "phlagster"), str(scores), "-d", dist_type,
           "--rho", str(rho), "--beta-prime", str(beta_prime)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def main(argv=None):
    args = parse_arguments(argv)
    from phlag.utils import expand_node_specs
    specs = expand_node_specs(args.specs)
    configs = [(r, b) for r in args.rho for b in args.beta_prime]

    scores = {(s, w): expected_scores(s, w) for s in specs for w in args.windows}
    missing = [k for k, p in scores.items() if not p.exists()]
    print(f"[caster] {len(missing)} of {len(scores)} (pattern, window) pairs need scores.tsv, {args.jobs} workers")
    if missing and not args.dry_run:
        with ProcessPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(run_caster, s, w, args.dist_type): (s, w) for s, w in missing}
            for fut in as_completed(futures):
                scores[futures[fut]] = fut.result()

    jobs = []
    for (spec, window), scores_path in sorted(scores.items()):
        for rho, bp in configs:
            report = scores_path.parent / args.dist_type / f"rho{rho}_betaprime{bp}" / "report.tsv"
            if not report_done(report, args.since):
                log = args.log_dir / spec / f"w{window}" / f"rho{rho}_betaprime{bp}.log"
                jobs.append((scores_path, rho, bp, log))
    print(f"[phlag] {len(jobs)} pending of {len(scores) * len(configs)} configs")
    if args.dry_run:
        for scores_path, rho, bp, _ in jobs:
            print(f"  {scores_path.parent} rho={rho} beta'={bp}")
        return

    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_phlag, s, r, b, args.dist_type, log): (s, r, b, log) for s, r, b, log in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            s, r, b, log = futures[fut]
            ok = fut.result() == 0
            if not ok:
                failed.append(log)
            print(f"[{i}/{len(jobs)}] {'ok' if ok else 'FAIL'} {s.parent} rho={r} beta'={b}", flush=True)
    if failed:
        print(f"{len(failed)} failed; logs:\n" + "\n".join(f"  {p}" for p in failed))


if __name__ == "__main__":
    main()
