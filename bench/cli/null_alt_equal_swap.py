import argparse
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_SEG, DST_SEG = "null-alt-equal", "null-alt-equal-swap"


def dest_of(path, src_run, dst_run):
    return dst_run / pathlib.Path(path).relative_to(src_run)


def rewrite(job):
    src_report, dst_report = map(pathlib.Path, job)
    from phlag import phlag as ph
    argv = shlex.split(src_report.read_text().splitlines()[0])
    i = argv.index("--np")
    argv[i + 1] = DST_SEG
    j = argv.index("--output-base")
    argv[j + 1] = re.sub(rf"/{SRC_SEG}/", f"/{DST_SEG}/", argv[j + 1], count=1)
    sys.argv = argv
    args = ph.parse_arguments(argv[1:] + ["-o", str(dst_report), "--plot", "none"])
    for k in ("rho", "beta", "beta_prime"):
        v = getattr(args, k, None)
        if isinstance(v, list) and len(v) == 1:
            setattr(args, k, v[0])
    p = ph.Phlag(args)
    p.load_fit(ph.fit_path_for(src_report))
    p.save_output()
    return str(dst_report)


def find_runs(root):
    return sorted(a.parent for a in pathlib.Path(root).glob(f"**/{SRC_SEG}/**/args.json")
                  if json.loads(a.read_text()).get("null_emission_parameterization") == SRC_SEG)


def dest_run_of(src_run, root, out_root):
    rel = src_run.relative_to(root).parts
    k = rel.index(SRC_SEG)
    return pathlib.Path(out_root, *rel[:k], DST_SEG, *rel[k + 1:])


def main():
    ap = argparse.ArgumentParser(description=f"Backfill --np {DST_SEG} runs from existing {SRC_SEG} fits (no EM rerun).")
    ap.add_argument("--root", default="store/phlag")
    ap.add_argument("--out-root", default=None, help="Defaults to --root; set elsewhere to test.")
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--no-resummarize", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Max reports per run (testing).")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    out_root = pathlib.Path(args.out_root or args.root)

    runs = find_runs(root)
    jobs, dst_runs = [], []
    for src_run in runs:
        dst_run = dest_run_of(src_run, root, out_root)
        dst_runs.append((src_run, dst_run))
        dst_run.mkdir(parents=True, exist_ok=True)
        a = json.loads((src_run / "args.json").read_text())
        a["null_emission_parameterization"] = DST_SEG
        (dst_run / "args.json").write_text(json.dumps(a, indent=2) + "\n")
        if (src_run / "source").is_dir() and not (dst_run / "source").exists():
            shutil.copytree(src_run / "source", dst_run / "source")
        reports = sorted(p for p in (src_run / "reports").rglob("*.tsv"))[: args.limit]
        for r in reports:
            d = dest_of(r, src_run, dst_run)
            if not (d.exists() and d.with_suffix(".fit.pkl").exists()):
                jobs.append((str(r), str(d)))
    print(f"{len(runs)} runs, {len(jobs)} reports to write")

    env_threads = {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
                   "OMP_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu"}
    os.environ.update(env_threads)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(rewrite, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                failed += 1
                print(f"[fail] {futs[f][0]}: {e!r}", file=sys.stderr)
            if n % 200 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} ({failed} failed)", flush=True)

    if args.no_resummarize or out_root.resolve() != root.resolve():
        return
    for src_run, dst_run in dst_runs:
        a = json.loads((dst_run / "args.json").read_text())
        cmd = ["benchmark", "--create", str(dst_run), "-w", str(a["window_size"]), "-s", str(a["step_size"]),
               "--np", DST_SEG, "--resummarize"]
        for k in ("rho", "beta", "beta_prime"):
            if a.get(k) is not None:
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
