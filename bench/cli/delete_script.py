import argparse
import contextlib
import datetime
import functools
import io
import itertools
import json
import pathlib
import shutil
import subprocess
import sys

from bench.cli.split_script import read_commands, split_command

BENCHMARK_OWN_DIRS = ("reports", "source")


def benchmark_targets(argv):
    from bench.cli.benchmark import parse_arguments, default_create_path
    from phlag.utils import get_repo_root
    args = parse_arguments(argv)
    run_dir = get_repo_root() / (args.create or default_create_path(args))
    return [run_dir], BENCHMARK_OWN_DIRS


@functools.lru_cache(maxsize=None)
def resolve_phlag_input(spec, variant_tokens=()):
    from phlag.phlag import parse_arguments
    with contextlib.redirect_stdout(io.StringIO()):
        return pathlib.Path(parse_arguments([spec, *variant_tokens]).caster_scores)


def phlag_targets(argv):
    from phlag.phlag import Phlag, build_parser
    from phlag.caster import substitute_ws_in_path
    args = build_parser().parse_args(argv)
    variant_flags = {"pair": "--pair", "zscale": "-z", "normalize": "-n",
                     "norm_eps": "--norm-eps", "ilr": "-i", "exp_minus": "--exp-minus"}
    variant_tokens = tuple(f for d, f in variant_flags.items() if getattr(args, d))
    if args.site is not None:
        variant_tokens += ("--site" if args.site else "--no-site",)
    base = resolve_phlag_input(str(args.caster_scores[0]), variant_tokens)
    if args.window_size is None:
        ws_paths = [base]
    else:
        steps = args.step_size if isinstance(args.step_size, list) else [1.0]
        ws_paths = [substitute_ws_in_path(base, w, s if isinstance(s, int) else max(1, round(s * w)))
                    for w, s in itertools.product(args.window_size, steps)]
    beta_primes = [None] if args.beta else (args.beta_prime or [None])
    dirs = []
    for path, (rho, beta, beta_prime) in itertools.product(
            ws_paths, itertools.product(args.rho or [None], args.beta or [None], beta_primes)):
        run_args = argparse.Namespace(**vars(args))
        run_args.caster_scores, run_args.rho, run_args.beta, run_args.beta_prime = path, rho, beta, beta_prime
        dirs.append(args.output_file.parent if args.output_file else Phlag.get_default_out_dir(argparse.Namespace(args=run_args)))
    return dirs, ()


WHERE_IGNORED_DESTS = ("output_base", "bench", "no_plots", "recompute", "plot")


def where_matches(record, wanted, defaults):
    def value(dest):
        return record.get(dest, defaults.get(dest))
    for dest, want in wanted.items():
        if dest == "step_size":
            steps = {s if isinstance(s, int) else max(1, round(s * value("window_size"))) for s in want}
            if value("step_size") not in steps:
                return False
        elif value(dest) not in (want if isinstance(want, list) else [want]):
            return False
    return True


def where_targets(argv):
    from bench.cli.benchmark import _explicit_dests
    from bench.cli.phlagster import build_parser
    from phlag.utils import PHLAG_SEGMENT_DEFAULTS, get_repo_root
    parser = build_parser(input_file=False)
    ns = parser.parse_args(argv)
    explicit = _explicit_dests(argv, parser)
    if explicit & set(WHERE_IGNORED_DESTS):
        sys.exit(f"delete: --where can't filter on {', '.join(sorted(explicit & set(WHERE_IGNORED_DESTS)))}")
    wanted = {d: getattr(ns, d) for d in explicit}
    wanted.setdefault("step_size", [1.0])
    defaults = {a.dest: a.default for a in parser._actions} | PHLAG_SEGMENT_DEFAULTS
    root = get_repo_root() / "store" / "phlag"
    dirs = sorted(f.parent for f in root.rglob("args.json") if where_matches(json.loads(f.read_text()), wanted, defaults))
    return dirs, wanted


TARGETS = {"benchmark": benchmark_targets, "phlag": phlag_targets}


def own_outputs(run_dir, own_dirs):
    if not run_dir.is_dir():
        return []
    return sorted(p for p in run_dir.iterdir() if p.is_file() or p.is_symlink() or p.name in own_dirs)


def running_jobs():
    out = subprocess.run(["pgrep", "-af", r"bin/(benchmark|phlag|caster|phlagster)|bench/cli/batch\.sh|-m (phlag|bench)\."],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "delete_script" not in line]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Delete the outputs bench/script.sh's runs would write, so batch regenerates them. Caster scores are kept.")
    ap.add_argument("script", nargs="?", default="bench/script.sh")
    ap.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation.")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Only log what would be deleted.")
    ap.add_argument("--where", action="store_true",
                    help="Everything after --where is phlagster flags (same defaults, so -s 1.0): delete every store/phlag "
                         "run whose args.json matches all of them, instead of reading script.")
    argv = sys.argv[1:] if argv is None else argv
    where_argv = argv[argv.index("--where") + 1:] if "--where" in argv else None
    args = ap.parse_args(argv if where_argv is None else argv[:argv.index("--where") + 1])

    targets = {}
    if where_argv is None:
        log = [f"script: {args.script}"]
        commands = read_commands(args.script)
    else:
        dirs, wanted = where_targets(where_argv)
        log = [f"where: {wanted}"]
        targets = {d: own_outputs(d, BENCHMARK_OWN_DIRS) for d in dirs}
        commands = []
    for tokens in commands:
        for name, run_argv in split_command(tokens):
            if run_argv[0] not in TARGETS:
                log.append(f"[keep] {name}: delete only handles {'/'.join(TARGETS)} runs")
                continue
            dirs, own_dirs = TARGETS[run_argv[0]](run_argv[1:])
            for d in dirs:
                targets[d] = own_outputs(d, own_dirs)

    files = []
    for d, paths in targets.items():
        log.append(f"{d}: {len(paths)} item(s)" if paths else f"{d}: nothing to delete")
        for p in paths:
            found = [p] if not p.is_dir() or p.is_symlink() else sorted(f for f in p.rglob("*") if not f.is_dir())
            log += [f"  {f}" for f in found]
            files += found
    log_path = pathlib.Path("logs/delete") / f"{datetime.datetime.now():%Y%m%d-%H%M%S}{'-dry' if args.dry_run else ''}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log) + "\n")
    if args.dry_run or not files:
        print(f"{'Would delete' if files else 'Nothing to delete:'} {len(files)} file(s) ({log_path})")
        return
    jobs = running_jobs()
    if jobs:
        sys.exit("delete: refusing while jobs are running:\n  " + "\n  ".join(jobs))
    if not args.yes and input(f"Delete {len(files)} file(s)? List: {log_path.resolve()} [y/N] ").strip().lower() != "y":
        sys.exit(f"delete: aborted, nothing deleted (list kept at {log_path})")
    for paths in targets.values():
        for p in paths:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    print(f"Deleted {len(files)} file(s) ({log_path})")

if __name__ == "__main__":
    main(sys.argv[1:])
