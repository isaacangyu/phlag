import argparse
import itertools
import pathlib
import re
import shlex
import sys

BATCH_DESTS = {"window_size", "step_size", "rho", "beta", "beta_prime"}
PATH_DESTS = {"create", "output_file"}


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def _capture(fn):
    original = argparse.ArgumentParser.parse_args

    def raise_self(self, *a, **k):
        raise _Captured(self)

    argparse.ArgumentParser.parse_args = raise_self
    try:
        fn([])
    except _Captured as c:
        return c.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    raise RuntimeError(f"{fn} never called parse_args")


def get_parser(cmd):
    if cmd == "benchmark":
        from bench.cli.benchmark import _build_parser
        return _build_parser()
    if cmd == "phlag":
        from phlag.phlag import build_parser
        return build_parser()
    if cmd == "caster":
        from phlag.caster import build_parser
        return build_parser()
    if cmd == "phlagster":
        from bench.cli.phlagster import parse_arguments
        return _capture(parse_arguments)
    raise SystemExit(f"split_script: unsupported command {cmd!r} (benchmark/phlag/caster/phlagster)")


def read_commands(path):
    commands = []
    for line in re.sub(r"\\\n", " ", pathlib.Path(path).read_text()).splitlines():
        lex = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lex.whitespace_split = True
        current = []
        for tok in lex:
            if set(tok) <= set(";&|"):
                if current:
                    commands.append(current)
                current = []
            else:
                current.append(tok)
        if current:
            commands.append(current)
    return commands


def group_tokens(tokens, parser):
    options = parser._option_string_actions
    positionals, groups = [], []
    for tok in tokens:
        flag, eq, value = tok.partition("=")
        if tok.startswith("-") and (tok in options or (eq and flag in options)):
            name = flag if eq else tok
            groups.append([name, options[name], [value] if eq else []])
        elif groups:
            groups[-1][2].append(tok)
        else:
            positionals.append(tok)
    return positionals, groups


def is_axis(action, values):
    if len(values) < 2:
        return False
    return action.nargs is None or action.dest in BATCH_DESTS


def _abbrev_int(raw):
    raw = raw.strip().lower()
    mult = {"k": 1000, "m": 1000000}.get(raw[-1:], 1)
    return int(float(raw[:-1] if mult > 1 else raw) * mult)


def resolve_step(combo):
    w, s = combo.get("window_size"), combo.get("step_size")
    if w is None or s is None or "." not in s:
        return combo
    return {**combo, "step_size": str(max(1, round(float(s) * _abbrev_int(w))))}


def path_segments(combo, other):
    from phlag.caster import format_val
    from phlag.utils import PHLAG_SEGMENT_DEFAULTS, phlag_param_segment
    segs = []
    vals = {**other, **combo}
    if "rho" in combo or "beta" in combo or "beta_prime" in combo:
        rho = float(vals["rho"]) if "rho" in vals else None
        if "beta" in vals:
            segs.append(f"rho{rho}_beta{float(vals['beta'])}")
        elif "beta_prime" in vals:
            segs.append(f"rho{rho}_betaprime{float(vals['beta_prime'])}")
        else:
            segs.append(f"rho{rho}")
    for dest, raw in combo.items():
        if dest in BATCH_DESTS:
            continue
        if dest in PHLAG_SEGMENT_DEFAULTS:
            default = PHLAG_SEGMENT_DEFAULTS[dest]
            value = raw if default is None or isinstance(default, bool) else type(default)(raw)
            segs.append(phlag_param_segment(dest, value))
        else:
            segs.append(f"{dest.replace('_', '-')}={raw}")
    ws = None
    if "window_size" in combo or "step_size" in combo:
        ws = (format_val(_abbrev_int(vals["window_size"])) if "window_size" in vals else None,
              format_val(_abbrev_int(vals.get("step_size", vals.get("window_size")))) if "step_size" in vals or "window_size" in vals else None)
    return ws, segs


def rewrite_path(path, ws, segs):
    p = pathlib.Path(path)
    if ws is not None:
        parts = list(p.parts)
        idx = next((i for i, part in enumerate(parts) if re.match(r"^[wc]\d+[km]?_s\d+(\.\d+)?[km]?", part)), None)
        if idx is not None:
            m = re.match(r"^([wc])(\d+[km]?)_s(\d+(?:\.\d+)?[km]?)", parts[idx])
            w = ws[0] or m.group(2)
            s = ws[1] or m.group(3)
            parts[idx] = f"{m.group(1)}{w}_s{s}{parts[idx][m.end():]}"
            p = pathlib.Path(*parts)
        elif ws[0] and ws[1]:
            p = p / f"w{ws[0]}_s{ws[1]}"
        else:
            p = p / "_".join(x for x in ws if x)
    for seg in segs:
        p = p / seg
    return str(p)


def split_command(tokens):
    cmd, rest = tokens[0], tokens[1:]
    parser = get_parser(cmd)
    positionals, groups = group_tokens(rest, parser)
    split_pos = len(positionals) > 1
    axes = [(g[1].dest, g[2]) for g in groups if is_axis(g[1], g[2])]
    fixed = {g[1].dest: g[2][0] for g in groups if len(g[2]) == 1}
    pos_choices = [[p] for p in positionals] if split_pos else [positionals]
    runs = []
    for pos in pos_choices:
        for values in itertools.product(*(vals for _, vals in axes)):
            combo = resolve_step({**{d: fixed[d] for d in ("window_size", "step_size") if d in fixed},
                                  **dict(zip((d for d, _ in axes), values))})
            axis_combo = {d: combo[d] for d, _ in axes}
            if "step_size" in combo and "step_size" not in axis_combo and "window_size" in axis_combo:
                axis_combo["step_size"] = combo["step_size"]
            ws, segs = path_segments(axis_combo, fixed)
            argv = [cmd, *pos]
            for flag, action, vals in groups:
                if action.dest in axis_combo:
                    argv += [flag, axis_combo[action.dest]]
                elif action.dest in PATH_DESTS and vals and (ws or segs):
                    argv += [flag, rewrite_path(vals[0], ws, segs)]
                else:
                    argv += [flag, *vals]
            label = [cmd]
            if split_pos:
                label.append(pos[0].removesuffix("/scores.tsv"))
            if "window_size" in axis_combo or "step_size" in axis_combo:
                w, s = ws
                label.append(f"w{w}" if s in (None, w) else f"s{s}" if w is None else f"w{w}_s{s}")
            label += segs
            runs.append(("-".join(label), argv))
    return runs


def plan(script, budget=None):
    if budget is None:
        from bench.cli.benchmark import TOTAL_CORE_BUDGET
        budget = TOTAL_CORE_BUDGET
    lines, runs = [], []
    for i, tokens in enumerate(read_commands(script), 1):
        lines.append(shlex.join(tokens))
        runs += [(i, f"{i:02d}-{name}", argv) for name, argv in split_command(tokens)]
    if not runs:
        raise SystemExit(f"split_script: no commands in {script}")
    parallel = min(len(runs), budget)
    cores = max(1, budget // parallel)
    seen, planned = {}, []
    for line, name, argv in runs:
        name = re.sub(r"[^\w.=,-]+", "_", name)
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}-{seen[name]}"
        threads = 1 if argv[0] == "benchmark" else cores
        env = f"PHLAG_BENCH_JOBS={parallel} OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads}"
        planned.append((line, name, f"{env} {shlex.join(argv)}"))
    return parallel, lines, planned


def main(argv=None):
    ap = argparse.ArgumentParser(description="Expand bench/script.sh's multi-valued flags into one run per combination.")
    ap.add_argument("script", nargs="?", default="bench/script.sh")
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args(argv)
    parallel, _, planned = plan(args.script, args.budget)
    print(parallel)
    for _, name, cmd in planned:
        print(f"{name}\t{cmd}")

if __name__ == "__main__":
    main(sys.argv[1:])
