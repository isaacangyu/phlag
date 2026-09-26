import sys
import pathlib
import os
import re
import argparse
import itertools
import subprocess
import shutil
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, expon, laplace, skew

def format_val(val):
    """
    Abbreviate numbers using 'k' for thousands and 'm' for millions.
    0 remains '0'.
    """
    if val == 0:
        return "0"
    if val % 1000000 == 0:
        return f"{val // 1000000}m"
    elif val % 1000 == 0:
        return f"{val // 1000}k"
    return str(val)

def int_or_abbrev(val_str):
    val_str = str(val_str).strip().lower()
    if val_str.endswith('k'):
        return int(float(val_str[:-1]) * 1000)
    elif val_str.endswith('m'):
        return int(float(val_str[:-1]) * 1000000)
    return int(val_str)

def step_size_or_fraction(val_str):
    """
    Ratio-vs-literal is decided lexically, not by numeric range: a value
    written with a decimal point (e.g. "0.5", "1.0", "2.0") is a ratio of
    -w's window size, multiplied out in main() (see the isinstance(..., float)
    branch there) -- "1.0" means non-overlapping (step==window), "2.0" means
    a window-sized gap between windows. A value with no decimal point (with
    or without a k/m suffix, e.g. "1000", "1k") is a literal/absolute step size.
    """
    val_str = str(val_str).strip().lower()
    if val_str.endswith('k') or val_str.endswith('m'):
        return int_or_abbrev(val_str)
    if '.' in val_str:
        return float(val_str)
    return int_or_abbrev(val_str)

def recover_source_fasta(scores_path):
    """
    Recovers the source FASTA path from a scores.tsv's 'file' column (its
    first data row) -- the same convention read_caster_scores relies on in
    phlag.py. Returns None if the file has no 'file'-led header or no data
    rows.
    """
    with open(scores_path, "r") as f:
        header = f.readline().strip().split("\t")
        if not header or header[0].lower() != "file":
            return None
        for line in f:
            if line.strip():
                return pathlib.Path(line.split("\t")[0])
    return None


def parse_ws_from_path(path):
    """
    Recovers (mode, window_or_chunk, step, is_site, is_zscale, is_ilr,
    is_normalize, norm_eps) from a 'w<...>_s<...>' (dstar), 'c<...>_s<...>'
    (--pair), or 'c<...>_s<...>[_site][_z][_i][_n][_norm-eps]' (--site/
    --zscale/--ilr/--normalize/--norm-eps) path segment, as written by
    caster.py's standalone out/ tree (flat suffixes) and older canonical
    store/caster/ runs (same flat suffixes). The current canonical
    store/caster/ tree instead nests 'site'/'ilr'/'normalize' as their own
    path components right after the size segment (zscale stays a flat '_z'
    suffix there), and 'normalize' itself may further nest a 'norm-eps'
    component when --norm-eps (a boolean flag) was set (see
    get_expected_caster_sim_dir/_derive_output_path) -- so after matching the
    size segment, also consume any immediately-following
    'site'/'ilr'/'normalize'/'norm-eps' components, OR'd into whatever the
    flat suffixes already captured. norm_eps is a plain bool (never None):
    True only when a 'norm-eps'/'_norm-eps' marker was actually found.
    Returns None if no 'w'/'c'-prefixed size segment is found anywhere in
    path's parts.
    """
    parts = path.parts
    for i, part in enumerate(parts):
        m = re.match(r'^([wc])(\d+[km]?)_s(\d+[km]?)(_site)?(_z)?(_i)?(_n)?(_norm-eps)?$', part, re.IGNORECASE)
        if m:
            is_site = bool(m.group(4))
            is_zscale = bool(m.group(5))
            is_ilr = bool(m.group(6))
            is_normalize = bool(m.group(7))
            norm_eps = bool(m.group(8))
            for nested in parts[i + 1:]:
                if nested == "site":
                    is_site = True
                elif nested == "ilr":
                    is_ilr = True
                elif nested == "normalize":
                    is_normalize = True
                elif nested == "norm-eps":
                    norm_eps = True
                else:
                    break
            return (
                m.group(1).lower(), int_or_abbrev(m.group(2)), int_or_abbrev(m.group(3)),
                is_site, is_zscale, is_ilr, is_normalize, norm_eps,
            )
    return None


def substitute_ws_in_path(path, new_val, new_step):
    """
    Inverse of parse_ws_from_path: returns a copy of `path` with its first
    'w<...>_s<...>'/'c<...>_s<...>' size segment's two numeric values swapped
    for new_val/new_step (formatted via format_val, so e.g. 1000 -> '1k',
    matching how caster.py itself names these directories), leaving the mode
    letter and any trailing suffix (_site/_z/_i/_n/_norm-eps, or a nested
    site/ilr/normalize/norm-eps path component) untouched. Used by phlag.py's
    -w/-s batch flags to locate a sibling scores.tsv for the same node/
    pattern at a different window/step. Returns None if no such segment is
    found anywhere in path's parts (mirrors parse_ws_from_path).
    """
    parts = path.parts
    for i, part in enumerate(parts):
        m = re.match(r'^([wc])(\d+[km]?)_s(\d+[km]?)', part, re.IGNORECASE)
        if m:
            new_part = f"{m.group(1)}{format_val(new_val)}_s{format_val(new_step)}{part[m.end():]}"
            return pathlib.Path(*parts[:i], new_part, *parts[i + 1:])
    return None


def adhoc_scores_path(repo_root, args, cats, node_rel, normalize_flag, ilr_flag):
    """
    Standalone scores.tsv path: out/[<category>/<subcategory>/]<node>/<pattern>/<size>/[variant/]scores.tsv,
    where <size> is w<W>_s<S>[_z] (or c<chunk>_s<step>[_z] for --pair/--site) and
    variant nests site/ilr/normalize[/norm-eps] exactly like store/caster/.
    """
    zscale_suffix = "_z" if args.zscale else ""
    step_str = format_val(args.step_size)
    if args.pair or args.site:
        chunk = args.chunk_size if args.chunk_size is not None else args.window_size
        size_dir = f"c{format_val(chunk)}_s{step_str}{zscale_suffix}"
    else:
        size_dir = f"w{format_val(args.window_size)}_s{step_str}{zscale_suffix}"
    from .utils import get_out_root
    path = get_out_root()
    if cats:
        path = path / cats[0] / cats[1]
    path = path / node_rel / size_dir
    if args.site:
        path = path / "site"
    if ilr_flag:
        path = path / "ilr"
    elif normalize_flag:
        path = path / "normalize"
        if args.norm_eps:
            path = path / "norm-eps"
    if args.exp_minus:
        path = path / "exp-minus"
    return path / "scores.tsv"


def apply_zscale(rows, keys):
    """
    Rescales each of `keys` (dict keys into `rows`, a list of dicts) to mean
    0.5, std 0.5 across the whole file: z-score to mean 0/std 1, then
    0.5 + 0.5*z. Computed once over all rows in float64, in place. Not
    clipped, so outlier windows can still land outside [0,1]. A zero-std
    column is left at a constant 0.5 rather than dividing by zero.
    """
    if not rows:
        return rows
    arr = np.array([[r[k] for k in keys] for r in rows], dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    scaled = 0.5 + 0.5 * (arr - mean) / std
    for row, vals in zip(rows, scaled):
        for k, v in zip(keys, vals):
            row[k] = float(v)
    return rows


def apply_zscale_to_scores_file(path, has_q123):
    """
    Rescales c*ABBA/c*BABA/c*AABB in an already-written scores TSV (used by
    run_caster_pair/run_caster_site, after their own K-rollup) to mean
    0.5/std 0.5 via apply_zscale, rewriting the file in place. If q1/q2/q3
    are also present (--pair), recomputes them as proportions of the
    rescaled sums to keep the file internally consistent -- their values are
    no longer literal proportions once the base sums are recentered, but
    CasterPlotter/read_caster_scores still just treat them as a derived
    per-topology column.
    """
    df = pd.read_csv(path, sep="\t")
    rows = df.to_dict("records")
    apply_zscale(rows, ["c*ABBA", "c*BABA", "c*AABB"])
    if has_q123:
        for row in rows:
            s0, s1, s2 = row["c*ABBA"], row["c*BABA"], row["c*AABB"]
            tot = s0 + s1 + s2
            if tot != 0:
                row["q1"], row["q2"], row["q3"] = s0 / tot, s1 / tot, s2 / tot
            else:
                row["q1"] = row["q2"] = row["q3"] = 1.0 / 3
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


DEFAULT_NORM_EPS = 1e-6


def apply_normalize(rows, keys, eps=None):
    """
    Normalizes each row's `keys` (dict keys into `rows`, a list of dicts) to
    proportions of that row's own sum (1/3 each if the sum is 0), in place.
    Mirrors caster-pair.cpp's q1/q2/q3 -- NOT the same as dividing by
    QuartetCnt (dstar.cpp's quartetCnt(), a per-site sequence-depth product
    unrelated to the sum of the three D* topology scores). D* itself is
    invariant to this rescaling (same denominator cancels), so it is left
    untouched by callers.

    `eps=None` (default) is the original, unguarded behavior: exact
    `denom == 0` check, 1/3 fallback only then, plain division otherwise --
    every existing cached normalize run used this, so it stays bit-for-bit
    reproducible by default. Bug: a window with no informative sites should
    have all three raw values at exactly 0, but float accumulation leaves
    residual noise around 1e-14 instead, so the exact-zero check misses it
    and dividing by that noise-level denom blows a noise-level numerator up
    to spurious values in the thousands to billions.

    `--norm-eps` (boolean CLI flag; passes `eps=DEFAULT_NORM_EPS` here when
    set) opts into the guarded behavior instead: 1/3-fallback the whole row
    whenever every value is below NOISE_FLOOR (catches the all-near-zero
    case above), and otherwise clamp `denom`'s magnitude (sign preserved) to
    at least `eps` before dividing -- also catches a row whose 3 values are
    each individually real but nearly cancel (c*ABBA/c*BABA/c*AABB are
    CASTER's signed scoreCnt() evidence, not a plain count, so this is a
    real, not-rare case). Boolean rather than a tunable float so the two
    behaviors stay exactly two cache entries -- 'normalize' (old data) and
    'normalize/norm-eps' (fixed data) -- not an open-ended eps<value> family.
    """
    NOISE_FLOOR = 1e-9
    for row in rows:
        vals = [row[k] for k in keys]
        denom = sum(vals)
        if eps is not None and max(abs(v) for v in vals) < NOISE_FLOOR:
            for k in keys:
                row[k] = 1.0 / len(keys)
        elif eps is None and denom == 0:
            for k in keys:
                row[k] = 1.0 / len(keys)
        else:
            denom_safe = (max(denom, eps) if denom >= 0 else min(denom, -eps)) if eps is not None else denom
            for k, v in zip(keys, vals):
                row[k] = v / denom_safe
    return rows


def apply_normalize_to_scores_file(src_path, dst_path, has_q123, eps=None):
    """
    Reads an already-written scores TSV at `src_path` (un-normalized), applies
    apply_normalize to c*ABBA/c*BABA/c*AABB, and writes the result to
    `dst_path` -- used by the --normalize short-circuit (see main()) to avoid
    recomputing dstar/caster-pair/caster-site when an un-normalized scores.tsv
    for the same window/step (or chunk/step) already exists. If q1/q2/q3 are
    also present (--pair), they collapse to exactly the normalized
    c*ABBA/c*BABA/c*AABB values (a row's proportions summed to 1). `eps`:
    see apply_normalize.
    """
    df = pd.read_csv(src_path, sep="\t")
    rows = df.to_dict("records")
    apply_normalize(rows, ["c*ABBA", "c*BABA", "c*AABB"], eps=eps)
    if has_q123:
        for row in rows:
            row["q1"], row["q2"], row["q3"] = row["c*ABBA"], row["c*BABA"], row["c*AABB"]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dst_path.parent), prefix=".scores_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            pd.DataFrame(rows).to_csv(f, sep="\t", index=False)
        os.replace(tmp_path, dst_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class _SkbioDeviceArray(np.ndarray):
    """
    Zero-copy ndarray view exposing a `.device` property. skbio 0.7.0's
    `ilr()` unconditionally reads `mat.device` (part of the Python array API
    standard) -- a real ndarray only gained that attribute in numpy 2.0, but
    this project is pinned to numpy<2.0 (via jax==0.4.30), so calling
    `ilr()` on a plain ndarray raises AttributeError. This shim (verified
    numerically identical to a working `ilr()` call) works around that
    version mismatch without touching the pinned numpy/skbio versions.
    """
    @property
    def device(self):
        return "cpu"


def apply_ilr(rows, keys, out_keys):
    """
    Replaces rows' `keys` (dict keys into `rows`, a list of dicts) with
    their len(keys)-1 isometric-log-ratio (ILR) coordinates under `out_keys`
    (must have length len(keys) - 1), computed via skbio.stats.composition:
      0. per-row shift -- dstar's raw counts are genuine non-negative
         site-pattern counts, but caster-pair's/caster-site's raw c*ABBA/
         c*BABA/c*AABB are CASTER's internal scoreCnt() quartet-support
         statistic instead (a signed evidence score, not a count) and can
         be negative -- verified directly against a real --site run.
         closure()/ilr() require strictly positive parts, so any row with a
         non-positive component is shifted up by that row's own
         |min| (relative-scaled, plus a tiny absolute floor) before closure,
         preserving the relative differences between the 3 parts within
         that row (an additive shift, not a rescale). Rows already all
         positive are left untouched.
      1. closure -- close each row to proportions summing to 1.
      2. multi_replace -- replace exact zeros with a small value,
         proportionally shrinking the other parts so the row still sums to
         1 (standard compositional-data-analysis zero handling; ilr's log
         step is undefined at 0).
      3. ilr -- map to len(keys)-1 real-valued coordinates (default
         Egozcue/Gram-Schmidt orthonormal basis).
    A row whose `keys` sum to exactly 0 (e.g. a window with no informative
    sites for any of the 3 topologies) is remapped to a uniform composition
    *before* closure -- closure's own 0/0 division would otherwise produce
    NaN, which multi_replace/ilr would silently propagate. This mirrors
    apply_normalize's identical "1/N each if the sum is 0" fallback for the
    same edge case, just applied one step earlier (as input to closure
    rather than as the final value).
    Mutates `rows` in place: pops `keys` and inserts `out_keys` (in row-dict
    insertion order, i.e. at the end) in their place, preserving every
    other key. Do not call this after apply_zscale, whose rescaled values
    can be negative in a way that no longer reflects the underlying
    scoreCnt() statistic (this is exactly why -i/--ilr and -z/--zscale are
    mutually exclusive at the CLI level).
    """
    if not rows:
        return rows
    from skbio.stats.composition import closure, multi_replace, ilr as skbio_ilr

    arr = np.array([[row[k] for k in keys] for row in rows], dtype=np.float64)

    # Only genuinely negative rows are shifted -- exact zeros are left for
    # multi_replace below (its proportional zero-replacement, not an
    # arbitrary additive shift, is the more standard CoDA treatment for a
    # part that's legitimately absent rather than negative-valued).
    row_min = arr.min(axis=1, keepdims=True)
    needs_shift = row_min[:, 0] < 0
    if needs_shift.any():
        arr = arr.copy()
        shift_amount = np.abs(row_min) * 1e-6 + 1e-9
        shift = np.where(row_min < 0, -row_min + shift_amount, 0.0)
        arr += shift

    zero_rows = arr.sum(axis=1) == 0
    if zero_rows.any():
        arr = arr.copy()
        arr[zero_rows] = 1.0  # closes to a uniform composition below

    # multi_replace squeezes a single-row (1, D) input down to (D,) (an
    # skbio 0.7.0 quirk) -- atleast_2d restores the batch dimension so a
    # file with exactly one window doesn't break the ilr() call below.
    closed = np.atleast_2d(np.asarray(multi_replace(closure(arr))))
    coords = np.asarray(skbio_ilr(closed.view(_SkbioDeviceArray)))
    assert coords.shape[1] == len(out_keys)

    for row, vals in zip(rows, coords):
        for k in keys:
            del row[k]
        for k, v in zip(out_keys, vals):
            row[k] = float(v)
    return rows


def apply_ilr_to_scores_file(src_path, dst_path, has_q123):
    """
    Reads an already-written scores TSV at `src_path` (raw, un-transformed
    c*ABBA/c*BABA/c*AABB counts), replaces those 3 columns with their 2 ILR
    coordinates (c*ILR1/c*ILR2) via apply_ilr, and writes the result to
    `dst_path` atomically -- mirrors apply_normalize_to_scores_file's
    split-path/atomic-write shape exactly (not apply_zscale_to_scores_file's
    simpler in-place-only shape), since this is used both in-place
    (run_caster_pair/run_caster_site, src_path == dst_path == chunk_scores_path,
    right after their own K-rollup) and out-of-place (the --bench --ilr
    short-circuit in main(), src_path = a cached raw sibling, dst_path =
    final_output_path, to avoid re-running dstar/caster-pair/caster-site
    from scratch).
    If q1/q2/q3 are also present (--pair), they are dropped entirely --
    they were proportions of the original 3-part composition and have no
    meaningful equivalent once it's replaced by 2 ILR coordinates.
    """
    df = pd.read_csv(src_path, sep="\t")
    rows = df.to_dict("records")
    apply_ilr(rows, ["c*ABBA", "c*BABA", "c*AABB"], ["c*ILR1", "c*ILR2"])
    if has_q123:
        for row in rows:
            del row["q1"], row["q2"], row["q3"]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dst_path.parent), prefix=".scores_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            pd.DataFrame(rows).to_csv(f, sep="\t", index=False)
        os.replace(tmp_path, dst_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def strip_exp_minus(path):
    return pathlib.Path(*[p for p in pathlib.Path(path).parts if p != "exp-minus"])


def apply_exp_minus_to_scores_file(src_path, dst_path):
    """
    Replaces each topology column (c*ABBA/c*BABA/c*AABB, or c*ILR1/c*ILR2
    under --ilr) of an already-written scores TSV with exp(-score), writing
    the result to `dst_path` atomically. Applied last, after any
    -n/-i/-z transform already baked into `src_path`. q1/q2/q3 (--pair) are
    left untouched since phlag never reads them when the c* columns exist.
    """
    df = pd.read_csv(src_path, sep="\t")
    cols = [c for c in df.columns if re.fullmatch(r"c\*(ABBA|BABA|AABB|ILR\d+)", c)]
    df[cols] = np.exp(-df[cols].astype(np.float64))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dst_path.parent), prefix=".scores_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            df.to_csv(f, sep="\t", index=False)
        os.replace(tmp_path, dst_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def is_plot_only_argv(raw_argv):
    """
    True if raw_argv's only flag is --plot (plus its own choice values) --
    used by regen mode (a scores.tsv/chunk_scores.tsv path passed as the
    positional arg) to tell "just redraw the plots from what's already
    there" apart from "recompute and overwrite", without requiring every
    other flag to be re-specified.
    """
    plot_choices = {"scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums"}
    i = 0
    saw_plot_flag = False
    while i < len(raw_argv):
        tok = raw_argv[i]
        if tok.startswith("-"):
            if tok != "--plot":
                return False
            saw_plot_flag = True
            i += 1
            while i < len(raw_argv) and raw_argv[i] in plot_choices:
                i += 1
        else:
            i += 1
    return saw_plot_flag


def copy_quartet_counts_if_missing(src_dir, dst_dir):
    """
    Propagates a sibling quartet_counts.tsv (dstar.cpp/caster-site.cpp's
    diagnostic per-window quartet-count companion, see plot_quartet_counts) from
    src_dir to dst_dir when dst_dir doesn't already have one -- used by the
    --ilr/--normalize/exact-cache short-circuits below, which reuse an
    already-computed scores.tsv instead of re-running the binary, so
    quartet_counts.tsv (unaffected by ILR/normalize -- it's about raw per-site
    score sign, not the scaled score columns) would otherwise never appear
    next to the transformed/cached output. No-op if src has none, or dst
    already has one (never overwrites).
    """
    src_path = src_dir / "quartet_counts.tsv"
    dst_path = dst_dir / "quartet_counts.tsv"
    if dst_path.exists() or not src_path.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def get_fasta_length(fasta_path):
    length = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                break
        for line in f:
            if line.startswith(">"):
                break
            length += len(line.strip())
    return length


class CasterPlotter:
    def __init__(self, scores_file, distribution='gaussian', data_dir=None, topologies=None, plot_scores=True, plot_dist=False, plot_correlation=False, plot_topology_pairs=False, plot_quartet_counts=False, plot_sums=False, locus_pattern=None, predicted_intervals=None):
        self.scores_file = scores_file
        self.distribution = distribution
        self.data_dir = data_dir if data_dir is not None else str(pathlib.Path(scores_file).parent)
        self.topologies = topologies
        self.locus_pattern = locus_pattern
        self.predicted_intervals = predicted_intervals

        os.makedirs(self.data_dir, exist_ok=True)

        from .utils import get_locus_description
        self.gene_name = get_locus_description(scores_file)

        self.load_data()

        # Transform/source tag for the plot title -- recovered from the
        # w<W>_s<S>[_z][_i][_n]/c<chunk>_s<step>[_site][_z][_i][_n] path
        # segment (see parse_ws_from_path) so the plot always states which
        # columns and which rescaling it's actually showing, matching what
        # phlag.py's read_caster_scores feeds the HMM. Falls back to
        # detecting q1/q2/q3 presence (--pair's own tell, since --site never
        # writes those columns) when the path doesn't encode it (e.g. a
        # custom -o path), and says nothing rather than guessing wrong.
        ws = parse_ws_from_path(pathlib.Path(scores_file))
        has_q123 = self.df is not None and all(c in self.df.columns for c in ('q1', 'q2', 'q3'))
        if ws:
            mode, _, _, is_site, is_zscale, is_ilr, is_normalize, _ = ws
            source = "site" if (mode == "c" and is_site) else ("pair" if mode == "c" else "dstar")
            transform_bits = [b for b, on in (("zscaled", is_zscale), ("ilr", is_ilr), ("normalized", is_normalize), ("exp-minus", "exp-minus" in pathlib.Path(scores_file).parts)) if on]
            tag_parts = [source] + (transform_bits or ["raw"])
        else:
            # Path doesn't encode a recognizable size-dir segment (e.g. a
            # custom -o destination) -- q1/q2/q3 presence still identifies
            # --pair, but whether -z/-n were applied isn't recoverable from
            # the file alone, so leave the transform unstated rather than
            # guessing "raw" and risking a mislabeled plot.
            source = "pair" if has_q123 else None
            tag_parts = [source] if source else []
        self.data_tag = ", ".join(tag_parts) if tag_parts else None

        if self.df is not None:
            # High-contrast, vibrant, and highly distinguishable color palette for the 3 topologies
            self.topo_colors = {
                'ABBA': '#1F77B4',    # Bold Royal Blue
                'BABA': '#D62728',    # Vivid Crimson Red
                'AABB': '#2CA02C',    # Vibrant Forest Green
            }

            if plot_scores:
                self.plot_topology_scatter()
            if plot_dist:
                self.plot_distribution()
            if plot_topology_pairs:
                self.plot_topology_pairs()
            if plot_correlation:
                self.plot_correlation()
            if plot_quartet_counts:
                self.plot_quartet_counts()
                self.plot_quartet_dists()
            if plot_sums:
                self.plot_sums()

    def load_data(self):
        """Parses the tab-separated value file into a Pandas DataFrame."""
        target_path = self.scores_file
        if not os.path.exists(target_path) and not os.path.isabs(target_path):
            target_path = os.path.join(self.data_dir, self.scores_file)

        try:
            self.df = pd.read_csv(target_path, sep='\t')
            print(f"Loaded {len(self.df)} windows for locus '{self.gene_name}' from: {target_path}")
            print("Detected columns:", self.df.columns.tolist())
            sns.set_theme(style="whitegrid")
        except Exception as e:
            print(f"Error reading dataset file {target_path}: {e}")
            self.df = None

    def _ground_truth_pattern(self):
        """
        The ground-truth locus pattern (e.g. '37-62' or 'n1a1n5...'), preferring
        the explicit locus_pattern the caller already knows (caster.py's own
        run passes it, since flat standalone output no longer encodes it in
        scores_file's path) and falling back to regex-parsing scores_file's
        path/filename otherwise (a bare scores.tsv invocation, or the
        canonical --bench tree, which still encodes it there).
        """
        if self.locus_pattern:
            return self.locus_pattern
        full_path_str = str(self.scores_file)
        m = re.search(r'((?:[an]\d+(?:-[an]?\d+)?(?:_)?)+|\d+-\d+(?:[;_,]\d+-\d+)*)', full_path_str)
        return m.group(1) if m else None

    def _shade_locus_pattern(self, ax):
        """
        Draws ground-truth Alt-region shading, null/alt divider lines, and
        block labels on ax -- factored out of plot_topology_scatter so every
        window-position plot in this file (including plot_quartet_counts)
        marks the same regions the same way. No-op if no ground-truth
        pattern is resolvable.
        """
        pattern_str = self._ground_truth_pattern()
        if not pattern_str:
            return
        from .utils import parse_pattern_string
        total_span = self.df['pos'].max() if ('pos' in self.df.columns and len(self.df) > 0) else None
        blocks, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)

        if anomaly_intervals and not any(b[0] == 'n' for b in blocks):
            # Pure interval format (e.g. 45-55)
            alt_shaded = False
            for start_pos, end_pos in anomaly_intervals:
                lbl = 'Alt' if not alt_shaded else None
                ax.axvspan(start_pos, end_pos, color='#E05638', alpha=0.12, label=lbl)
                alt_shaded = True
                ax.axvline(x=start_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)
                ax.axvline(x=end_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)
        elif blocks:
            curr_pos_bp = 0
            alt_shaded = False
            for idx, (b_type, b_id, length_bp) in enumerate(blocks):
                start_pos = curr_pos_bp
                end_pos = curr_pos_bp + length_bp
                mid_pos = (start_pos + end_pos) / 2.0
                label_text = "null" if b_type == 'n' else "alt"

                if b_type == 'a':
                    lbl = 'Alt' if not alt_shaded else None
                    ax.axvspan(start_pos, end_pos, color='#E05638', alpha=0.12, label=lbl)
                    alt_shaded = True

                if start_pos > 0:
                    ax.axvline(x=start_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)

                ax.text(mid_pos, 0.02, label_text, transform=ax.get_xaxis_transform(),
                         ha='center', va='bottom', fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

                curr_pos_bp = end_pos
            ax.axvline(x=curr_pos_bp, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)

    def _shade_predicted_intervals(self, ax):
        """
        Overlays phlag's own Viterbi-predicted Alt regions in yellow, same
        axvspan style as _shade_locus_pattern's ground-truth shading -- set
        by Phlag re-opening an already-plotted scatter.png (see
        Phlag.compute_output) so agreement with the ground truth reads as
        blended red+yellow, disagreement as pure red or pure yellow.
        """
        if not self.predicted_intervals:
            return
        shaded = False
        for start_pos, end_pos in self.predicted_intervals:
            lbl = 'Predicted Alt' if not shaded else None
            ax.axvspan(start_pos, end_pos, color='#F4D03F', alpha=0.18, label=lbl)
            shaded = True

    def calculate_summary_statistics(self, series):
        """Calculates summary statistics, returns and sets self.params dict for scipy.stats."""
        if self.distribution:
            dist_name = self.distribution.lower()
            if dist_name in ['gaussian', 'normal']:
                dist_name = 'norm'

            import scipy.stats as stats_module
            dist_class = getattr(stats_module, dist_name)
            fit_vals = dist_class.fit(series)

            param_names = []
            if dist_class.shapes:
                param_names.extend([s.strip() for s in dist_class.shapes.split(',')])
            param_names.extend(['loc', 'scale'])
            self.params = dict(zip(param_names, fit_vals))
        else:
            self.params = {
                'loc': series.mean(),
                'scale': series.std()
            }
        return self.params

    @staticmethod
    def _needs_log_scale(values, ratio_threshold=100.0):
        """
        Auto-detects whether an axis (scatter/line values, or histogram bar
        heights -- whatever's actually plotted against it) needs a log
        scale, replacing the old manual --plot log modifier. True when the
        data's 5th-to-95th-percentile spread, positive values only (log is
        undefined at/below zero, and matplotlib silently drops non-positive
        points on a log axis anyway), covers at least `ratio_threshold`x.

        Percentile- rather than true min/max-based so one stray outlier
        can't flip it on its own. The default threshold, 100x (2 orders of
        magnitude), is the point past which a linear axis genuinely can't
        resolve both ends at once -- the small end rounds to a sub-pixel
        sliver next to the large one -- while staying loose enough that
        ordinary, moderately-skewed data (under ~100x top-to-bottom) still
        reads fine linear and isn't switched over unnecessarily.
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) < 5:
            return False
        lo, hi = np.percentile(vals, [5, 95])
        return lo > 0 and (hi / lo) >= ratio_threshold

    @staticmethod
    def _safe_bin_edges(values, lo, hi, min_bins=10, max_bins=100):
        """
        bin_edges for a bins='auto' request, computed and clamped ourselves
        instead of trusting numpy's (or seaborn histplot's, which delegates
        to it) own 'auto' estimator directly: its Freedman-Diaconis bin
        width is derived from `values`' full-array IQR, not the (lo, hi)
        range callers already clip the visible axis to -- so a near-zero
        IQR (the data's bulk sitting almost on top of itself, common at
        small window sizes -- see "bins='auto' small-window OOM") still
        requests an astronomical bin count over that (lo, hi) span and OOMs
        np.linspace inside histogram_bin_edges, even with the range clipped.
        Clamping the bin count ourselves to [min_bins, max_bins] instead of
        letting numpy's estimate through unchecked closes that hole for good.
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if hi <= lo or len(vals) < 2:
            return np.linspace(lo, hi, min_bins + 1)
        q25, q75 = np.percentile(vals, [25, 75])
        fd_width = 2 * (q75 - q25) * (len(vals) ** (-1.0 / 3.0))
        bin_width = fd_width if fd_width > 0 else (hi - lo) / max_bins
        n_bins = int(np.clip(np.ceil((hi - lo) / bin_width), min_bins, max_bins))
        return np.linspace(lo, hi, n_bins + 1)

    @staticmethod
    def resolve_topology_columns(df, topologies=None):
        """
        Shared topology-column resolution for both the scatter plot and
        write_ground_truth_stats: caster-pair's chunk_scores.tsv carries both
        raw per-window quartet-support sums (c*ABBA/c*BABA/c*AABB, unbounded,
        scale with window size, possibly rescaled at write time by -z/-n) and
        their normalized proportions (q1/q2/q3, in [0,1], positionally
        ABBA/BABA/AABB -- see caster-pair.cpp's scoreChunksForBranch). Prefer
        the raw c*/avg* columns whenever present, matching phlag.py's
        read_caster_scores exactly (it always reads those over q1/q2/q3 when
        both exist), so the plot shows the same values the HMM actually
        fits on -- q1/q2/q3 divide out nearly all real signal (the three
        sums move almost in lockstep) and are only a fallback for files that
        never had c*/avg* columns to begin with. Returns (avg_cols, rename_map).
        """
        avg_cols = [c for c in df.columns if 'avg' in c or 'c*' in c]
        rename_map = {}
        for col in avg_cols:
            match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
            rename_map[col] = match.group(1).upper() if match else col

        if not avg_cols:
            pair_cols = ['q1', 'q2', 'q3']
            if all(c in df.columns for c in pair_cols):
                avg_cols = pair_cols
                rename_map = {'q1': 'ABBA', 'q2': 'BABA', 'q3': 'AABB'}

        if topologies is not None:
            filtered_cols = []
            for col in avg_cols:
                mapped_name = rename_map.get(col, col)
                for t in topologies:
                    if t.lower() in mapped_name.lower():
                        filtered_cols.append(col)
                        break
            avg_cols = filtered_cols

        return avg_cols, rename_map

    def plot_topology_scatter(self):
        avg_cols, rename_map = self.resolve_topology_columns(self.df, self.topologies)

        if not avg_cols:
            print("No matching topology columns found to scatter plot.")
            return

        plt.figure(figsize=(12, 6))

        renamed_df = self.df.rename(columns=rename_map)
        clean_cols = [rename_map.get(c, c) for c in avg_cols]

        melted_df = renamed_df.melt(id_vars=['pos'], value_vars=clean_cols,
                                 var_name='Topology', value_name='Score')

        # self.topo_colors is keyed by ABBA/BABA/AABB only -- an ILR-transformed
        # file's clean_cols are 'c*ILR1'/'c*ILR2' (resolve_topology_columns
        # doesn't rename those, no ABBA/BABA/AABB substring to match), so
        # seaborn's palette= would KeyError on an unrecognized hue level.
        palette = self.topo_colors if set(clean_cols) <= set(self.topo_colors) else None
        sns.scatterplot(data=melted_df, x='pos', y='Score', hue='Topology', palette=palette, alpha=0.6, s=12)

        scores = melted_df['Score'].to_numpy(dtype=float)
        finite_scores = scores[np.isfinite(scores)]
        abs_scores = np.abs(finite_scores)
        if self._needs_log_scale(abs_scores):
            # Auto-detected off magnitude (see _needs_log_scale), same
            # treatment as plot_dist/plot_sums, so a scatter of raw c*/avg*
            # sums (which can span orders of magnitude across window sizes)
            # doesn't squash the low end flat. c*ABBA/c*BABA are CASTER's
            # signed scoreCnt() evidence (not a plain count) and are
            # genuinely negative roughly half the time -- a plain log axis
            # silently drops those points, so switch to symlog instead
            # whenever negatives are present, with the same NOISE_FLOOR
            # (1e-9, see apply_normalize) as linthresh: below that, values
            # are float cancellation residue around a "should be exactly 0"
            # result, not real signal, so they belong in the linear zone
            # around zero rather than stretched out over more log decades.
            if (finite_scores < 0).any():
                plt.gca().set_yscale('symlog', linthresh=1e-9)
            else:
                plt.gca().set_yscale('log')

        # Draw vertical split lines and shade alt regions if a ground truth pattern is known
        self._shade_locus_pattern(plt.gca())
        self._shade_predicted_intervals(plt.gca())

        # Format names for cleaner legend and title
        title = f'Genomic Topology Profile: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
        plt.title(title, fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('Genomic Position (pos)', fontsize=11, labelpad=8)
        plt.ylabel('Topology Score Value', fontsize=11, labelpad=8)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_scatter = os.path.join(output_dir, 'scatter.png')
        plt.savefig(save_path_scatter, dpi=300)
        print(f"Saved empirical topology scatter plot to: {save_path_scatter}")
        plt.close()

    def plot_distribution(self):
        """
        Pre-refactor 'dist' plot, restored: one subplot per resolved topology
        column, each an empirical Null/Alt histogram (the same ground-truth
        split _compute_null_alt_labels resolves for the 3D/pairs/correlation
        plots) with Gaussian fit overlays and annotated mean/std. Unlike
        plot_topology_scatter, which just skips the shading when no
        ground-truth pattern is resolvable, there's no meaningful Null/Alt
        histogram without one, so this skips entirely (CLAUDE.md's "needs a
        locus pattern... else eval skips, not errors") rather than the old
        code's sys.exit.
        """
        avg_cols, rename_map = self.resolve_topology_columns(self.df, self.topologies)
        if not avg_cols:
            print("No matching topology columns found to plot distributions for.")
            return

        labels = self._compute_null_alt_labels()
        if labels is None:
            print("No resolvable ground-truth locus pattern; skipping distribution plot.")
            return

        norm_label = 'Normalized (Min-Max)' if pathlib.Path(self.scores_file).stem.endswith('_n') else 'Raw'

        import matplotlib.transforms as transforms
        from .utils import format_adaptive, gaussian_hellinger2_nd, exponential_hellinger2_nd, laplace_hellinger2_nd

        is_exponential = self.distribution == "exp"
        is_double_exponential = self.distribution == "dexp"

        # Joint (all-topology) Hellinger^2, via the Gaussian Bhattacharyya
        # coefficient over the full covariance matrix -- same computation
        # write_ground_truth_stats uses for gt_stats.txt's own Hellinger2, so
        # every subplot's title reports the real joint separability (capturing
        # cross-topology covariance) instead of a per-topology univariate
        # value that ignores it. Computed once, shared across all subplots.
        null_mask, alt_mask = labels == 'Null', labels == 'Alt'
        h2_joint = None
        if null_mask.sum() > 1 and alt_mask.sum() > 1:
            Y_all = self.df[avg_cols].to_numpy(dtype=float)
            null_Y, alt_Y = Y_all[null_mask], Y_all[alt_mask]
            if is_exponential:
                # A per-column true min() is fragile to a single extreme
                # outlier -- it can sit far below the bulk of the data,
                # which then inflates the fitted scale (deflates the rate)
                # until the exponential curve is indistinguishable from
                # flat over any reasonably-zoomed view (see the identical,
                # per-topology version of this fix below). 1st-percentile
                # floor instead, clipping any point still under it to
                # exactly the floor so the shifted values stay >= 0.
                shift_all = np.percentile(Y_all, 1, axis=0)
                null_rates_joint = 1.0 / np.clip(null_Y - shift_all, 0, None).mean(axis=0)
                alt_rates_joint = 1.0 / np.clip(alt_Y - shift_all, 0, None).mean(axis=0)
                h2_joint = exponential_hellinger2_nd(null_rates_joint, alt_rates_joint)
            elif is_double_exponential:
                # laplace.fit's MLE per column: loc=median, scale=mean
                # absolute deviation from that median -- no shift/clip
                # needed (unlike "exp" above), since a Laplace's support is
                # all of R rather than being cut off at a fitted floor.
                null_loc_joint, null_scale_joint = zip(*(laplace.fit(null_Y[:, j]) for j in range(null_Y.shape[1])))
                alt_loc_joint, alt_scale_joint = zip(*(laplace.fit(alt_Y[:, j]) for j in range(alt_Y.shape[1])))
                h2_joint = laplace_hellinger2_nd(null_loc_joint, null_scale_joint, alt_loc_joint, alt_scale_joint)
            else:
                n = len(avg_cols)
                null_cov = np.cov(null_Y, rowvar=False).reshape(n, n)
                alt_cov = np.cov(alt_Y, rowvar=False).reshape(n, n)
                h2_joint = gaussian_hellinger2_nd(null_Y.mean(axis=0), null_cov, alt_Y.mean(axis=0), alt_cov)

        num_plots = len(avg_cols)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), squeeze=False)
        axes = axes[0]

        # Fixed Null/Alt colors (not per-topology self.topo_colors) so Null
        # stays visually distinct from Alt's red on every panel -- matches
        # the Null/Alt convention plot_topology_pairs/correlation use.
        null_color = self.topo_colors['ABBA']
        for ax, col in zip(axes, avg_cols):
            topo_name = rename_map.get(col, col)
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            vals = self.df[col].to_numpy(dtype=float)
            # Percentile-clipped (not full min/max) range for both the fit
            # curve's x_grid and the axis view below -- a handful of extreme
            # rows would otherwise stretch x_grid into the far gaussian/exp
            # tail, where the pdf underflows toward 0 and forces a
            # comically large log-y range that dwarfs the real histogram.
            p_lo, p_hi = np.percentile(vals, [1, 99])
            if p_hi <= p_lo:
                p_lo, p_hi = vals.min(), vals.max()
            margin = (p_hi - p_lo) * 0.15 if p_hi > p_lo else 1.0
            x_grid = np.linspace(p_lo - margin, p_hi + margin, 200)

            null_vals = self.df.loc[labels == 'Null', col]
            alt_vals = self.df.loc[labels == 'Alt', col]
            # Same 1st-percentile floor as x_grid/p_lo above, not the raw
            # min -- a single extreme outlier below the bulk of the data
            # (seen directly: a w10_s10 raw dstar column with std~300 but
            # one row at -10173) inflates scale/deflates rate until the fit
            # curve is flat over any reasonably-zoomed view. Points still
            # under the floor get clipped to it (scale=0 contribution)
            # below so expon.pdf's domain (x >= shift) stays valid.
            shift = p_lo if is_exponential else None

            # Gaussian mean/std fits, computed here (before binning below)
            # so bin width can be sized off std_null/std_alt directly.
            mu_null = std_null = mu_alt = std_alt = None
            if not is_exponential and not is_double_exponential:
                if len(null_vals) > 1:
                    mu_null, std_null = norm.fit(null_vals)
                if len(alt_vals) > 1:
                    mu_alt, std_alt = norm.fit(alt_vals)

            # Bin width from the SMALLER of the two fitted stds (gaussian
            # only -- exp/dexp have no directly comparable std), not
            # FD/IQR 'auto': sizing off the tighter cluster keeps it
            # resolved instead of smoothed away to match the wider one's
            # scale, while the wider cluster just gets more, finer bins
            # across its own spread. Mirrors phlag.py PhlagPlotter's
            # _compute_bin_edges (same /4 divisor, same [15, 60] clip) but
            # with min std instead of max, since there's no second row of
            # histograms here needing to share one bin count. Falls back to
            # the FD-based _safe_bin_edges when neither std is available
            # (exp/dexp, or too little data to fit a std).
            combined_vals = np.concatenate([
                null_vals.to_numpy(dtype=float), alt_vals.to_numpy(dtype=float)
            ])
            candidate_stds = [s for s in (std_null, std_alt) if s is not None and s > 0]
            if candidate_stds:
                bin_width = max(min(candidate_stds) / 4, 1e-6)
                n_bins = int(np.clip(np.ceil((p_hi - p_lo) / bin_width), 15, 60))
                bin_edges_topo = np.linspace(p_lo, p_hi, n_bins + 1)
            else:
                bin_edges_topo = self._safe_bin_edges(combined_vals, p_lo, p_hi)

            if len(null_vals) > 0:
                sns.histplot(null_vals, ax=ax, stat='density', element='step', kde=False, alpha=0.35, color=null_color, label='Null Histogram', bins=bin_edges_topo)
            if len(alt_vals) > 0:
                sns.histplot(alt_vals, ax=ax, stat='density', element='step', kde=False, alpha=0.35, color='#E05638', label='Alt Histogram', bins=bin_edges_topo)

            null_rate = alt_rate = None
            if len(null_vals) > 1:
                if is_exponential:
                    scale_null = np.clip(null_vals.to_numpy(dtype=float) - shift, 0, None).mean()
                    null_rate = 1.0 / scale_null
                    mean_null = shift + scale_null
                    ax.plot(x_grid, expon.pdf(x_grid - shift, scale=scale_null), color=null_color, linewidth=2.2, label='Null Fit')
                    ax.axvline(mean_null, color=null_color, linestyle='--', linewidth=1.5)
                    ax.text(mean_null, 0.90, f"$\\lambda_{{null}}={null_rate:.3g}$", transform=trans, color=null_color, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                elif is_double_exponential:
                    loc_null, scale_null = laplace.fit(null_vals)
                    ax.plot(x_grid, laplace.pdf(x_grid, loc_null, scale_null), color=null_color, linewidth=2.2, label='Null Fit')
                    ax.axvline(loc_null, color=null_color, linestyle='--', linewidth=1.5)
                    ax.text(loc_null, 0.90, f"$\\mu_{{null}}={loc_null:.4g}$", transform=trans, color=null_color, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.text(loc_null + scale_null, 0.82, f"$b_{{null}}={scale_null:.4g}$", transform=trans, color=null_color, fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                else:
                    ax.plot(x_grid, norm.pdf(x_grid, mu_null, std_null), color=null_color, linewidth=2.2, label='Null Fit')
                    ax.axvline(mu_null, color=null_color, linestyle='--', linewidth=1.5)
                    ax.text(mu_null, 0.90, f"$\\mu_{{null}}={format_adaptive(mu_null, mu_alt)}$", transform=trans, color=null_color, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.text(mu_null + std_null, 0.82, f"$\\sigma_{{null}}={format_adaptive(std_null, std_alt)}$", transform=trans, color=null_color, fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            if len(alt_vals) > 1:
                if is_exponential:
                    scale_alt = np.clip(alt_vals.to_numpy(dtype=float) - shift, 0, None).mean()
                    alt_rate = 1.0 / scale_alt
                    mean_alt = shift + scale_alt
                    ax.plot(x_grid, expon.pdf(x_grid - shift, scale=scale_alt), color='#E05638', linewidth=2.2, linestyle='--', label='Alt Fit')
                    ax.axvline(mean_alt, color='#E05638', linestyle=':', linewidth=1.5)
                    ax.text(mean_alt, 0.75, f"$\\lambda_{{alt}}={alt_rate:.3g}$", transform=trans, color='#E05638', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                elif is_double_exponential:
                    loc_alt, scale_alt = laplace.fit(alt_vals)
                    ax.plot(x_grid, laplace.pdf(x_grid, loc_alt, scale_alt), color='#E05638', linewidth=2.2, linestyle='--', label='Alt Fit')
                    ax.axvline(loc_alt, color='#E05638', linestyle=':', linewidth=1.5)
                    ax.text(loc_alt, 0.75, f"$\\mu_{{alt}}={loc_alt:.4g}$", transform=trans, color='#E05638', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.text(loc_alt + scale_alt, 0.67, f"$b_{{alt}}={scale_alt:.4g}$", transform=trans, color='#E05638', fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                else:
                    ax.plot(x_grid, norm.pdf(x_grid, mu_alt, std_alt), color='#E05638', linewidth=2.2, linestyle='--', label='Alt Fit')
                    ax.axvline(mu_alt, color='#E05638', linestyle=':', linewidth=1.5)
                    ax.text(mu_alt, 0.75, f"$\\mu_{{alt}}={format_adaptive(mu_alt, mu_null)}$", transform=trans, color='#E05638', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.text(mu_alt + std_alt, 0.67, f"$\\sigma_{{alt}}={format_adaptive(std_alt, std_null)}$", transform=trans, color='#E05638', fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            # Percentile-clipped view (not the fit itself, which is still
            # computed on the full null_vals/alt_vals) regardless of
            # log/linear -- a handful of extreme rows would otherwise
            # stretch the axis out until the real bulk of the distribution
            # is squeezed into a sliver of pixels near the middle.
            ax.set_xlim(x_grid[0], x_grid[-1])

            if self._needs_log_scale(combined_vals):
                # Auto-detected (see _needs_log_scale): a straight-line decay
                # reads as exponential and a downward curve reads as Gaussian
                # regardless of how high- or low-variance the data is, the
                # same treatment plot_sums/_stacked_sum_hist applies.
                ax.set_yscale('log')

                # Well-separated Null/Alt (e.g. a strong H^2) still leaves
                # each fit curve underflowing toward 0 out under the OTHER
                # cluster's peak, even within x_grid's own percentile-
                # clipped range -- log-scale would otherwise auto-expand the
                # bottom to fit that underflow, dwarfing the real histogram
                # bars several times taller than tall. Cap the visible
                # dynamic range to 5 decades below the tallest bar/curve
                # actually drawn instead.
                _, ymax_auto = ax.get_ylim()
                ax.set_ylim(bottom=ymax_auto * 1e-5, top=ymax_auto)

            ax.set_title(f'Topology: {topo_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{norm_label} Score')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.5)

        title = f'Topology Histograms & {self.distribution} Fits: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
        if h2_joint is not None:
            title += f'  ($H_d^2$={format_adaptive(h2_joint, min_decimals=3)})'
        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout()

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_dist = os.path.join(output_dir, 'dist.png')
        plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
        print(f"Saved topology histogram chart to: {save_path_dist}")
        plt.close()

    def _resolve_topo_columns_strict(self):
        """
        Like resolve_topology_columns, but additionally dedupes to one column
        per ABBA/BABA/AABB name (first match wins, mirroring
        write_ground_truth_stats' col_for_topo) and returns None unless all
        three are present -- both the 3D plot and the correlation heatmap
        need exactly the three raw topology axes (not the 2D c*ILR1/c*ILR2
        columns an --ilr file carries instead).
        """
        avg_cols, rename_map = self.resolve_topology_columns(self.df, self.topologies)
        topo_order = ['ABBA', 'BABA', 'AABB']
        col_for_topo = {}
        for col in avg_cols:
            mapped = rename_map.get(col, col)
            if mapped in topo_order and mapped not in col_for_topo:
                col_for_topo[mapped] = col
        if not all(t in col_for_topo for t in topo_order):
            return None
        return col_for_topo

    def _compute_null_alt_labels(self, df=None):
        """
        Shared ground-truth Null/Alt per-window labeling, factored out so
        plot_topology_pairs and plot_correlation's null/alt split can reuse
        the exact same parse_pattern_string logic
        instead of re-deriving it. Returns a numpy object array of
        'Null'/'Alt' (one per row of df, positional) or None if no
        ground-truth locus pattern is resolvable. df defaults to self.df;
        plot_quartet_counts passes its own quartet_counts_df instead, since
        that file's 'pos' column is a separate per-window table.
        """
        if df is None:
            df = self.df
        pattern_str = self._ground_truth_pattern()
        if not (pattern_str and 'pos' in df.columns and len(df) > 0):
            return None
        from .utils import parse_pattern_string
        positions = df['pos'].to_numpy()
        total_span = positions.max()
        _, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)
        if not anomaly_intervals:
            return None
        labels = np.full(len(positions), 'Null', dtype=object)
        for idx, pos in enumerate(positions):
            for start_bp, end_bp in anomaly_intervals:
                if start_bp <= pos <= end_bp:
                    labels[idx] = 'Alt'
                    break
        return labels

    def plot_topology_pairs(self):
        """
        Per-window ABBA/BABA/AABB points with the same Null/Alt ground-truth
        coloring plot_topology_scatter shades as an axvspan, projected onto
        each of the three 2D axis pairs (ABBA-vs-BABA, ABBA-vs-AABB,
        BABA-vs-AABB) -- laid out as a single 1x3 subplot grid saved to one
        PNG (replaces the old single 3D ABBA/BABA/AABB scatter, which needed
        rotation to read and isn't legible in a static PNG).
        """
        col_for_topo = self._resolve_topo_columns_strict()
        if col_for_topo is None:
            print("Need all three ABBA/BABA/AABB topology columns for a pairwise topology plot; skipping.")
            return

        vals = {t: self.df[col_for_topo[t]].to_numpy(dtype=float) for t in ('ABBA', 'BABA', 'AABB')}
        labels = self._compute_null_alt_labels()
        label_colors = {'Null': self.topo_colors['ABBA'], 'Alt': '#E05638'}

        pairs = [('ABBA', 'BABA'), ('ABBA', 'AABB'), ('BABA', 'AABB')]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax, (xt, yt) in zip(axes, pairs):
            x, y = vals[xt], vals[yt]
            if labels is not None:
                for lbl in ('Null', 'Alt'):
                    mask = labels == lbl
                    if mask.any():
                        ax.scatter(x[mask], y[mask], c=label_colors[lbl], label=lbl, alpha=0.6, s=14, edgecolors='none')
                ax.legend(loc='upper right', framealpha=0.9)
            else:
                ax.scatter(x, y, c=self.topo_colors['ABBA'], alpha=0.6, s=14, edgecolors='none')
            ax.set_xlabel(xt, fontsize=10, labelpad=6)
            ax.set_ylabel(yt, fontsize=10, labelpad=6)
            ax.set_title(f'{xt} vs {yt}', fontsize=11)

        title = f'Pairwise Topology Space: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_pairs = os.path.join(output_dir, 'topology_pairs.png')
        plt.savefig(save_path_pairs, dpi=300)
        print(f"Saved pairwise topology plot to: {save_path_pairs}")
        plt.close()

    def plot_correlation(self):
        """
        Heatmap(s) of the pairwise Pearson correlation between the three raw
        topology score columns (ABBA/BABA/AABB) across windows. When a
        ground-truth locus pattern is resolvable (same per-window Null/Alt
        split plot_topology_pairs uses), shows two heatmaps side by side --
        one computed over Null-only windows, one over Alt-only windows -- on
        a single figure, since the two classes can have meaningfully
        different topology correlation structure. Falls back to one
        aggregate heatmap (the original behavior) when no ground-truth
        pattern is resolvable, the same fallback convention plot_topology_pairs
        uses for its own coloring.
        """
        col_for_topo = self._resolve_topo_columns_strict()
        if col_for_topo is None:
            print("Need all three ABBA/BABA/AABB topology columns for a correlation plot; skipping.")
            return

        topo_order = ['ABBA', 'BABA', 'AABB']
        corr_df = self.df[[col_for_topo[t] for t in topo_order]].rename(
            columns={col_for_topo[t]: t for t in topo_order}
        )

        title = f'Topology Score Correlation: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'

        labels = self._compute_null_alt_labels()

        if labels is None:
            corr = corr_df.corr(method='pearson')
            plt.figure(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                        square=True, cbar_kws={'label': 'Pearson r'})
            plt.title(title, fontsize=13, fontweight='bold', pad=10)
            plt.tight_layout()
        else:
            n_by_lbl = {}
            corr_by_lbl = {}
            for lbl in ('Null', 'Alt'):
                lbl_mask = labels == lbl
                n = int(lbl_mask.sum())
                n_by_lbl[lbl] = n
                corr_by_lbl[lbl] = corr_df.loc[lbl_mask].corr(method='pearson') if n >= 2 else None

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            n_topo = len(topo_order)
            triu_mask = ~np.triu(np.ones((n_topo, n_topo), dtype=bool), k=1)

            null_ax, alt_ax = axes
            null_corr = corr_by_lbl['Null']
            if null_corr is not None:
                sns.heatmap(null_corr, mask=triu_mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                            square=True, cbar_kws={'label': 'Pearson r'}, ax=null_ax)
            else:
                null_ax.text(0.5, 0.5, 'Not enough windows', ha='center', va='center', transform=null_ax.transAxes)
                null_ax.set_xticks([])
                null_ax.set_yticks([])
            null_ax.set_title(f'Null (n={n_by_lbl["Null"]})', fontsize=11, fontweight='bold')

            alt_corr = corr_by_lbl['Alt']
            if null_corr is not None and alt_corr is not None:
                diff_corr = alt_corr - null_corr
                vmax = max(np.abs(diff_corr.values).max(), 0.05)
                vmin = -vmax
                sns.heatmap(diff_corr, mask=triu_mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=vmin, vmax=vmax,
                            square=True, cbar_kws={'label': 'Alt − Null r'}, ax=alt_ax)
            else:
                alt_ax.text(0.5, 0.5, 'Not enough windows', ha='center', va='center', transform=alt_ax.transAxes)
                alt_ax.set_xticks([])
                alt_ax.set_yticks([])
            alt_ax.set_title(f'Alt (n={n_by_lbl["Alt"]})', fontsize=11, fontweight='bold')

            fig.suptitle(title, fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_corr = os.path.join(output_dir, 'correlation.png')
        plt.savefig(save_path_corr, dpi=300, bbox_inches='tight')
        print(f"Saved topology correlation heatmap to: {save_path_corr}")
        plt.close()

    def _load_quartet_counts_df(self):
        """
        Shared quartet_counts.tsv loader/validator for plot_quartet_counts
        and plot_quartet_dists -- both need the same optional per-window,
        per-topology zero/negative/positive raw per-site score counts (see
        caster/dstar.cpp's scoreIntervalWithCounts and sequence.hpp's
        Quadripartition::Gene::signCounts), parsed and column-checked the
        same way. Purely diagnostic: this file is never read by phlag's
        HMM and has no bearing on scores.tsv/chunk_scores.tsv. Returns the
        DataFrame, or None (after printing why) if it's missing, unreadable,
        or missing expected columns -- callers skip gracefully rather than
        raising, matching the convention the other optional plots in this
        class already follow.
        """
        quartet_counts_path = pathlib.Path(self.scores_file).parent / "quartet_counts.tsv"
        if not quartet_counts_path.exists():
            print(f"No quartet_counts.tsv found at '{quartet_counts_path}'; skipping quartet counts plot.")
            return None

        try:
            quartet_counts_df = pd.read_csv(quartet_counts_path, sep='\t')
        except Exception as e:
            print(f"Error reading quartet counts file {quartet_counts_path}: {e}; skipping quartet counts plot.")
            return None

        if 'pos' not in quartet_counts_df.columns:
            print(f"Quartet counts file '{quartet_counts_path}' has no 'pos' column; skipping quartet counts plot.")
            return None

        topo_order = ['ABBA', 'BABA', 'AABB']
        kind_cols = {'zero': '_zero', 'negative': '_neg', 'positive': '_pos'}
        missing = [f"{t}{suffix}" for t in topo_order for suffix in kind_cols.values()
                   if f"{t}{suffix}" not in quartet_counts_df.columns]
        if missing:
            print(f"Quartet counts file '{quartet_counts_path}' is missing expected column(s) {missing}; skipping quartet counts plot.")
            return None
        return quartet_counts_df

    def plot_quartet_counts(self):
        """
        Per-window line plot of quartet_counts.tsv (see
        _load_quartet_counts_df): one subplot per topology, each with the
        zero/negative/positive count columns plotted against genomic
        position, shaded with the same ground-truth Alt regions as the
        other window-position plots in this file. Skips gracefully when
        quartet_counts.tsv is missing/malformed (_load_quartet_counts_df
        prints why).
        """
        quartet_counts_df = self._load_quartet_counts_df()
        if quartet_counts_df is None:
            return

        topo_order = ['ABBA', 'BABA', 'AABB']
        kind_colors = {'zero': '#7F7F7F', 'negative': self.topo_colors['BABA'], 'positive': self.topo_colors['ABBA']}
        kind_cols = {'zero': '_zero', 'negative': '_neg', 'positive': '_pos'}

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        for ax, topo in zip(axes, topo_order):
            topo_vals = []
            for kind, suffix in kind_cols.items():
                col_vals = quartet_counts_df[f"{topo}{suffix}"].to_numpy(dtype=float)
                ax.plot(quartet_counts_df['pos'], col_vals, color=kind_colors[kind], label=kind, linewidth=1.2)
                topo_vals.append(col_vals)
            # Auto-detected (see _needs_log_scale): zero/negative/positive
            # site counts can differ by orders of magnitude per topology
            # (e.g. a near-flat 'zero' count towering over sparse
            # 'negative'/'positive' spikes), squashing the smaller lines
            # flat on a linear axis.
            if self._needs_log_scale(np.concatenate(topo_vals)):
                ax.set_yscale('log')
            self._shade_locus_pattern(ax)
            ax.set_title(f'Topology: {topo}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Site count')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.5)

        axes[-1].set_xlabel('Genomic Position (pos)', fontsize=11, labelpad=8)

        title = f'Per-Site Quartet Counts: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'quartet_counts.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved quartet counts plot to: {save_path}")
        plt.close()

    def plot_quartet_dists(self):
        """
        Distributional companion to plot_quartet_counts (see
        _load_quartet_counts_df): 3 (topology) x 2 (Null row / Alt row)
        grid, each subplot a KDE of the zero/negative/positive count
        columns for that topology's windows in that ground-truth class --
        shows how each count's distribution shifts between Null and Alt,
        rather than plot_quartet_counts' per-window line-over-position
        view. Additionally skips (prints why) when no ground-truth locus
        pattern is resolvable, since the Null/Alt split needs one.
        """
        quartet_counts_df = self._load_quartet_counts_df()
        if quartet_counts_df is None:
            return

        topo_order = ['ABBA', 'BABA', 'AABB']
        kind_colors = {'zero': '#7F7F7F', 'negative': self.topo_colors['BABA'], 'positive': self.topo_colors['ABBA']}
        kind_cols = {'zero': '_zero', 'negative': '_neg', 'positive': '_pos'}

        labels = self._compute_null_alt_labels(quartet_counts_df)
        if labels is None:
            print("No resolvable ground-truth locus pattern; skipping quartet counts distribution plot.")
            return

        row_order = ['Null', 'Alt']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex='col')

        for col_idx, topo in enumerate(topo_order):
            col_vals = []
            for row_idx, row_label in enumerate(row_order):
                ax = axes[row_idx][col_idx]
                row_mask = labels == row_label
                for kind, suffix in kind_cols.items():
                    vals = quartet_counts_df.loc[row_mask, f"{topo}{suffix}"].to_numpy(dtype=float)
                    col_vals.append(vals)
                    if len(vals) > 1 and np.ptp(vals) > 0:
                        sns.kdeplot(vals, ax=ax, color=kind_colors[kind], label=kind, linewidth=1.4)
                    elif len(vals) > 0:
                        # kdeplot needs variance to fit a bandwidth -- a
                        # constant (or single-row) column still gets a
                        # visible marker instead of silently disappearing.
                        ax.axvline(vals[0], color=kind_colors[kind], label=kind, linewidth=1.4)
                ax.set_title(f'{topo} ({row_label})', fontsize=11, fontweight='bold')
                ax.set_xlabel('Site count')
                ax.set_ylabel('Density')
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
                ax.grid(True, linestyle=':', alpha=0.5)

            # Auto-detected (see _needs_log_scale) per column, off both rows'
            # zero/negative/positive counts together -- a near-flat 'zero'
            # count sitting two-plus orders of magnitude above the sparse
            # 'negative'/'positive' spikes (as in the real data, ~1000 vs
            # ~10) otherwise squashes the latter into a sliver near 0 on a
            # linear x-axis. Decided once per column (sharex='col' ties both
            # rows' scale together anyway) rather than per subplot.
            if self._needs_log_scale(np.concatenate(col_vals)):
                axes[0][col_idx].set_xscale('log')
                axes[1][col_idx].set_xscale('log')

        title = f'Per-Site Quartet Count Distributions: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'quartet_dists.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved quartet counts distribution plot to: {save_path}")
        plt.close()

    def _stacked_sum_hist(self, ax, col_for_topo, title):
        """
        histograms row_sum = ABBA+BABA+AABB (the loaded columns as-is -- for
        dstar.cpp's window/step mode these are already per-site averages,
        see the "dstar window average site-count fix"), with bin count
        picked by _safe_bin_edges (Freedman-Diaconis' bin width from the
        IQR, clamped to [10, 100] bins) rather than a fixed bin count,
        since the topology score scale varies enormously with window size.

        Each bar's total height is a real count, so the outline of the
        stack is the actual row_sum distribution -- when _needs_log_scale
        auto-detects the bin counts span it, a straight-line decay on the
        resulting log axis reads as exponential, a downward curve as
        Gaussian, regardless of how high- or low-variance the data is.
        The bar is then split into 3 stacked
        segments by that bin's mean per-topology value, clamped to >= 0 and
        renormalized to sum to 1 (CASTER's scoreCnt()-based columns can be
        negative; a topology with a negative bin-mean just contributes no
        visible segment there rather than an invalid negative-height slice,
        falling back to an equal 3-way split only if all three are <= 0).

        The x-axis view is clipped to the row_sum's own [1st, 99th]
        percentile range (padded 8%) rather than its full min/max -- a
        handful of extreme rows would otherwise stretch the axis out until
        the real bulk of the distribution is squeezed into a sliver of
        pixels near the middle.
        """
        topo_order = ['ABBA', 'BABA', 'AABB']
        vals = {t: self.df[col_for_topo[t]].to_numpy(dtype=float) for t in topo_order}

        row_sum = sum(vals[t] for t in topo_order)
        if len(row_sum) < 2:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=11, fontweight='bold')
            return

        p_lo, p_hi = np.percentile(row_sum, [1, 99])
        if p_hi <= p_lo:
            p_lo, p_hi = row_sum.min(), row_sum.max()
        # Explicit clamped edges (see _safe_bin_edges) instead of a raw
        # bins='auto' -- 'auto's own FD bin-width estimate comes from
        # row_sum's FULL-array IQR, not the (p_lo, p_hi)-clipped range
        # passed as range= below, which only bounds where edges start/end.
        # A near-zero IQR (row_sum's bulk sitting almost on top of itself,
        # common at small window sizes) still requests an astronomical bin
        # count over that clipped span and OOMs np.linspace regardless.
        bin_edges = self._safe_bin_edges(row_sum, p_lo, p_hi)
        bin_idx = np.clip(np.digitize(row_sum, bin_edges, right=False) - 1, 0, len(bin_edges) - 2)
        bin_widths = np.diff(bin_edges)
        n_bins = len(bin_edges) - 1

        counts = np.array([np.count_nonzero(bin_idx == i) for i in range(n_bins)], dtype=float)
        avg_per_bin = {
            t: np.array([vals[t][bin_idx == i].mean() if counts[i] > 0 else 0.0 for i in range(n_bins)])
            for t in topo_order
        }

        positive = {t: np.clip(avg_per_bin[t], 0, None) for t in topo_order}
        positive_total = sum(positive[t] for t in topo_order)
        all_nonpositive = positive_total <= 0
        safe_total = np.where(all_nonpositive, 1.0, positive_total)

        bottom = np.zeros(n_bins)
        for t in topo_order:
            proportion = np.where(all_nonpositive, 1.0 / len(topo_order), positive[t] / safe_total)
            seg_height = proportion * counts
            ax.bar(bin_edges[:-1], seg_height, width=bin_widths, bottom=bottom,
                   align='edge', color=self.topo_colors[t], label=t, edgecolor='white', linewidth=0.3)
            bottom += seg_height

        if p_hi > p_lo:
            pad = (p_hi - p_lo) * 0.08
            ax.set_xlim(p_lo - pad, p_hi + pad)
        if self._needs_log_scale(counts):
            ax.set_yscale('log')

        ax.set_xlabel('Row Sum (ABBA + BABA + AABB)')
        ax.set_ylabel('Count (split by topology share)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5)

    def plot_sums(self):
        """
        sums.png: single log-y count histogram of row_sum = ABBA+BABA+AABB
        (the loaded columns as-is), each bar split into 3 stacked segments
        by that bin's topology composition -- see _stacked_sum_hist.
        """
        col_for_topo = self._resolve_topo_columns_strict()
        if col_for_topo is None:
            print("Need all three ABBA/BABA/AABB topology columns for a sums plot; skipping.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        self._stacked_sum_hist(ax, col_for_topo, title='Over Windows')

        title = f'Topology Score Sums: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'sums.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved topology score sums plot to: {save_path}")
        plt.close()


def write_ground_truth_stats(scores_file, output_dir, locus_pattern=None, topologies=None, dist_type="gaussian"):
    """
    Writes gt_stats.txt (Null/Alt/Overall mean+covariance across the 3
    topology dimensions, ABBA/BABA/AABB order, plus each label's per-topology
    Fisher-Pearson skewness -- scipy.stats.skew, bias=True, i.e. g1 =
    m3/m2^1.5 -- as "<label>Skewness") next to scores_file, so
    downstream ground-truth-divergence consumers (phlag.py's em_gt_hd) can
    read the joint mean/covariance directly instead of re-loading and
    re-splitting scores_file themselves every time.

    Called unconditionally after scores.tsv/chunk_scores.tsv is finalized,
    independent of --plot (CasterPlotter itself is only constructed when
    plotting is requested, but this data artifact -- like scores.tsv itself
    -- should exist regardless). Overall needs only >=2 windows total, always
    computable; Null/Alt additionally need a resolvable ground-truth locus
    pattern and >=2 windows in each class. Any missing piece degrades
    gracefully (skips that section, or writes nothing at all) rather than
    raising, same convention as phlag.py's own ground-truth handling.

    dist_type="gaussian" (default) computes Hellinger2 via
    gaussian_hellinger2_nd on the joint 3D mean/covariance. dist_type=
    "exp" instead shifts each of the 3 topology columns by its own
    1st-percentile floor (over the whole file, so Null and Alt stay
    shifted by the same reference and remain comparable; a percentile
    rather than the true min so one extreme outlier can't drag it far
    below the bulk of the data and deflate every rate -- see the
    identical fix in plot_distribution/CasterPlotter._stacked_sum_hist),
    clips anything still negative after that shift to 0, then reads
    Exponential rates directly off the shifted Null/Alt means (mean of
    shifted data == 1/rate) and computes Hellinger2 via
    exponential_hellinger2_nd. Mean/covariance bookkeeping in gt_stats.txt
    itself is otherwise unaffected -- the shift only changes what the means
    represent (shifted-space means, whose reciprocal is the rate) and only
    when dist_type="exp". dist_type="dexp" (double-exponential/Laplace,
    unshifted -- a Laplace's support is all of R) instead fits loc/scale
    per topology via laplace.fit on the raw Null/Alt values and computes
    Hellinger2 via laplace_hellinger2_nd.
    """
    from .utils import parse_pattern_string, write_gt_stats_file, mardia_skewness, gaussian_hellinger2_nd, exponential_hellinger2_nd, laplace_hellinger2_nd, GT_STATS_FILENAME

    try:
        df = pd.read_csv(scores_file, sep='\t')
    except Exception:
        return

    avg_cols, rename_map = CasterPlotter.resolve_topology_columns(df, topologies)
    topo_order = ['ABBA', 'BABA', 'AABB']
    col_for_topo = {}
    for col in avg_cols:
        mapped = rename_map.get(col, col)
        if mapped in topo_order and mapped not in col_for_topo:
            col_for_topo[mapped] = col
    if 'pos' not in df.columns or not all(t in col_for_topo for t in topo_order):
        return

    Y = df[[col_for_topo[t] for t in topo_order]].to_numpy(dtype=float)
    if len(Y) < 2:
        return

    if dist_type == "exp":
        Y = np.clip(Y - np.percentile(Y, 1, axis=0), 0, None)

    stats = {"Overall": (Y.mean(axis=0), np.cov(Y, rowvar=False).reshape(3, 3))}
    stats["OverallSkewness"] = skew(Y, axis=0, bias=True).tolist()
    stats["OverallMardiaSkewness"] = mardia_skewness(Y)

    # Within-window variance/mean: how much the raw per-site topology
    # scores vary WITHIN a single window (what a coarser window's own
    # averaging step is smoothing over), as opposed to the mean/covariance
    # above (spread BETWEEN whole-window values across the file). Needs
    # the sibling non-overlapping w1_s1 (per-site) scores.tsv for this same
    # node/pattern -- only meaningful for non-overlapping dstar windows
    # (step == window), since bucketing raw sites by pos // window
    # reconstructs the real window boundaries only then. Degrades
    # gracefully (adds nothing) if the sibling file, mode, or window size
    # don't support it. Buckets with <2 sites are dropped (variance
    # undefined) -- same convention as bench/caster.ipynb's
    # topology_variance_within_window, which this mirrors.
    site_df = bucket_means = bucket_vars = None
    ws = parse_ws_from_path(pathlib.Path(scores_file))
    if ws is not None:
        ws_mode, ws_window, ws_step = ws[0], ws[1], ws[2]
        if ws_mode == 'w' and ws_window > 1 and ws_window == ws_step:
            site_path = substitute_ws_in_path(pathlib.Path(scores_file), 1, 1)
            if site_path is not None and site_path != pathlib.Path(scores_file) and site_path.exists():
                try:
                    site_df = pd.read_csv(site_path, sep='\t', usecols=['pos', 'c*ABBA', 'c*BABA', 'c*AABB'])
                except Exception:
                    site_df = None
                if site_df is not None and len(site_df) >= 2:
                    site_df['_bucket'] = site_df['pos'] // ws_window
                    grouped = site_df.groupby('_bucket')[['c*ABBA', 'c*BABA', 'c*AABB']]
                    keep = grouped.size()
                    keep = keep[keep >= 2].index
                    if len(keep):
                        bucket_means = grouped.mean().loc[keep]
                        bucket_vars = grouped.var().loc[keep]
                    else:
                        site_df = None

    if bucket_means is not None:
        stats["OverallWithinMean"] = bucket_means.mean(axis=0).to_numpy().tolist()
        stats["OverallWithinVariance"] = bucket_vars.mean(axis=0).to_numpy().tolist()

    pattern_str = locus_pattern
    if not pattern_str:
        m = re.search(r'((?:[an]\d+(?:-[an]?\d+)?(?:_)?)+|\d+-\d+(?:[;_,]\d+-\d+)*)', str(scores_file))
        pattern_str = m.group(1) if m else None

    if pattern_str:
        positions = df['pos'].to_numpy()
        total_span = positions.max() if len(positions) else None
        blocks, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)
        if blocks:
            y_true = np.zeros(len(positions), dtype=int)
            for idx, pos in enumerate(positions):
                for start_bp, end_bp in anomaly_intervals:
                    if start_bp <= pos <= end_bp:
                        y_true[idx] = 1
                        break
            # Row-normalized by each state's own count of "from" occurrences
            # (not total windows), matching phlag.py's fitted transition
            # matrix convention so the two are directly comparable.
            n_null_from = int(np.sum(y_true[:-1] == 0))
            n_alt_from = int(np.sum(y_true[:-1] == 1))
            n_null_to_alt = int(np.sum((y_true[:-1] == 0) & (y_true[1:] == 1)))
            n_alt_to_null = int(np.sum((y_true[:-1] == 1) & (y_true[1:] == 0)))
            p_null_alt = n_null_to_alt / n_null_from if n_null_from > 0 else float("nan")
            p_alt_null = n_alt_to_null / n_alt_from if n_alt_from > 0 else float("nan")
            stats["TransitionMatrix"] = [[1.0 - p_null_alt, p_null_alt], [p_alt_null, 1.0 - p_alt_null]]

            null_vals = Y[y_true == 0]
            alt_vals = Y[y_true == 1]
            if len(null_vals) > 1:
                stats["Null"] = (null_vals.mean(axis=0), np.cov(null_vals, rowvar=False).reshape(3, 3))
                stats["NullSkewness"] = skew(null_vals, axis=0, bias=True).tolist()
                stats["NullMardiaSkewness"] = mardia_skewness(null_vals)
            if len(alt_vals) > 1:
                stats["Alt"] = (alt_vals.mean(axis=0), np.cov(alt_vals, rowvar=False).reshape(3, 3))
                stats["AltSkewness"] = skew(alt_vals, axis=0, bias=True).tolist()
                stats["AltMardiaSkewness"] = mardia_skewness(alt_vals)

            if bucket_means is not None:
                # Each bucket (window) is labeled Null/Alt by its MAJORITY
                # site, same convention as the window-level y_true above --
                # a window straddling the Null/Alt boundary still gets one
                # label rather than being dropped.
                site_pos = site_df['pos'].to_numpy()
                site_is_alt = np.zeros(len(site_df), dtype=bool)
                for start_bp, end_bp in anomaly_intervals:
                    site_is_alt |= (site_pos >= start_bp) & (site_pos <= end_bp)
                site_df['_is_alt'] = site_is_alt
                region = site_df.groupby('_bucket')['_is_alt'].mean().reindex(bucket_means.index).ge(0.5)
                null_idx, alt_idx = region[~region].index, region[region].index
                if len(null_idx):
                    stats["NullWithinMean"] = bucket_means.loc[null_idx].mean(axis=0).to_numpy().tolist()
                    stats["NullWithinVariance"] = bucket_vars.loc[null_idx].mean(axis=0).to_numpy().tolist()
                if len(alt_idx):
                    stats["AltWithinMean"] = bucket_means.loc[alt_idx].mean(axis=0).to_numpy().tolist()
                    stats["AltWithinVariance"] = bucket_vars.loc[alt_idx].mean(axis=0).to_numpy().tolist()

            if "Null" in stats and "Alt" in stats:
                if dist_type == "exp":
                    stats["Hellinger2"] = exponential_hellinger2_nd(
                        1.0 / stats["Null"][0], 1.0 / stats["Alt"][0],
                    )
                elif dist_type == "dexp":
                    null_loc, null_scale = zip(*(laplace.fit(null_vals[:, j]) for j in range(null_vals.shape[1])))
                    alt_loc, alt_scale = zip(*(laplace.fit(alt_vals[:, j]) for j in range(alt_vals.shape[1])))
                    stats["Hellinger2"] = laplace_hellinger2_nd(null_loc, null_scale, alt_loc, alt_scale)
                else:
                    stats["Hellinger2"] = gaussian_hellinger2_nd(
                        stats["Null"][0], stats["Null"][1], stats["Alt"][0], stats["Alt"][1],
                    )

    output_path = pathlib.Path(output_dir) / GT_STATS_FILENAME
    write_gt_stats_file(output_path, stats)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Caster: Load scores and generate topology distribution and scatter plots."
    )

    parser.add_argument(
        "fasta_file",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help="Input FASTA file path"
    )
    
    # Recent flag & Left/Right indices
    parser.add_argument(
        "-r",
        "--recent",
        dest="recent",
        action="store_true",
        help="Use the most recently created FASTA file in store/msa/concat"
    )
    parser.add_argument(
        "-l",
        dest="left",
        type=int_or_abbrev,
        default=0,
        help="Left index of range (0-indexed, inclusive)"
    )
    parser.add_argument(
        "-R",
        "--right",
        dest="right",
        type=int_or_abbrev,
        required=False,
        default=None,
        help="Right index of range (0-indexed, exclusive)"
    )
    
    # Dstar parameters
    parser.add_argument(
        "-w",
        dest="window_size",
        type=int_or_abbrev,
        nargs="+",
        default=[50000],
        help="Window size (default: 50000 / 50k). Multiple space-separated "
             "values run one caster command per value (cartesian product "
             "with -s if it also has multiple values)."
    )
    parser.add_argument(
        "-n",
        dest="normalize",
        action="store_true",
        help="Normalize each site's c*ABBA/c*BABA/c*AABB by their sum into proportions (like caster-pair's q1/q2/q3), before window averaging"
    )
    parser.add_argument(
        "--norm-eps",
        dest="norm_eps",
        action="store_true",
        help=f"With -n/--normalize, guard the per-row c*ABBA+c*BABA+c*AABB sum "
             f"against near-zero (sign-cancelling) blowup: 1/3-fallback an "
             f"all-noise row and clamp the divisor's magnitude to at least "
             f"{DEFAULT_NORM_EPS} otherwise. Off by default to keep existing "
             f"normalize output bit-for-bit reproducible; on writes to its own "
             f"'normalize/norm-eps' cache entry instead of overwriting it."
    )
    parser.add_argument(
        "--exp-minus",
        dest="exp_minus",
        action="store_true",
        help="Replace each output topology column x with exp(-x), applied last "
             "(after -n/-i/-z). Reuses the untransformed sibling scores.tsv if "
             "present; writes to its own nested 'exp-minus' cache entry."
    )
    parser.add_argument(
        "-s",
        dest="step_size",
        type=step_size_or_fraction,
        nargs="+",
        default=[1.0],
        help="Step size (default: 1.0, i.e. non-overlapping / step==window). "
             "A value with a decimal point "
             "is treated as a ratio of -w's window size, multiplied out (e.g. "
             "-w 50000 -s 0.1 -> step=5000; -s 1.0 -> step=50000, i.e. "
             "non-overlapping). A whole value (e.g. 1000, 1k) is a literal/"
             "absolute step size. Multiple space-separated values run one "
             "caster command per value (cartesian product with -w if it also "
             "has multiple values)."
    )

    parser.add_argument(
        "--shift-caster",
        dest="shift_caster",
        action="store_true",
        default=False,
        help="Shift each window's reported pos right by window_size/2 so it "
             "marks the window's center instead of its left edge (default: omitted)"
    )
    parser.add_argument(
        "-m",
        dest="mapping",
        type=pathlib.Path,
        default=None,
        help="Optional population mapping file path"
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=["scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums"],
        default=None,
        help="List of plots to generate (choices: scatter, the topology "
             "scatter plot; dist, per-topology Null/Alt "
             "histograms with Gaussian fit overlays (requires a resolvable "
             "ground-truth locus pattern, else skipped); topology_pairs, "
             "per-window ABBA/BABA/AABB points projected onto each of the "
             "three 2D axis pairs (1x3 subplot grid, one PNG); correlation, "
             "a pairwise Pearson correlation heatmap of the same three "
             "columns (split into Null/Alt side-by-side heatmaps when a "
             "ground-truth pattern is resolvable); quartet_counts, per-topology "
             "raw per-site score quartet counts (zero/negative/positive) from "
             "the optional quartet_counts.tsv companion file -- two PNGs: "
             "quartet_counts.png (1x3 per-topology line plot over genomic "
             "position, ground-truth shaded) and quartet_dists.png (3x2 "
             "topology x Null/Alt grid of KDEs, requires a resolvable "
             "ground-truth locus pattern, else that second PNG alone is "
             "skipped); both require quartet_counts.tsv, else skipped "
             "entirely; "
             "sums, histogram of the summed ABBA+BABA+AABB topology score per "
             "row (bins=auto), each bar split into 3 stacked colors by that "
             "bin's average per-topology score. scatter/dist/sums/quartet_counts "
             "each auto-detect their own need for a log y-axis instead of a "
             "manual modifier (see CasterPlotter._needs_log_scale): log kicks "
             "in when the plotted values' 5th-to-95th-percentile spread (positive "
             "values only) covers at least 100x, since past that a linear axis "
             "can't resolve both ends at once. Default: all of the above when "
             "--bench is omitted, scatter only under --bench. Passing --plot "
             "with no choices explicitly requests all of the above, in "
             "either mode)",
    )
    parser.add_argument(
        "-t",
        "--topologies",
        dest="topologies",
        nargs="+",
        default=None,
        help="List of topologies to plot (default: all)"
    )
    parser.add_argument(
        "-d",
        "--dist-type",
        dest="dist_type",
        default="gaussian",
        choices=["gaussian", "gmm", "exp", "dexp"],
        help="Distribution type used for CasterPlotter's statistical fits (default: "
             "gaussian). dexp is a double-exponential/Laplace fit -- unlike exp's "
             "one-sided fit (cut off below a fitted floor), it's peaked at a fitted "
             "location and decays on both sides, matching data (like raw CASTER "
             "scoreCnt() topology sums) whose histogram doesn't have a hard left "
             "edge. No longer affects scores.tsv's output location -- that's "
             "shared across dist_types, see --bench."
    )
    parser.add_argument(
        "-o", dest="output_file", type=pathlib.Path, default=None,
        help="Path to save scores.tsv (its directory is also where scatter.png is "
             "saved, if plotting). Ignored when --bench is set (canonical tree "
             "always wins there)."
    )
    parser.add_argument(
        "--output-base",
        dest="output_base",
        default=None,
        help="Accepted for CLI compatibility with phlag/phlagster (which pass "
             "the same flag to both stages), but has no effect here: "
             "scores.tsv always lives in one canonical, --output-base/dist_type-"
             "independent location keyed only by w<W>_s<S>, shared across every "
             "phlag-side --output-base/--base/dist_type variant instead of being "
             "recomputed/duplicated per variant."
    )
    parser.add_argument(
        "--bench",
        dest="bench",
        action="store_true",
        default=False,
        help="Set by benchmark's run_all() for its own subprocess invocations -- "
             "not meant to be passed by hand. Accepted for CLI compatibility with "
             "phlag/phlagster; has no effect on scores.tsv's location when set "
             "(stays in the canonical shared tree, same as always: "
             "store/caster/w<W>_s<S>/...). When NOT set (standalone use, the "
             "default), scores go to <repo_root>/out/msa/<category>/<subcategory>/"
             "w<W>_s<S>[/variant]/<node_name>/<pattern>/scores.tsv instead of the "
             "shared canonical tree."
    )
    parser.add_argument(
        "--pair",
        dest="pair",
        action="store_true",
        default=False,
        help="Run ./caster/bin/caster-pair (auto-(re)compiling from "
             "caster/caster-pair.cpp if missing/stale) instead of dstar -- scores one "
             "fixed quartet branch's topology per genomic chunk, instead of D*/ABBA-"
             "BABA-AABB windows. The branch is the -m/--mapping population mapping "
             "file (auto-detected the same way as for dstar), which must assign every "
             "taxon to exactly one of 4 groups. See --chunk-scores."
    )
    parser.add_argument(
        "--chunk-scores",
        dest="chunk_scores",
        type=pathlib.Path,
        default=None,
        help="Output path for --pair's/--site's per-chunk quartet scores TSV (default: "
             "wherever the scores file would normally go)."
    )
    parser.add_argument(
        "-c",
        "--chunk",
        dest="chunk_size",
        type=int_or_abbrev,
        default=None,
        help="Window size (in sites) for --pair's/--site's per-window quartet scores, "
             "rolled up from caster-pair's/caster-site's own fine-grained -s-sized chunks "
             "the same way dstar's window/step aggregation works below (default: -w's "
             "window size). See -s for the step/stride between windows -- when -s < "
             "--chunk, windows overlap."
    )
    parser.add_argument(
        "--site",
        dest="site",
        action="store_true",
        default=False,
        help="Run ./caster/bin/caster-site (auto-(re)compiling from "
             "caster/caster-site.cpp if missing/stale) instead of dstar -- scores one "
             "fixed quartet branch's CASTER-site topology per genomic chunk, instead of "
             "D*/ABBA-BABA-AABB windows or a species tree. The branch is the -m/--mapping "
             "population mapping file (auto-detected the same way as for dstar), which "
             "must assign every taxon to exactly one of 4 groups. Mutually exclusive with "
             "--pair. See --chunk-scores."
    )
    parser.add_argument(
        "-z",
        "--zscale",
        dest="zscale",
        action="store_true",
        default=False,
        help="Rescale each output c*ABBA/c*BABA/c*AABB column to mean 0.5, std 0.5 across "
             "the whole file (z-score to mean 0/std 1, then 0.5 + 0.5*z; not clipped, so "
             "outlier windows can still land outside [0,1]). Mainly for --pair's/--site's "
             "raw quartet sums (~1e11-scale for --pair), which cause float32 catastrophic "
             "cancellation in phlag's Gaussian fit otherwise; works for dstar's D* output "
             "too, though its own raw sums are already a numerically-safe ~1e4-1e5 scale."
    )
    parser.add_argument(
        "-i",
        "--ilr",
        dest="ilr",
        action="store_true",
        default=False,
        help="Replace each output row's 3 c*ABBA/c*BABA/c*AABB counts with their 2 "
             "isometric-log-ratio (ILR) coordinates (c*ILR1/c*ILR2) instead -- a 3-part "
             "composition has only 2 independent degrees of freedom once closed to "
             "proportions, so this is a genuine dimensionality reduction, not a rescaling "
             "like -z/--zscale. Implies closure to proportions internally (like "
             "-n/--normalize) regardless of whether -n is also passed (harmless if so -- "
             "a no-op, not a double-transform). --pair/--site's raw scores are CASTER's "
             "signed scoreCnt() support statistic (not a count) and can be negative, "
             "unlike dstar's; any row with a non-positive part is shifted to positive "
             "first (preserving relative differences), then zeros are handled via "
             "skbio's multiplicative-replacement technique before taking logs. Drops "
             "q1/q2/q3 entirely for --pair (they were proportions of the original 3-part "
             "composition, meaningless once replaced by 2 ILR coordinates). Mutually "
             "exclusive with -z/--zscale (rescaling before ILR corrupts the composition; "
             "rescaling the resulting ILR coordinates themselves is not supported)."
    )
    return parser


def parse_arguments(argv=None):
    parser = build_parser()
    return parser.parse_args(argv)


def run_caster_pair(args, repo_root, data_dir, final_output_path, locus_pattern):
    plot_data_dir = final_output_path.parent
    binary_name = "caster-pair.exe" if sys.platform == "win32" else "caster-pair"
    binary_candidates = [
        data_dir / "bin" / binary_name,
        repo_root / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "bin" / binary_name,
    ]
    which_path = shutil.which(binary_name)
    if which_path:
        binary_candidates.append(pathlib.Path(which_path))

    binary_path = None
    for candidate in binary_candidates:
        if candidate.exists():
            binary_path = candidate
            break

    if not binary_path:
        binary_path = repo_root / "caster" / "bin" / binary_name

    caster_pair_cpp = repo_root / "caster" / "caster-pair.cpp"
    if caster_pair_cpp.exists():
        if not binary_path.exists() or os.path.getmtime(caster_pair_cpp) > os.path.getmtime(binary_path):
            target_bin = repo_root / "caster" / "bin" / binary_name
            os.makedirs(target_bin.parent, exist_ok=True)
            print(f"Compiling 'caster-pair' binary from {caster_pair_cpp}...")
            includes_dir = repo_root / "caster" / "includes"
            compile_cmd = ["g++", "-std=gnu++17", "-O2", "-I", str(includes_dir), str(caster_pair_cpp), "-o", str(target_bin)]
            try:
                subprocess.run(compile_cmd, check=True)
                binary_path = target_bin
                print(f"Successfully compiled 'caster-pair' binary at {binary_path}")
            except Exception as e:
                print(f"Warning: Could not auto-compile 'caster-pair': {e}")

    if sys.platform != "win32" and binary_path.exists() and not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, os.stat(binary_path).st_mode | 0o755)
        except Exception as e:
            print(f"Warning: Failed to set executable permission on '{binary_path}': {e}")

    if not binary_path.exists():
        sys.exit(f"Error: 'caster-pair' binary not found and could not be compiled (looked for source at {caster_pair_cpp}).")

    if not args.mapping.exists():
        sys.exit(f"Error: Mapping file not found at '{args.mapping}'")

    chunk_scores_path = args.chunk_scores if args.chunk_scores else final_output_path
    chunk_scores_path.parent.mkdir(parents=True, exist_ok=True)
    window_size = args.chunk_size if args.chunk_size is not None else args.window_size
    pair_step = min(args.step_size, window_size)

    print(f"Running caster-pair on '{args.fasta_file}' with branch mapping '{args.mapping}', chunk(window)={window_size}, step={pair_step}...")
    cmd = [
        str(binary_path),
        "--branch-mapping", str(args.mapping.resolve()),
        "--chunk-scores", str(chunk_scores_path.resolve()),
        "--chunk", str(pair_step),
        str(args.fasta_file.resolve()),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error running 'caster-pair' binary:\nCommand: {e.cmd}\nExit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")

    # caster-pair itself only ever partitions a locus into non-overlapping
    # --chunk-sized blocks -- it has no native step/stride concept. To get an
    # overlapping window+step model (window_size != pair_step), we ran it above
    # at the fine pair_step granularity, and now roll consecutive per-locus rows
    # into window_size-wide sliding windows here -- the same two-stage
    # fine-grained-rows -> rolling-window trick used for dstar above.
    K = max(1, window_size // pair_step)
    if K > 1:
        raw_df = pd.read_csv(chunk_scores_path, sep="\t")
        agg_rows = []
        for source_file, locus_df in raw_df.groupby("file", sort=False):
            locus_df = locus_df.sort_values("pos").reset_index(drop=True)
            n = len(locus_df)
            if n < K:
                continue
            abba_vals = locus_df["c*ABBA"].to_numpy()
            baba_vals = locus_df["c*BABA"].to_numpy()
            aabb_vals = locus_df["c*AABB"].to_numpy()
            pos_vals = locus_df["pos"].to_numpy()

            # O(1) sliding window (same incoming/outgoing increment trick as
            # dstar's rolling window above), not a fresh iloc[i:i+K].sum()
            # per step -- that re-summed all K raw chunks on every step
            # (O(N*K) total instead of O(N)).
            s0 = abba_vals[:K].sum()
            s1 = baba_vals[:K].sum()
            s2 = aabb_vals[:K].sum()
            for i in range(n - K + 1):
                if i > 0:
                    s0 += abba_vals[i + K - 1] - abba_vals[i - 1]
                    s1 += baba_vals[i + K - 1] - baba_vals[i - 1]
                    s2 += aabb_vals[i + K - 1] - aabb_vals[i - 1]
                tot = s0 + s1 + s2
                pos_val = int(pos_vals[i])
                if args.shift_caster:
                    pos_val += window_size // 2
                agg_rows.append({
                    "file": source_file,
                    "pos": pos_val,
                    "c*ABBA": s0,
                    "c*BABA": s1,
                    "c*AABB": s2,
                    "q1": s0 / tot if tot > 0 else 1.0 / 3,
                    "q2": s1 / tot if tot > 0 else 1.0 / 3,
                    "q3": s2 / tot if tot > 0 else 1.0 / 3,
                })
        pd.DataFrame(agg_rows).to_csv(chunk_scores_path, sep="\t", index=False)

    if args.ilr:
        apply_ilr_to_scores_file(chunk_scores_path, chunk_scores_path, has_q123=True)
    elif args.normalize:
        apply_normalize_to_scores_file(chunk_scores_path, chunk_scores_path, has_q123=True, eps=(DEFAULT_NORM_EPS if args.norm_eps else None))

    if args.zscale:
        apply_zscale_to_scores_file(chunk_scores_path, has_q123=True)

    print(f"Success: chunk scores written to: {chunk_scores_path}")

    write_ground_truth_stats(
        scores_file=str(chunk_scores_path.resolve()),
        output_dir=str(plot_data_dir.resolve()),
        locus_pattern=locus_pattern,
        topologies=args.topologies,
        dist_type=args.dist_type,
    )

    if args.plot and any(p in args.plot for p in ("scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums")):
        # chunk_scores.tsv's columns (pos, c*ABBA, c*BABA, c*AABB) are written
        # by caster-pair.cpp to mirror dstar's scores.tsv exactly, so the same
        # CasterPlotter -- same palette, same ground-truth shading -- renders
        # it directly instead of a separate plotting path.
        CasterPlotter(
            scores_file=str(chunk_scores_path.resolve()),
            distribution=args.dist_type,
            data_dir=str(plot_data_dir.resolve()),
            topologies=args.topologies,
            plot_scores=("scatter" in args.plot),
            plot_dist=("dist" in args.plot),
            plot_correlation=("correlation" in args.plot),
            plot_topology_pairs=("topology_pairs" in args.plot),
            plot_quartet_counts=("quartet_counts" in args.plot),
            plot_sums=("sums" in args.plot),
            locus_pattern=locus_pattern,
        )

    return chunk_scores_path


def run_caster_site(args, repo_root, data_dir, final_output_path, locus_pattern):
    plot_data_dir = final_output_path.parent
    binary_name = "caster-site.exe" if sys.platform == "win32" else "caster-site"
    binary_candidates = [
        data_dir / "bin" / binary_name,
        repo_root / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "bin" / binary_name,
    ]
    which_path = shutil.which(binary_name)
    if which_path:
        binary_candidates.append(pathlib.Path(which_path))

    binary_path = None
    for candidate in binary_candidates:
        if candidate.exists():
            binary_path = candidate
            break

    if not binary_path:
        binary_path = repo_root / "caster" / "bin" / binary_name

    caster_site_cpp = repo_root / "caster" / "caster-site.cpp"
    if caster_site_cpp.exists():
        if not binary_path.exists() or os.path.getmtime(caster_site_cpp) > os.path.getmtime(binary_path):
            target_bin = repo_root / "caster" / "bin" / binary_name
            os.makedirs(target_bin.parent, exist_ok=True)
            print(f"Compiling 'caster-site' binary from {caster_site_cpp}...")
            includes_dir = repo_root / "caster" / "includes"
            compile_cmd = ["g++", "-std=gnu++17", "-O2", "-I", str(includes_dir), str(caster_site_cpp), "-o", str(target_bin)]
            try:
                subprocess.run(compile_cmd, check=True)
                binary_path = target_bin
                print(f"Successfully compiled 'caster-site' binary at {binary_path}")
            except Exception as e:
                print(f"Warning: Could not auto-compile 'caster-site': {e}")

    if sys.platform != "win32" and binary_path.exists() and not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, os.stat(binary_path).st_mode | 0o755)
        except Exception as e:
            print(f"Warning: Failed to set executable permission on '{binary_path}': {e}")

    if not binary_path.exists():
        sys.exit(f"Error: 'caster-site' binary not found and could not be compiled (looked for source at {caster_site_cpp}).")

    if not args.mapping.exists():
        sys.exit(f"Error: Mapping file not found at '{args.mapping}'")

    chunk_scores_path = args.chunk_scores if args.chunk_scores else final_output_path
    chunk_scores_path.parent.mkdir(parents=True, exist_ok=True)
    # Diagnostic-only companion file (per-window, per-topology raw per-site
    # quartet counts) -- see caster-site.cpp's --quartet-counts. Never read by
    # phlag; written purely for CasterPlotter.plot_quartet_counts.
    quartet_counts_path = chunk_scores_path.parent / "quartet_counts.tsv"
    window_size = args.chunk_size if args.chunk_size is not None else args.window_size
    site_step = min(args.step_size, window_size)

    print(f"Running caster-site on '{args.fasta_file}' with branch mapping '{args.mapping}', chunk(step)={site_step}, window={window_size}...")
    cmd = [
        str(binary_path),
        "--branch-mapping", str(args.mapping.resolve()),
        "--chunk-scores", str(chunk_scores_path.resolve()),
        "--quartet-counts", str(quartet_counts_path.resolve()),
        "--chunk", str(site_step),
        "--window", str(window_size),
        str(args.fasta_file.resolve()),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error running 'caster-site' binary:\nCommand: {e.cmd}\nExit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")

    # caster-site now does the fine-chunk -> sliding-window rolling itself
    # (dividing each window's quartet sums by that window's own
    # informative-site count, since --chunk's site-filter makes that count
    # vary chunk to chunk -- a raw sum would bias windows with more
    # informative sites). Only the cosmetic shift-to-window-center remains
    # here, since it's a pure pos-column offset unrelated to aggregation.
    if args.shift_caster:
        raw_df = pd.read_csv(chunk_scores_path, sep="\t")
        raw_df["pos"] = raw_df["pos"] + window_size // 2
        raw_df.to_csv(chunk_scores_path, sep="\t", index=False)
        # Keep quartet_counts.tsv's pos column in sync so it stays joinable
        # against chunk_scores_path by pos after the shift above.
        if quartet_counts_path.exists():
            quartet_counts_df = pd.read_csv(quartet_counts_path, sep="\t")
            quartet_counts_df["pos"] = quartet_counts_df["pos"] + window_size // 2
            quartet_counts_df.to_csv(quartet_counts_path, sep="\t", index=False)

    if args.ilr:
        apply_ilr_to_scores_file(chunk_scores_path, chunk_scores_path, has_q123=False)
    elif args.normalize:
        apply_normalize_to_scores_file(chunk_scores_path, chunk_scores_path, has_q123=False, eps=(DEFAULT_NORM_EPS if args.norm_eps else None))

    if args.zscale:
        apply_zscale_to_scores_file(chunk_scores_path, has_q123=False)

    print(f"Success: chunk scores written to: {chunk_scores_path}")

    write_ground_truth_stats(
        scores_file=str(chunk_scores_path.resolve()),
        output_dir=str(plot_data_dir.resolve()),
        locus_pattern=locus_pattern,
        topologies=args.topologies,
        dist_type=args.dist_type,
    )

    if args.plot and any(p in args.plot for p in ("scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums")):
        # chunk_scores.tsv's columns (pos, c*ABBA, c*BABA, c*AABB) are written
        # by caster-site.cpp to mirror dstar's scores.tsv exactly, so the same
        # CasterPlotter -- same palette, same ground-truth shading -- renders
        # it directly instead of a separate plotting path.
        CasterPlotter(
            scores_file=str(chunk_scores_path.resolve()),
            distribution=args.dist_type,
            data_dir=str(plot_data_dir.resolve()),
            topologies=args.topologies,
            plot_scores=("scatter" in args.plot),
            plot_dist=("dist" in args.plot),
            plot_correlation=("correlation" in args.plot),
            plot_topology_pairs=("topology_pairs" in args.plot),
            plot_quartet_counts=("quartet_counts" in args.plot),
            plot_sums=("sums" in args.plot),
            locus_pattern=locus_pattern,
        )

    return chunk_scores_path


def _strip_ws_flags(argv):
    """
    Drops any -w/-s and their nargs='+' values from argv, mirroring
    argparse's own consumption rule (values never start with '-'), so the
    multi-value loop in main() can re-append a single -w/-s pairing per
    recursive invocation without duplicating flags.
    """
    result = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-w", "-s"):
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                i += 1
            continue
        result.append(tok)
        i += 1
    return result


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_arguments(argv)

    window_sizes = args.window_size
    step_sizes = args.step_size
    if len(window_sizes) > 1 or len(step_sizes) > 1:
        base_argv = _strip_ws_flags(raw_argv)
        results = []
        for w, s in itertools.product(window_sizes, step_sizes):
            print(f"[caster] -w/-s got multiple values -- running with -w {w} -s {s}...")
            results.append(main(base_argv + ["-w", str(w), "-s", str(s)]))
        return results
    args.window_size = window_sizes[0]
    args.step_size = step_sizes[0]

    if args.pair and args.site:
        sys.exit("Error: --pair and --site are mutually exclusive.")

    if args.ilr and args.zscale:
        sys.exit("Error: --ilr and --zscale are mutually exclusive.")

    # Inject defaults for CLI flags if not provided
    if args.step_size is None:
        args.step_size = args.window_size
    elif isinstance(args.step_size, float):
        args.step_size = max(1, round(args.step_size * args.window_size))

    ALL_PLOTS = ["scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums"]
    if args.plot is None:
        # --plot omitted entirely: bench default is no plots, ad-hoc default is everything.
        args.plot = [] if args.bench else ALL_PLOTS
    elif args.plot == []:
        # Bare "--plot" (no choices given): plot everything, in either mode.
        args.plot = ALL_PLOTS

    if not args.bench:
        flags_str = " ".join(f"{k}={v}" for k, v in vars(args).items())
        print(f"[caster] Effective flags: {flags_str}")

    from .utils import get_data_dir, get_repo_root, get_most_recent_file, clean_locus_name
    repo_root = get_repo_root()
    data_dir = get_data_dir()
    
    # Resolve FASTA file fallback if not found or recent flag requested
    if args.recent or args.fasta_file == pathlib.Path("-r") or args.fasta_file is None:
        recent_fasta = get_most_recent_file(
            default_subdirs=["store/msa/concat", "msa/concat", "concat", "store/msa", "msa"],
            default_exts=[".fa", ".fasta", ".fa.gz"],
            target_dir_name="concat"
        )
        if recent_fasta is None or not recent_fasta.exists():
            sys.exit("Error: No input file found in store/msa/concat or candidate MSA directories.")
        args.fasta_file = recent_fasta

    if not args.fasta_file.exists():
        from .utils import resolve_locus_spec
        spec_fasta = resolve_locus_spec(args.fasta_file)
        if spec_fasta is not None:
            print(f"[caster] Resolved locus spec '{args.fasta_file}' -> '{spec_fasta}'")
            args.fasta_file = spec_fasta

    # Regenerate mode: a scores.tsv (or chunk_scores.tsv) path was passed
    # instead of a FASTA. Recover the source FASTA from its 'file' column
    # (same convention phlag.py's read_caster_scores relies on) and the
    # window/step (or chunk/step) that produced it from a 'w<...>_s<...>'/
    # 'c<...>_s<...>'/'c<...>_s<...>_site' path segment, then recompute and
    # overwrite that exact path -- regardless of --bench, since the
    # destination is already given. Exception: if --plot is the only other
    # flag passed, skip the recompute and just redraw the plots from what's
    # already at that path (see regen_plot_only / is_plot_only_argv below).
    regen_plot_only = False
    regen_output_path = None
    if args.fasta_file.suffix == ".tsv" and args.fasta_file.exists():
        regen_output_path = args.fasta_file.resolve()
        regen_plot_only = is_plot_only_argv(raw_argv)
        source_fasta = recover_source_fasta(regen_output_path)
        if source_fasta is None:
            sys.exit(f"Error: Could not recover source FASTA path from 'file' column in '{regen_output_path}'.")
        if not source_fasta.is_absolute():
            for base in (pathlib.Path.cwd(), repo_root):
                candidate = base / source_fasta
                if candidate.exists():
                    source_fasta = candidate
                    break
        if not source_fasta.exists():
            sys.exit(f"Error: Source FASTA '{source_fasta}' (recovered from '{regen_output_path}') no longer exists.")

        # An explicit -w/-s on this invocation (including each leg of the
        # multi-value cartesian loop above, which always re-adds a single
        # -w/-s) means the caller wants *new* window/step values computed
        # from the recovered source FASTA, not a same-path self-heal -- so
        # the path-recovered size must not clobber it, and the output must
        # land at the fresh derived path below rather than back at
        # regen_output_path (else every leg overwrites the same file with
        # the last leg's data, see the multi -w/-s + existing-scores bug).
        explicit_ws = "-w" in raw_argv or "-s" in raw_argv

        ws = parse_ws_from_path(regen_output_path)
        if ws:
            mode, val, step, is_site, is_zscale, is_ilr, is_normalize, norm_eps = ws
            if not explicit_ws:
                args.step_size = step
            args.zscale = is_zscale
            args.ilr = is_ilr
            args.normalize = is_normalize
            args.norm_eps = norm_eps
            args.exp_minus = "exp-minus" in regen_output_path.parts
            if mode == "c" and is_site:
                args.site = True
                args.pair = False
                if not explicit_ws:
                    args.chunk_size = val
            elif mode == "c":
                args.pair = True
                args.site = False
                if not explicit_ws:
                    args.chunk_size = val
            elif not explicit_ws:
                args.pair = False
                args.site = False
                args.window_size = val

        if explicit_ws:
            print(f"Recomputing from source FASTA '{source_fasta}' (recovered from '{regen_output_path}') at -w {args.window_size} -s {args.step_size}...")
        else:
            print(f"Regenerating '{regen_output_path}' from source FASTA '{source_fasta}'...")
        args.fasta_file = source_fasta
        if not explicit_ws:
            args.output_file = regen_output_path
        args.bench = False

    # Ground-truth locus pattern (e.g. '37-62') for CasterPlotter's scatter.png shading.
    window_str = format_val(args.window_size)
    step_str = format_val(args.step_size)
    norm_suffix = "_n" if args.normalize else ""
    zscale_suffix = "_z" if args.zscale else ""
    clean_stem = clean_locus_name(args.fasta_file.stem)
    left_str = format_val(args.left)
    right_str = format_val(args.right if args.right is not None else 0)

    from .utils import parse_filename_to_dir_structure, get_simulation_categories, get_short_sim_name
    parsed = parse_filename_to_dir_structure(clean_stem)
    locus_pattern = parsed["pattern"] if parsed else clean_stem

    parts = args.fasta_file.parts
    is_sim = "simulations" in parts
    cats = None
    short_sim = None
    if is_sim:
        sim_dir = args.fasta_file.parent
        if sim_dir.name in ["concat"] or sim_dir.name.startswith("concat_"):
            sim_dir = sim_dir.parent
        cats = get_simulation_categories(args.fasta_file)
        short_sim = get_short_sim_name(sim_dir.name)

    def _derive_output_path(normalize_flag, ilr_flag):
        """
        Same derivation as below, parameterized on the normalize/ilr flags so
        the --normalize/--ilr short-circuits (see below) can also derive the
        sibling raw path (both flags False) that they read from, without
        duplicating this whole tree. --ilr implies closure, so its keying
        takes the place of --normalize's, never stacking with it, even if
        --normalize was also explicitly passed.
        """
        if args.bench:
            # --bench (set only by benchmark's own subprocess invocations) keeps
            # scores.tsv in the shared canonical tree, keyed only by window/step
            # -- caster's windowed dstar statistics don't depend on dist_type or
            # --output-base/--base at all (those only select a phlag-side
            # model/variant tree), so every dist_type/--base variant reads and
            # writes the same cached scores.tsv here instead of each getting its
            # own copy recomputed from scratch. Lives in its own store/caster/
            # tree, not nested under store/phlag/, since it isn't a phlag output.
            # --pair/--site key this the same way they key the standalone tree
            # (c<chunk>_s<step>, see below) -- otherwise a --pair/--site run
            # sharing a dstar run's -w/-s values would collide on the exact
            # same cached scores.tsv, silently mixing quartet-branch scores
            # with D* scores. --site and --normalize each nest their own
            # named subdirectory ('site'/'normalize') under the size segment
            # instead of a flat suffix -- keeps the caster/ tree's directory
            # names legible (c<chunk>_s<step>/site/normalize/... rather than
            # c<chunk>_s<step>_site_n) -- '--site' likewise keeps --site from
            # colliding with a --pair run sharing the same chunk/step, and
            # 'normalize' keeps normalized and raw scores from sharing a
            # cache entry. --zscale still appends a flat zscale_suffix ("_z")
            # to the size segment itself, unchanged. --norm-eps is a boolean
            # (see apply_normalize) that nests its own 'norm-eps' segment
            # under 'normalize' when set, leaving plain 'normalize' as the
            # original unguarded data (bit-for-bit reproducible, nothing
            # existing needs to move) and 'normalize/norm-eps' as the fixed
            # data -- exactly two cache entries, not an eps<value> family.
            # Any future new flag that changes what ends up in
            # scores.tsv/report.tsv should get the same treatment -- its own
            # named segment here (and mirrored in bench/benchmark.py's
            # get_expected_caster_sim_dir) -- rather than folding into an
            # existing directory's cache entry.
            if args.pair or args.site:
                chunk = args.chunk_size if args.chunk_size is not None else args.window_size
                caster_root = data_dir / "caster" / f"c{format_val(chunk)}_s{step_str}{zscale_suffix}"
                if args.site:
                    caster_root = caster_root / "site"
            else:
                caster_root = data_dir / "caster" / f"w{window_str}_s{step_str}{zscale_suffix}"
            if ilr_flag:
                caster_root = caster_root / "ilr"
            elif normalize_flag:
                caster_root = caster_root / "normalize"
                if args.norm_eps:
                    caster_root = caster_root / "norm-eps"
            if args.exp_minus:
                caster_root = caster_root / "exp-minus"
            if parsed:
                rel_dir = parsed["relative_dir_no_window"]
                return caster_root / rel_dir / "scores.tsv"
            elif is_sim:
                pattern_stem = clean_stem
                if cats:
                    return caster_root / cats[0] / cats[1] / short_sim / pattern_stem / "scores.tsv"
                else:
                    return caster_root / short_sim / pattern_stem / "scores.tsv"
            else:
                pattern_stem = clean_stem
                final_output_name = f"{clean_stem}_{left_str}_{right_str}_w{window_str}_s{step_str}.tsv"
                return caster_root / pattern_stem / final_output_name
        else:
            if parsed:
                node_name = get_short_sim_name(parsed["alt"])
                node_rel = pathlib.Path(node_name, parsed["pattern"])
            elif is_sim:
                node_rel = pathlib.Path(short_sim, clean_stem)
            else:
                node_rel = pathlib.Path(clean_stem)
            return adhoc_scores_path(repo_root, args, cats if is_sim else None, node_rel, normalize_flag, ilr_flag)

    final_output_path = _derive_output_path(args.normalize, args.ilr)

    # -o override (standalone use only -- ignored under --bench, where the
    # canonical tree always wins): redirects where scores.tsv itself gets
    # written (and, via its parent, where CasterPlotter's scatter.png lands).
    if args.output_file and not args.bench:
        final_output_path = args.output_file
    plot_data_dir = final_output_path.parent

    # Exact-cache check: if final_output_path itself already exists (this
    # precise -w/-s[/--pair/--site]+ilr/normalize combination was already
    # computed), skip regeneration entirely -- cheaper than even the
    # raw-sibling short-circuits below, since no transform needs to be
    # (re)applied at all. final_output_path already resolves to the
    # mode-appropriate cache (store/caster/ under --bench, out/ standalone),
    # so this check covers both without branching on args.bench itself.
    # regen_plot_only (regen mode -- a scores.tsv was passed positionally --
    # with --plot as the only other flag) also takes this path: final_output_path
    # IS the passed-in file here, so no ilr/normalize is needed to justify
    # skipping recompute -- just redraw the plots from what's already there.
    if (args.ilr or args.normalize or args.exp_minus or regen_plot_only) and final_output_path.exists():
        print(f"Found existing scores at '{final_output_path}' -- skipping regeneration.")
        copy_quartet_counts_if_missing(strip_exp_minus(_derive_output_path(False, False)).parent, plot_data_dir)
        write_ground_truth_stats(
            scores_file=str(final_output_path.resolve()),
            output_dir=str(plot_data_dir.resolve()),
            locus_pattern=locus_pattern,
            topologies=args.topologies,
            dist_type=args.dist_type,
        )
        if args.plot and any(p in args.plot for p in ("scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums")):
            CasterPlotter(
                scores_file=str(final_output_path.resolve()),
                distribution=args.dist_type,
                data_dir=str(plot_data_dir.resolve()),
                topologies=args.topologies,
                plot_scores=("scatter" in args.plot),
                plot_dist=("dist" in args.plot),
                    plot_correlation=("correlation" in args.plot),
                plot_topology_pairs=("topology_pairs" in args.plot),
                plot_quartet_counts=("quartet_counts" in args.plot),
                plot_sums=("sums" in args.plot),
                locus_pattern=locus_pattern,
            )
        return final_output_path

    if args.exp_minus:
        base_argv = [t for t in raw_argv if t != "--exp-minus"]
        if regen_output_path is not None:
            base_regen = strip_exp_minus(regen_output_path)
            if not base_regen.exists():
                sys.exit(f"Error: --exp-minus regen needs its untransformed sibling '{base_regen}'.")
            base_argv = [str(base_regen) if (t.endswith(".tsv") and pathlib.Path(t).resolve() == regen_output_path) else t for t in base_argv]
        base_path = main(base_argv)
        print(f"Applying exp(-x) to '{base_path}'...")
        apply_exp_minus_to_scores_file(base_path, final_output_path)
        print(f"Success: TSV output file generated at: {final_output_path}")
        copy_quartet_counts_if_missing(base_path.parent, plot_data_dir)
        write_ground_truth_stats(
            scores_file=str(final_output_path.resolve()),
            output_dir=str(plot_data_dir.resolve()),
            locus_pattern=locus_pattern,
            topologies=args.topologies,
            dist_type=args.dist_type,
        )
        if args.plot and any(p in args.plot for p in ("scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums")):
            CasterPlotter(
                scores_file=str(final_output_path.resolve()),
                distribution=args.dist_type,
                data_dir=str(plot_data_dir.resolve()),
                topologies=args.topologies,
                plot_scores=("scatter" in args.plot),
                plot_dist=("dist" in args.plot),
                plot_correlation=("correlation" in args.plot),
                plot_topology_pairs=("topology_pairs" in args.plot),
                plot_quartet_counts=("quartet_counts" in args.plot),
                plot_sums=("sums" in args.plot),
                locus_pattern=locus_pattern,
            )
        return final_output_path

    # --ilr short-circuit: if the raw sibling scores.tsv (same -w/-s or
    # --pair/--site chunk/step, same cache tree -- store/caster/ under
    # --bench, out/ standalone) already exists, just compute ILR on its rows
    # into final_output_path instead of re-running dstar/caster-pair/
    # caster-site from scratch. Checked before --normalize's short-circuit
    # below so --ilr wins if both are set.
    if args.ilr:
        raw_path = _derive_output_path(False, False)
        if raw_path != final_output_path and raw_path.exists():
            print(f"Found raw scores at '{raw_path}' -- computing ILR without recomputing caster...")
            apply_ilr_to_scores_file(raw_path, final_output_path, has_q123=args.pair)
            print(f"Success: TSV output file generated at: {final_output_path}")
            copy_quartet_counts_if_missing(raw_path.parent, plot_data_dir)
            write_ground_truth_stats(
                scores_file=str(final_output_path.resolve()),
                output_dir=str(plot_data_dir.resolve()),
                locus_pattern=locus_pattern,
                topologies=args.topologies,
                dist_type=args.dist_type,
            )
            if args.plot and any(p in args.plot for p in ("scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums")):
                CasterPlotter(
                    scores_file=str(final_output_path.resolve()),
                    distribution=args.dist_type,
                    data_dir=str(plot_data_dir.resolve()),
                    topologies=args.topologies,
                    plot_scores=("scatter" in args.plot),
                    plot_dist=("dist" in args.plot),
                            plot_correlation=("correlation" in args.plot),
                    plot_topology_pairs=("topology_pairs" in args.plot),
                    plot_quartet_counts=("quartet_counts" in args.plot),
                    plot_sums=("sums" in args.plot),
                    locus_pattern=locus_pattern,
                )
            return final_output_path

    # --normalize short-circuit: if the un-normalized sibling scores.tsv (same
    # -w/-s or --pair/--site chunk/step, same cache tree -- store/caster/
    # under --bench, out/ standalone) already exists, just normalize its rows
    # into final_output_path instead of re-running dstar/caster-pair/
    # caster-site from scratch.
    if args.normalize:
        unnormalized_path = _derive_output_path(False, False)
        if unnormalized_path != final_output_path and unnormalized_path.exists():
            print(f"Found un-normalized scores at '{unnormalized_path}' -- normalizing without recomputing caster...")
            apply_normalize_to_scores_file(unnormalized_path, final_output_path, has_q123=args.pair, eps=(DEFAULT_NORM_EPS if args.norm_eps else None))
            print(f"Success: TSV output file generated at: {final_output_path}")
            copy_quartet_counts_if_missing(unnormalized_path.parent, plot_data_dir)
            write_ground_truth_stats(
                scores_file=str(final_output_path.resolve()),
                output_dir=str(plot_data_dir.resolve()),
                locus_pattern=locus_pattern,
                topologies=args.topologies,
                dist_type=args.dist_type,
            )
            if args.plot and any(p in args.plot for p in ("scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums")):
                CasterPlotter(
                    scores_file=str(final_output_path.resolve()),
                    distribution=args.dist_type,
                    data_dir=str(plot_data_dir.resolve()),
                    topologies=args.topologies,
                    plot_scores=("scatter" in args.plot),
                    plot_dist=("dist" in args.plot),
                            plot_correlation=("correlation" in args.plot),
                    plot_topology_pairs=("topology_pairs" in args.plot),
                    plot_quartet_counts=("quartet_counts" in args.plot),
                    plot_sums=("sums" in args.plot),
                    locus_pattern=locus_pattern,
                )
            return final_output_path

    if not args.fasta_file.exists():
        sys.exit(f"Error: FASTA file not found at '{args.fasta_file}'")
    if args.left < 0:
        sys.exit(f"Error: Left index must be >= 0, got {args.left}")
    if args.right is None:
        try:
            args.right = get_fasta_length(args.fasta_file)
        except Exception as e:
            sys.exit(f"Error reading FASTA file to compute right endpoint: {e}")
    if args.right <= args.left:
        sys.exit(f"Error: Right index must be greater than left index, got left={args.left}, right={args.right}")

    # Locate dstar binary, auto-(re)compiling from source if missing or stale
    binary_name = "dstar.exe" if sys.platform == "win32" else "dstar"
    binary_candidates = [
        data_dir / "bin" / binary_name,
        repo_root / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "bin" / binary_name,
    ]
    which_path = shutil.which(binary_name)
    if which_path:
        binary_candidates.append(pathlib.Path(which_path))

    binary_path = None
    for candidate in binary_candidates:
        if candidate.exists():
            binary_path = candidate
            break

    if not binary_path:
        binary_path = repo_root / "caster" / "bin" / binary_name

    dstar_cpp = repo_root / "caster" / "dstar.cpp"
    if dstar_cpp.exists():
        if not binary_path.exists() or os.path.getmtime(dstar_cpp) > os.path.getmtime(binary_path):
            target_bin = repo_root / "caster" / "bin" / binary_name
            os.makedirs(target_bin.parent, exist_ok=True)
            print(f"Compiling 'dstar' binary from {dstar_cpp}...")
            compile_cmd = ["g++", "-std=gnu++17", "-O2", str(dstar_cpp), "-o", str(target_bin)]
            try:
                subprocess.run(compile_cmd, check=True)
                binary_path = target_bin
                print(f"Successfully compiled 'dstar' binary at {binary_path}")
            except Exception as e:
                print(f"Warning: Could not auto-compile 'dstar': {e}")

    if sys.platform != "win32" and binary_path.exists() and not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, os.stat(binary_path).st_mode | 0o755)
        except Exception as e:
            print(f"Warning: Failed to set executable permission on '{binary_path}': {e}")

    # Auto-detect a population mapping file if none was given explicitly.
    # Simulation mapping files live alongside 'concat/' as neoaves_{node}_mapping.tsv,
    # where {node} may carry a ':clade' disambiguator suffix (e.g. 'Strigiformes:3').
    if args.mapping is None:
        sim_dir = args.fasta_file.parent
        if sim_dir.name == "concat" or sim_dir.name.startswith("concat_"):
            sim_dir = sim_dir.parent
        mapping_candidates = sorted(sim_dir.glob("neoaves_*_mapping.tsv")) or sorted(sim_dir.glob("*_mapping.tsv"))

        if len(mapping_candidates) == 1:
            args.mapping = mapping_candidates[0]
            print(f"Auto-detected mapping file: {args.mapping}")
        elif len(mapping_candidates) > 1:
            sys.exit(f"Error: Multiple candidate mapping files found in '{sim_dir}': {[str(p) for p in mapping_candidates]}. Please specify one explicitly with --mapping.")
        else:
            sys.exit(f"Error: No mapping file found in '{sim_dir}'. Please generate one or specify an existing one explicitly with --mapping.")

    if args.pair:
        return run_caster_pair(args, repo_root, data_dir, final_output_path, locus_pattern)
    if args.site:
        return run_caster_site(args, repo_root, data_dir, final_output_path, locus_pattern)

    # Run dstar binary on original fasta file directly, then aggregate its
    # fine-grained (step_size-spaced) rows into rolling window_size averages
    print(f"Running D* calculation on original file '{args.fasta_file}' with window={args.window_size}, step={args.step_size}...")
    cmd = [str(binary_path), str(args.fasta_file.resolve())]
    if args.mapping:
        if not args.mapping.exists():
            sys.exit(f"Error: Mapping file not found at '{args.mapping}'")
        cmd.append(str(args.mapping.resolve()))
    else:
        cmd.append("-")
    cmd.append(str(args.step_size))

    temp_dir = tempfile.mkdtemp()
    # dstar's own positional WINDOW_SIZE (4th arg) only controls its internal
    # pi-estimation blocking within each step_size-sized interval row -- it's
    # unrelated to args.window_size (the Python-side rolling window below)
    # and was never passed before this diagnostic was added, so it's kept at
    # dstar's own prior default (10000) here to leave the main scores.tsv
    # table byte-for-byte unchanged. The 5th positional arg is new: a
    # diagnostic companion TSV of per-window, per-topology raw per-site
    # quartet counts, sign-classified (zero/negative/positive), written into
    # the same temp_dir as the main stdout table and rolled up below exactly
    # like c*ABBA/BABA/AABB.
    dstar_quartet_counts_bin_path = os.path.join(temp_dir, "quartet_counts.tsv")
    cmd.append("10000")
    cmd.append(dstar_quartet_counts_bin_path)
    try:
        try:
            result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Error running 'dstar' binary:\nCommand: {e.cmd}\nExit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")

        lines = result.stdout.splitlines()
        if not lines:
            sys.exit(f"Error: D* output was empty. Stderr: {result.stderr}")

        header_idx = 0
        for idx, line in enumerate(lines):
            if "pos" in line:
                header_idx = idx
                break

        raw_rows = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            parts = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(parts) >= 8:
                raw_rows.append(parts)

        if not raw_rows:
            mapping_name = args.mapping.name if args.mapping else "-"
            sys.exit(f"Error: No window scores calculated for '{args.fasta_file.name}' using mapping '{mapping_name}'. Please check that sequence headers in the FASTA match species in the mapping file.")

        # Diagnostic-only quartet-counts companion table, read the same way as
        # the main stdout table above; rows line up 1:1 with raw_rows since
        # both are written by the same per-interval loop in dstar.cpp.
        quartet_counts_raw_rows = []
        if os.path.exists(dstar_quartet_counts_bin_path):
            with open(dstar_quartet_counts_bin_path) as f:
                quartet_counts_lines = f.read().splitlines()
            for line in quartet_counts_lines[1:]:
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 11:
                    quartet_counts_raw_rows.append(parts)

        # Perform rolling window average with O(1) sliding window
        K = max(1, args.window_size // args.step_size)

        parsed_rows = [
            (
                row[0],             # file
                int(row[1]),        # pos
                float(row[2]),      # abba
                float(row[3]),      # baba
                float(row[4]),      # aabb
                float(row[6]),      # qcnt
                int(row[7])         # site count covered by this row
            )
            for row in raw_rows
        ]

        results = []
        if len(parsed_rows) >= K:
            run_abba = sum(r[2] for r in parsed_rows[:K])
            run_baba = sum(r[3] for r in parsed_rows[:K])
            run_aabb = sum(r[4] for r in parsed_rows[:K])
            run_qcnt = sum(r[5] for r in parsed_rows[:K])
            run_sitecnt = sum(r[6] for r in parsed_rows[:K])

            for i in range(len(parsed_rows) - K + 1):
                if i > 0:
                    outgoing = parsed_rows[i - 1]
                    incoming = parsed_rows[i + K - 1]
                    run_abba += incoming[2] - outgoing[2]
                    run_baba += incoming[3] - outgoing[3]
                    run_aabb += incoming[4] - outgoing[4]
                    run_qcnt += incoming[5] - outgoing[5]
                    run_sitecnt += incoming[6] - outgoing[6]

                pos_val = parsed_rows[i][1]  # Position of the start of the window
                if args.shift_caster:
                    pos_val += args.window_size // 2

                # Per-site average (like caster-site.cpp's siteSum division),
                # not per-chunk (K) -- a window with more sites summed into it
                # would otherwise report a proportionally larger raw mean for
                # no biological reason.
                site_denom = run_sitecnt if run_sitecnt > 0 else 1
                avg_abba = run_abba / site_denom
                avg_baba = run_baba / site_denom
                avg_aabb = run_aabb / site_denom
                avg_qcnt = run_qcnt / site_denom

                # Recalculate D* for the combined window (denom ratio is invariant to K)
                denom = run_abba + run_baba + run_aabb
                dstar_val = (run_abba - run_baba) / denom if denom != 0 else 0.0

                if args.left <= pos_val < args.right:
                    file_val = parsed_rows[i][0]
                    results.append({
                        'file': file_val,
                        'pos': pos_val,
                        'abba': avg_abba,
                        'baba': avg_baba,
                        'aabb': avg_aabb,
                        'dstar': dstar_val,
                        'qcnt': avg_qcnt,
                        'sitecnt': run_sitecnt
                    })

        # Diagnostic-only: same O(1) sliding-window logic as above, but
        # SUMMING (not averaging) each of the 9 quartet-count columns -- these
        # are literal per-site counts, not per-informative-site averages.
        # Iterated separately (rather than folded into the loop above) to
        # keep this purely-diagnostic addition from touching the already-
        # tested scores.tsv rolling logic. Skips gracefully (warns, doesn't
        # raise) if the quartet-counts file is missing or its row count doesn't
        # line up with the main table's.
        quartet_counts_results = []
        if not quartet_counts_raw_rows:
            print("Warning: quartet counts output missing or empty; skipping quartet_counts.tsv.")
        elif len(quartet_counts_raw_rows) != len(parsed_rows):
            print(f"Warning: quartet counts row count ({len(quartet_counts_raw_rows)}) does not match main table row count "
                  f"({len(parsed_rows)}); skipping quartet_counts.tsv.")
        elif len(parsed_rows) >= K:
            quartet_counts_parsed_rows = [tuple(int(x) for x in row[2:11]) for row in quartet_counts_raw_rows]
            run_quartet_counts = [sum(r[c] for r in quartet_counts_parsed_rows[:K]) for c in range(9)]
            for i in range(len(parsed_rows) - K + 1):
                if i > 0:
                    outgoing = quartet_counts_parsed_rows[i - 1]
                    incoming = quartet_counts_parsed_rows[i + K - 1]
                    for c in range(9):
                        run_quartet_counts[c] += incoming[c] - outgoing[c]

                pos_val = parsed_rows[i][1]
                if args.shift_caster:
                    pos_val += args.window_size // 2

                if args.left <= pos_val < args.right:
                    quartet_counts_results.append((parsed_rows[i][0], pos_val, tuple(run_quartet_counts)))

        if quartet_counts_results:
            quartet_counts_output_lines = ["file\tpos\tABBA_zero\tABBA_neg\tABBA_pos\tBABA_zero\tBABA_neg\tBABA_pos\tAABB_zero\tAABB_neg\tAABB_pos\n"]
            for file_val, pos_val, counts in quartet_counts_results:
                quartet_counts_output_lines.append(f"{file_val}\t{pos_val}\t" + "\t".join(str(c) for c in counts) + "\n")

            quartet_counts_out_path = final_output_path.parent / "quartet_counts.tsv"
            quartet_counts_out_path.parent.mkdir(parents=True, exist_ok=True)
            quartet_counts_tmp_fd, quartet_counts_tmp_path = tempfile.mkstemp(
                dir=str(quartet_counts_out_path.parent), prefix=".quartet_counts_", suffix=".tmp"
            )
            try:
                with os.fdopen(quartet_counts_tmp_fd, "w") as f:
                    f.writelines(quartet_counts_output_lines)
                os.replace(quartet_counts_tmp_path, quartet_counts_out_path)
            except BaseException:
                try:
                    os.remove(quartet_counts_tmp_path)
                except OSError:
                    pass
                raise
            print(f"Success: quartet counts TSV output file generated at: {quartet_counts_out_path}")

        if args.ilr:
            # D* itself is invariant to closure (same denominator cancels),
            # so it's untouched by the composition being replaced entirely
            # with its 2 ILR coordinates below.
            apply_ilr(results, ['abba', 'baba', 'aabb'], ['ilr1', 'ilr2'])
        elif args.normalize:
            # D* itself is invariant to this rescaling (same denominator
            # cancels) -- only the c*ABBA/c*BABA/c*AABB columns change, into
            # proportions of their own row sum (mirroring caster-pair.cpp's
            # q1/q2/q3; NOT the same as dividing by QuartetCnt, a per-site
            # sequence-depth product unrelated to this sum -- see dstar.cpp's
            # quartetCnt()). Applied at write time (after window
            # rolling-average, like --zscale below) so the --normalize
            # short-circuit above can reproduce it exactly from an
            # already-window-averaged un-normalized scores.tsv.
            apply_normalize(results, ['abba', 'baba', 'aabb'], eps=(DEFAULT_NORM_EPS if args.norm_eps else None))

        if args.zscale:
            # D* itself is left as originally computed from the raw sums --
            # only the c*ABBA/c*BABA/c*AABB columns phlag reads get rescaled.
            # (Never reached alongside --ilr -- the two are mutually exclusive.)
            apply_zscale(results, ['abba', 'baba', 'aabb'])

        if args.ilr:
            output_lines = ["file\tpos\tc*ILR1\tc*ILR2\tD*\tQuartetCnt\tSiteCnt\n"]
            for r in results:
                output_lines.append(f"{r['file']}\t{r['pos']}\t{r['ilr1']:.6g}\t{r['ilr2']:.6g}\t{r['dstar']:.6g}\t{r['qcnt']:.0f}\t{r['sitecnt']:.0f}\n")
        else:
            output_lines = ["file\tpos\tc*ABBA\tc*BABA\tc*AABB\tD*\tQuartetCnt\tSiteCnt\n"]
            for r in results:
                output_lines.append(f"{r['file']}\t{r['pos']}\t{r['abba']:.6g}\t{r['baba']:.6g}\t{r['aabb']:.6g}\t{r['dstar']:.6g}\t{r['qcnt']:.0f}\t{r['sitecnt']:.0f}\n")

        # final_output_path may be the shared, base-independent caster/
        # cache -- concurrent benchmark runs across different --base
        # variants can legitimately be computing the same scores.tsv at
        # once, so the write itself must be atomic (write-then-rename)
        # rather than an in-place open("w"): any reader checking
        # final_output_path.exists() must only ever see either nothing
        # or a complete file, never a partial one.
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(final_output_path.parent), prefix=".scores_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.writelines(output_lines)
            os.replace(tmp_path, final_output_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        print(f"Success: TSV output file generated at: {final_output_path}")
    finally:
        shutil.rmtree(temp_dir)

    print(f"Using scores file: {final_output_path}")

    write_ground_truth_stats(
        scores_file=str(final_output_path.resolve()),
        output_dir=str(plot_data_dir.resolve()),
        locus_pattern=locus_pattern,
        topologies=args.topologies,
        dist_type=args.dist_type,
    )

    if args.plot:
        plot_scores = "scatter" in args.plot
        plot_dist = "dist" in args.plot
        plot_correlation = "correlation" in args.plot
        plot_topology_pairs = "topology_pairs" in args.plot
        plot_quartet_counts = "quartet_counts" in args.plot
        plot_sums = "sums" in args.plot

        if plot_scores or plot_dist or plot_correlation or plot_topology_pairs or plot_quartet_counts or plot_sums:
            CasterPlotter(
                scores_file=str(final_output_path.resolve()),
                distribution=args.dist_type,
                data_dir=str(plot_data_dir.resolve()),
                topologies=args.topologies,
                plot_scores=plot_scores,
                plot_dist=plot_dist,
                plot_correlation=plot_correlation,
                plot_topology_pairs=plot_topology_pairs,
                plot_quartet_counts=plot_quartet_counts,
                plot_sums=plot_sums,
                locus_pattern=locus_pattern,
            )

    return final_output_path

if __name__ == "__main__":
    main()
