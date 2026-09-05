import sys
import pathlib
import os
import re
import argparse
import subprocess
import shutil
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, expon

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
    val_str = str(val_str).strip().lower()
    if val_str.endswith('k') or val_str.endswith('m'):
        return int_or_abbrev(val_str)
    frac = float(val_str)
    if 0 < frac < 1:
        return frac
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
    (--pair), or 'c<...>_s<...>[_site][_z][_i][_n]' (--site/--zscale/--ilr/
    --normalize) path segment, as written by caster.py's standalone out/ tree
    (flat suffixes) and older canonical store/caster/ runs (same flat
    suffixes). The current canonical store/caster/ tree instead nests
    'site'/'ilr'/'normalize' as their own path components right after the
    size segment (zscale stays a flat '_z' suffix there), and 'normalize'
    itself may further nest an 'eps<value>' component for a non-default
    --norm-eps (see get_expected_caster_sim_dir/_derive_output_path) -- so
    after matching the size segment, also consume any immediately-following
    'site'/'ilr'/'normalize'/'eps<value>' components, OR'd into whatever the
    flat suffixes already captured. norm_eps is None when no 'eps<value>'
    component is found (caster's own default applies). Returns None if no
    'w'/'c'-prefixed size segment is found anywhere in path's parts.
    """
    parts = path.parts
    for i, part in enumerate(parts):
        m = re.match(r'^([wc])(\d+[km]?)_s(\d+[km]?)(_site)?(_z)?(_i)?(_n)?$', part, re.IGNORECASE)
        if m:
            is_site = bool(m.group(4))
            is_zscale = bool(m.group(5))
            is_ilr = bool(m.group(6))
            is_normalize = bool(m.group(7))
            norm_eps = None
            for nested in parts[i + 1:]:
                if nested == "site":
                    is_site = True
                elif nested == "ilr":
                    is_ilr = True
                elif nested == "normalize":
                    is_normalize = True
                else:
                    eps_m = re.match(r'^eps([\d.eE+-]+)$', nested)
                    if eps_m:
                        norm_eps = float(eps_m.group(1))
                    break
            return (
                m.group(1).lower(), int_or_abbrev(m.group(2)), int_or_abbrev(m.group(3)),
                is_site, is_zscale, is_ilr, is_normalize, norm_eps,
            )
    return None


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


def apply_normalize(rows, keys, eps=DEFAULT_NORM_EPS):
    """
    Normalizes each row's `keys` (dict keys into `rows`, a list of dicts) to
    proportions of that row's own sum (1/3 each if the sum is 0), in place.
    Mirrors caster-pair.cpp's q1/q2/q3 -- NOT the same as dividing by
    QuartetCnt (dstar.cpp's quartetCnt(), a per-site sequence-depth product
    unrelated to the sum of the three D* topology scores). D* itself is
    invariant to this rescaling (same denominator cancels), so it is left
    untouched by callers.

    A window with no informative sites should have all three raw values at
    exactly 0, but float accumulation leaves residual noise around 1e-14
    instead -- an exact `denom == 0` check misses that, so dividing by a
    ~1e-14 (or smaller) denom blows a noise-level numerator up to spurious
    values in the thousands to billions. Treat the whole row as noise (fall
    back to 1/3 each) whenever every value in it is already below a
    noise floor, well under any real per-window topology count.

    That NOISE_FLOOR check only catches an all-near-zero row -- it misses a
    row whose 3 raw values are each individually real (not noise) but nearly
    cancel (caster-pair/caster-site's c*ABBA/c*BABA/c*AABB are CASTER's
    signed scoreCnt() evidence score, not a plain non-negative count, so
    this cancellation is a real, not-rare case, not just float noise), which
    still blows the ratio up to the thousands-to-billions range. `eps`
    guards this second case by clamping `denom`'s magnitude (sign
    preserved) to at least `eps` before dividing, independent of the
    NOISE_FLOOR all-zero fallback above.
    """
    NOISE_FLOOR = 1e-9
    for row in rows:
        vals = [row[k] for k in keys]
        denom = sum(vals)
        if max(abs(v) for v in vals) < NOISE_FLOOR:
            for k in keys:
                row[k] = 1.0 / len(keys)
        else:
            denom_safe = max(denom, eps) if denom >= 0 else min(denom, -eps)
            for k, v in zip(keys, vals):
                row[k] = v / denom_safe
    return rows


def apply_normalize_to_scores_file(src_path, dst_path, has_q123, eps=DEFAULT_NORM_EPS):
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
    def __init__(self, scores_file, distribution='gaussian', data_dir=None, topologies=None, plot_scores=True, plot_dist=False, plot_correlation=False, plot_topology_pairs=False, plot_quartet_counts=False, locus_pattern=None):
        self.scores_file = scores_file
        self.distribution = distribution
        self.data_dir = data_dir if data_dir is not None else str(pathlib.Path(scores_file).parent)
        self.topologies = topologies
        self.locus_pattern = locus_pattern

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
            transform_bits = [b for b, on in (("zscaled", is_zscale), ("ilr", is_ilr), ("normalized", is_normalize)) if on]
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

        # Draw vertical split lines and shade alt regions if a ground truth pattern is known
        self._shade_locus_pattern(plt.gca())

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
            xmin, xmax = vals.min(), vals.max()
            margin = (xmax - xmin) * 0.15 if xmax > xmin else 1.0
            x_grid = np.linspace(xmin - margin, xmax + margin, 200)

            null_vals = self.df.loc[labels == 'Null', col]
            alt_vals = self.df.loc[labels == 'Alt', col]
            is_exponential = self.distribution == "exponential"
            shift = vals.min() if is_exponential else None

            if len(null_vals) > 0:
                sns.histplot(null_vals, ax=ax, stat='density', element='step', kde=False, alpha=0.35, color=null_color, label='Null Histogram', bins=30)
            if len(alt_vals) > 0:
                sns.histplot(alt_vals, ax=ax, stat='density', element='step', kde=False, alpha=0.35, color='#E05638', label='Alt Histogram', bins=30)

            null_rate = alt_rate = None
            if len(null_vals) > 1:
                if is_exponential:
                    scale_null = (null_vals.to_numpy(dtype=float) - shift).mean()
                    null_rate = 1.0 / scale_null
                    mean_null = shift + scale_null
                    ax.plot(x_grid, expon.pdf(x_grid - shift, scale=scale_null), color=null_color, linewidth=2.2, label='Null Fit')
                    ax.axvline(mean_null, color=null_color, linestyle='--', linewidth=1.5)
                    ax.text(mean_null, 0.90, f"$\\lambda_{{null}}={null_rate:.3g}$", transform=trans, color=null_color, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                else:
                    mu_null, std_null = norm.fit(null_vals)
                    ax.plot(x_grid, norm.pdf(x_grid, mu_null, std_null), color=null_color, linewidth=2.2, label='Null Fit')
                    ax.axvline(mu_null, color=null_color, linestyle='--', linewidth=1.5)
                    ax.text(mu_null, 0.90, f"$\\mu_{{null}}={mu_null:.2f}$", transform=trans, color=null_color, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.text(mu_null + std_null, 0.82, f"$\\sigma_{{null}}={std_null:.2f}$", transform=trans, color=null_color, fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            if len(alt_vals) > 1:
                if is_exponential:
                    scale_alt = (alt_vals.to_numpy(dtype=float) - shift).mean()
                    alt_rate = 1.0 / scale_alt
                    mean_alt = shift + scale_alt
                    ax.plot(x_grid, expon.pdf(x_grid - shift, scale=scale_alt), color='#E05638', linewidth=2.2, linestyle='--', label='Alt Fit')
                    ax.axvline(mean_alt, color='#E05638', linestyle=':', linewidth=1.5)
                    ax.text(mean_alt, 0.75, f"$\\lambda_{{alt}}={alt_rate:.3g}$", transform=trans, color='#E05638', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                else:
                    mu_alt, std_alt = norm.fit(alt_vals)
                    ax.plot(x_grid, norm.pdf(x_grid, mu_alt, std_alt), color='#E05638', linewidth=2.2, linestyle='--', label='Alt Fit')
                    ax.axvline(mu_alt, color='#E05638', linestyle=':', linewidth=1.5)
                    ax.text(mu_alt, 0.75, f"$\\mu_{{alt}}={mu_alt:.2f}$", transform=trans, color='#E05638', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.text(mu_alt + std_alt, 0.67, f"$\\sigma_{{alt}}={std_alt:.2f}$", transform=trans, color='#E05638', fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            title = f'Topology: {topo_name}'
            if len(null_vals) > 1 and len(alt_vals) > 1:
                if is_exponential:
                    from .utils import exponential_hellinger2_nd
                    h2 = exponential_hellinger2_nd([null_rate], [alt_rate])
                else:
                    from .utils import gaussian_hellinger2_nd
                    h2 = gaussian_hellinger2_nd([mu_null], [[std_null ** 2]], [mu_alt], [[std_alt ** 2]])
                title += f'  ($H^2$={h2:.3f})'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{norm_label} Score')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.5)

        title = f'Topology Histograms & {self.distribution} Fits: {self.gene_name}'
        if self.data_tag:
            title += f' ({self.data_tag})'
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

    def _compute_null_alt_labels(self):
        """
        Shared ground-truth Null/Alt per-window labeling, factored out so
        plot_topology_pairs and plot_correlation's null/alt split can reuse
        the exact same parse_pattern_string logic
        instead of re-deriving it. Returns a numpy object array of
        'Null'/'Alt' (one per row of self.df, positional) or None if no
        ground-truth locus pattern is resolvable.
        """
        pattern_str = self._ground_truth_pattern()
        if not (pattern_str and 'pos' in self.df.columns and len(self.df) > 0):
            return None
        from .utils import parse_pattern_string
        positions = self.df['pos'].to_numpy()
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, lbl in zip(axes, ('Null', 'Alt')):
                mask = labels == lbl
                n = int(mask.sum())
                sub_corr = corr_df.loc[mask].corr(method='pearson') if n >= 2 else None
                if sub_corr is not None:
                    sns.heatmap(sub_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                                square=True, cbar_kws={'label': 'Pearson r'}, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'Not enough windows', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_title(f'{lbl} (n={n})', fontsize=11, fontweight='bold')
            fig.suptitle(title, fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_corr = os.path.join(output_dir, 'correlation.png')
        plt.savefig(save_path_corr, dpi=300, bbox_inches='tight')
        print(f"Saved topology correlation heatmap to: {save_path_corr}")
        plt.close()

    def plot_quartet_counts(self):
        """
        Diagnostic plot for dstar.cpp/caster-site.cpp's optional quartet_counts.tsv
        companion file (per-window, per-topology counts of raw per-site
        scores classified zero/negative/positive -- see caster/dstar.cpp's
        scoreIntervalWithCounts and sequence.hpp's
        Quadripartition::Gene::signCounts). Purely diagnostic: this file is
        never read by phlag's HMM and has no bearing on scores.tsv/
        chunk_scores.tsv. Skips gracefully (prints a warning, doesn't raise)
        when quartet_counts.tsv is missing, matching the convention the other
        optional plots in this class already follow.
        """
        quartet_counts_path = pathlib.Path(self.scores_file).parent / "quartet_counts.tsv"
        if not quartet_counts_path.exists():
            print(f"No quartet_counts.tsv found at '{quartet_counts_path}'; skipping quartet counts plot.")
            return

        try:
            quartet_counts_df = pd.read_csv(quartet_counts_path, sep='\t')
        except Exception as e:
            print(f"Error reading quartet counts file {quartet_counts_path}: {e}; skipping quartet counts plot.")
            return

        if 'pos' not in quartet_counts_df.columns:
            print(f"Quartet counts file '{quartet_counts_path}' has no 'pos' column; skipping quartet counts plot.")
            return

        topo_order = ['ABBA', 'BABA', 'AABB']
        kind_colors = {'zero': '#7F7F7F', 'negative': self.topo_colors['BABA'], 'positive': self.topo_colors['ABBA']}
        kind_cols = {'zero': '_zero', 'negative': '_neg', 'positive': '_pos'}

        missing = [f"{t}{suffix}" for t in topo_order for suffix in kind_cols.values()
                   if f"{t}{suffix}" not in quartet_counts_df.columns]
        if missing:
            print(f"Quartet counts file '{quartet_counts_path}' is missing expected column(s) {missing}; skipping quartet counts plot.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        for ax, topo in zip(axes, topo_order):
            for kind, suffix in kind_cols.items():
                ax.plot(quartet_counts_df['pos'], quartet_counts_df[f"{topo}{suffix}"], color=kind_colors[kind], label=kind, linewidth=1.2)
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


def write_ground_truth_stats(scores_file, output_dir, locus_pattern=None, topologies=None, dist_type="gaussian"):
    """
    Writes gt_stats.txt (Null/Alt/Overall mean+covariance across the 3
    topology dimensions, ABBA/BABA/AABB order) next to scores_file, so
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
    "exponential" instead shifts each of the 3 topology columns by its own
    global min (over the whole file, so Null and Alt stay shifted by the
    same reference and remain comparable) so all values are >= 0, then
    reads Exponential rates directly off the shifted Null/Alt means
    (mean of shifted data == 1/rate) and computes Hellinger2 via
    exponential_hellinger2_nd. Mean/covariance bookkeeping in gt_stats.txt
    itself is otherwise unaffected -- the shift only changes what the means
    represent (shifted-space means, whose reciprocal is the rate) and only
    when dist_type="exponential".
    """
    from .utils import parse_pattern_string, write_gt_stats_file, gaussian_hellinger2_nd, exponential_hellinger2_nd, GT_STATS_FILENAME

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

    if dist_type == "exponential":
        Y = Y - Y.min(axis=0)

    stats = {"Overall": (Y.mean(axis=0), np.cov(Y, rowvar=False).reshape(3, 3))}

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
            null_vals = Y[y_true == 0]
            alt_vals = Y[y_true == 1]
            if len(null_vals) > 1:
                stats["Null"] = (null_vals.mean(axis=0), np.cov(null_vals, rowvar=False).reshape(3, 3))
            if len(alt_vals) > 1:
                stats["Alt"] = (alt_vals.mean(axis=0), np.cov(alt_vals, rowvar=False).reshape(3, 3))
            if "Null" in stats and "Alt" in stats:
                if dist_type == "exponential":
                    stats["Hellinger2"] = exponential_hellinger2_nd(
                        1.0 / stats["Null"][0], 1.0 / stats["Alt"][0],
                    )
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
        default=50000,
        help="Window size (default: 50000 / 50k)"
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
        type=float,
        default=DEFAULT_NORM_EPS,
        help=f"With -n/--normalize, clamp the per-row c*ABBA+c*BABA+c*AABB sum's "
             f"magnitude to at least this before dividing, to stop a near-zero "
             f"(sign-cancelling) sum blowing the ratio up to spurious huge values "
             f"(default: {DEFAULT_NORM_EPS})"
    )
    parser.add_argument(
        "-s",
        dest="step_size",
        type=step_size_or_fraction,
        default=1000,
        help="Step size (default: 1000 / 1k). A value strictly between 0 and 1 "
             "is treated as a fraction of -w and multiplied out (e.g. -w 50000 "
             "-s 0.1 -> step=5000)."
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
        choices=["scores", "scatter", "dist", "correlation", "topology_pairs", "quartet_counts"],
        default=["scores"],
        help="List of plots to generate (choices: scores/scatter, aliases for "
             "the same topology scatter plot; dist, per-topology Null/Alt "
             "histograms with Gaussian fit overlays (requires a resolvable "
             "ground-truth locus pattern, else skipped); topology_pairs, "
             "per-window ABBA/BABA/AABB points projected onto each of the "
             "three 2D axis pairs (1x3 subplot grid, one PNG); correlation, "
             "a pairwise Pearson correlation heatmap of the same three "
             "columns (split into Null/Alt side-by-side heatmaps when a "
             "ground-truth pattern is resolvable); quartet_counts, per-topology "
             "raw per-site score quartet counts (zero/negative/positive) from "
             "the optional quartet_counts.tsv companion file, one PNG with 3 "
             "stacked subplots (requires quartet_counts.tsv, else skipped). "
             "Default: scores)",
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
        choices=["gaussian", "gmm", "exponential"],
        help="Distribution type used for CasterPlotter's statistical fits (default: "
             "gaussian). No longer affects scores.tsv's output location -- that's "
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
             "default), scores go to <repo_root>/out/w<W>_s<S>/<node_name>/"
             "<node_name>.tsv instead of the shared canonical tree."
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
            for i in range(len(locus_df) - K + 1):
                window = locus_df.iloc[i:i + K]
                s0 = window["c*ABBA"].sum()
                s1 = window["c*BABA"].sum()
                s2 = window["c*AABB"].sum()
                tot = s0 + s1 + s2
                pos_val = int(locus_df.iloc[i]["pos"])
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
        apply_normalize_to_scores_file(chunk_scores_path, chunk_scores_path, has_q123=True, eps=args.norm_eps)

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

    if args.plot and any(p in args.plot for p in ("scores", "scatter", "dist", "correlation", "topology_pairs", "quartet_counts")):
        # chunk_scores.tsv's columns (pos, c*ABBA, c*BABA, c*AABB) are written
        # by caster-pair.cpp to mirror dstar's scores.tsv exactly, so the same
        # CasterPlotter -- same palette, same ground-truth shading -- renders
        # it directly instead of a separate plotting path.
        CasterPlotter(
            scores_file=str(chunk_scores_path.resolve()),
            distribution=args.dist_type,
            data_dir=str(plot_data_dir.resolve()),
            topologies=args.topologies,
            plot_scores=("scores" in args.plot or "scatter" in args.plot),
            plot_dist=("dist" in args.plot),
            plot_correlation=("correlation" in args.plot),
            plot_topology_pairs=("topology_pairs" in args.plot),
            plot_quartet_counts=("quartet_counts" in args.plot),
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
        apply_normalize_to_scores_file(chunk_scores_path, chunk_scores_path, has_q123=False, eps=args.norm_eps)

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

    if args.plot and any(p in args.plot for p in ("scores", "scatter", "dist", "correlation", "topology_pairs", "quartet_counts")):
        # chunk_scores.tsv's columns (pos, c*ABBA, c*BABA, c*AABB) are written
        # by caster-site.cpp to mirror dstar's scores.tsv exactly, so the same
        # CasterPlotter -- same palette, same ground-truth shading -- renders
        # it directly instead of a separate plotting path.
        CasterPlotter(
            scores_file=str(chunk_scores_path.resolve()),
            distribution=args.dist_type,
            data_dir=str(plot_data_dir.resolve()),
            topologies=args.topologies,
            plot_scores=("scores" in args.plot or "scatter" in args.plot),
            plot_dist=("dist" in args.plot),
            plot_correlation=("correlation" in args.plot),
            plot_topology_pairs=("topology_pairs" in args.plot),
            plot_quartet_counts=("quartet_counts" in args.plot),
            locus_pattern=locus_pattern,
        )

    return chunk_scores_path


def main(argv=None):
    args = parse_arguments(argv)

    if args.pair and args.site:
        sys.exit("Error: --pair and --site are mutually exclusive.")

    if args.ilr and args.zscale:
        sys.exit("Error: --ilr and --zscale are mutually exclusive.")

    # Inject defaults for CLI flags if not provided
    if args.step_size is None:
        args.step_size = args.window_size
    elif isinstance(args.step_size, float):
        args.step_size = max(1, round(args.step_size * args.window_size))

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

    # Regenerate mode: a scores.tsv (or chunk_scores.tsv) path was passed
    # instead of a FASTA. Recover the source FASTA from its 'file' column
    # (same convention phlag.py's read_caster_scores relies on) and the
    # window/step (or chunk/step) that produced it from a 'w<...>_s<...>'/
    # 'c<...>_s<...>'/'c<...>_s<...>_site' path segment, then recompute and
    # overwrite that exact path -- regardless of --bench, since the
    # destination is already given.
    if args.fasta_file.suffix == ".tsv" and args.fasta_file.exists():
        regen_output_path = args.fasta_file.resolve()
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

        ws = parse_ws_from_path(regen_output_path)
        if ws:
            mode, val, step, is_site, is_zscale, is_ilr, is_normalize, norm_eps = ws
            args.step_size = step
            args.zscale = is_zscale
            args.ilr = is_ilr
            args.normalize = is_normalize
            if norm_eps is not None:
                args.norm_eps = norm_eps
            if mode == "c" and is_site:
                args.site = True
                args.pair = False
                args.chunk_size = val
            elif mode == "c":
                args.pair = True
                args.site = False
                args.chunk_size = val
            else:
                args.pair = False
                args.site = False
                args.window_size = val

        print(f"Regenerating '{regen_output_path}' from source FASTA '{source_fasta}'...")
        args.fasta_file = source_fasta
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
        local_norm_suffix = "_n" if (normalize_flag and not ilr_flag) else ""
        if normalize_flag and not ilr_flag and args.norm_eps != DEFAULT_NORM_EPS:
            local_norm_suffix += f"_eps{args.norm_eps:g}"
        local_ilr_suffix = "_i" if ilr_flag else ""
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
            # to the size segment itself, unchanged. --norm-eps nests its own
            # 'eps<value>' segment under 'normalize' (only when non-default --
            # every prior normalize run used the default, so nothing existing
            # needs to move), same reasoning: a non-default eps changes the
            # cached scores.tsv's actual values, so it must not share a cache
            # entry with the default-eps run. Any future new flag that
            # changes what ends up in scores.tsv/report.tsv should get the
            # same treatment -- its own named segment here (and mirrored in
            # bench/benchmark.py's get_expected_caster_sim_dir) -- rather than
            # folding into an existing directory's cache entry.
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
                if args.norm_eps != DEFAULT_NORM_EPS:
                    caster_root = caster_root / f"eps{args.norm_eps:g}"
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
            # Standalone use (the default): no shared/canonical tree, no
            # dist_type/category nesting -- everything for a node lands under
            # <repo_root>/out/w<W>_s<S>[_z][_n]/<node_name>/<node_name>.tsv,
            # alongside phlag's report.tsv for the same node (see phlag.py).
            # Node name is the short simulation name for sim inputs, the alt
            # name for parsed null/alt filenames, or the cleaned stem otherwise.
            # Experiment (parsed null/alt) files additionally nest a <pattern>
            # subdir under node_name -- <node_name>/<pattern>/<node_name>.tsv --
            # mirroring the canonical tree's relative_dir_no_window, since
            # multiple loci/patterns can share one node_name; sim/plain-file
            # inputs stay flat (no pattern-equivalent worth nesting on).
            if parsed:
                node_name = get_short_sim_name(parsed["alt"])
            elif is_sim:
                node_name = short_sim
            else:
                node_name = clean_stem
            node_rel = pathlib.Path(node_name, parsed["pattern"]) if parsed else pathlib.Path(node_name)
            if args.pair or args.site:
                # --pair's/--site's chunk-rollup granularity is an independent
                # axis from dstar's window/step (it may not even be run for the
                # same node), so their output gets its own
                # out/c<chunk>_s<step>[_site][_z][_n]/<node_name>/ prefix instead
                # of colliding with dstar's out/w<W>_s<S>/<node_name>/ -- the
                # '_site' suffix keeps --site from colliding with a --pair run
                # sharing the same chunk/step, and '_z'/'_n' (--zscale/--normalize)
                # keep those from colliding with raw/un-zscaled output the same
                # way the canonical tree distinguishes them.
                chunk = args.chunk_size if args.chunk_size is not None else args.window_size
                chunk_str = format_val(chunk)
                step_str_local = format_val(args.step_size)
                site_suffix = "_site" if args.site else ""
                size_dir = f"c{chunk_str}_s{step_str_local}{site_suffix}{zscale_suffix}{local_ilr_suffix}{local_norm_suffix}"
            else:
                size_dir = f"w{window_str}_s{step_str}{zscale_suffix}{local_ilr_suffix}{local_norm_suffix}"
            return repo_root / "out" / size_dir / node_rel / f"{node_name}.tsv"

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
    if (args.ilr or args.normalize) and final_output_path.exists():
        print(f"Found existing scores at '{final_output_path}' -- skipping regeneration.")
        copy_quartet_counts_if_missing(_derive_output_path(False, False).parent, plot_data_dir)
        write_ground_truth_stats(
            scores_file=str(final_output_path.resolve()),
            output_dir=str(plot_data_dir.resolve()),
            locus_pattern=locus_pattern,
            topologies=args.topologies,
            dist_type=args.dist_type,
        )
        if args.plot and any(p in args.plot for p in ("scores", "scatter", "dist", "correlation", "topology_pairs", "quartet_counts")):
            CasterPlotter(
                scores_file=str(final_output_path.resolve()),
                distribution=args.dist_type,
                data_dir=str(plot_data_dir.resolve()),
                topologies=args.topologies,
                plot_scores=("scores" in args.plot or "scatter" in args.plot),
                plot_dist=("dist" in args.plot),
                    plot_correlation=("correlation" in args.plot),
                plot_topology_pairs=("topology_pairs" in args.plot),
                plot_quartet_counts=("quartet_counts" in args.plot),
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
            if args.plot and any(p in args.plot for p in ("scores", "scatter", "dist", "correlation", "topology_pairs", "quartet_counts")):
                CasterPlotter(
                    scores_file=str(final_output_path.resolve()),
                    distribution=args.dist_type,
                    data_dir=str(plot_data_dir.resolve()),
                    topologies=args.topologies,
                    plot_scores=("scores" in args.plot or "scatter" in args.plot),
                    plot_dist=("dist" in args.plot),
                            plot_correlation=("correlation" in args.plot),
                    plot_topology_pairs=("topology_pairs" in args.plot),
                    plot_quartet_counts=("quartet_counts" in args.plot),
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
            apply_normalize_to_scores_file(unnormalized_path, final_output_path, has_q123=args.pair, eps=args.norm_eps)
            print(f"Success: TSV output file generated at: {final_output_path}")
            copy_quartet_counts_if_missing(unnormalized_path.parent, plot_data_dir)
            write_ground_truth_stats(
                scores_file=str(final_output_path.resolve()),
                output_dir=str(plot_data_dir.resolve()),
                locus_pattern=locus_pattern,
                topologies=args.topologies,
                dist_type=args.dist_type,
            )
            if args.plot and any(p in args.plot for p in ("scores", "scatter", "dist", "correlation", "topology_pairs", "quartet_counts")):
                CasterPlotter(
                    scores_file=str(final_output_path.resolve()),
                    distribution=args.dist_type,
                    data_dir=str(plot_data_dir.resolve()),
                    topologies=args.topologies,
                    plot_scores=("scores" in args.plot or "scatter" in args.plot),
                    plot_dist=("dist" in args.plot),
                            plot_correlation=("correlation" in args.plot),
                    plot_topology_pairs=("topology_pairs" in args.plot),
                    plot_quartet_counts=("quartet_counts" in args.plot),
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
            if len(parts) >= 7:
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
                float(row[6])       # qcnt
            )
            for row in raw_rows
        ]

        results = []
        if len(parsed_rows) >= K:
            run_abba = sum(r[2] for r in parsed_rows[:K])
            run_baba = sum(r[3] for r in parsed_rows[:K])
            run_aabb = sum(r[4] for r in parsed_rows[:K])
            run_qcnt = sum(r[5] for r in parsed_rows[:K])

            for i in range(len(parsed_rows) - K + 1):
                if i > 0:
                    outgoing = parsed_rows[i - 1]
                    incoming = parsed_rows[i + K - 1]
                    run_abba += incoming[2] - outgoing[2]
                    run_baba += incoming[3] - outgoing[3]
                    run_aabb += incoming[4] - outgoing[4]
                    run_qcnt += incoming[5] - outgoing[5]

                pos_val = parsed_rows[i][1]  # Position of the start of the window
                if args.shift_caster:
                    pos_val += args.window_size // 2

                avg_abba = run_abba / K
                avg_baba = run_baba / K
                avg_aabb = run_aabb / K
                avg_qcnt = run_qcnt / K

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
                        'qcnt': avg_qcnt
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
            apply_normalize(results, ['abba', 'baba', 'aabb'], eps=args.norm_eps)

        if args.zscale:
            # D* itself is left as originally computed from the raw sums --
            # only the c*ABBA/c*BABA/c*AABB columns phlag reads get rescaled.
            # (Never reached alongside --ilr -- the two are mutually exclusive.)
            apply_zscale(results, ['abba', 'baba', 'aabb'])

        if args.ilr:
            output_lines = ["file\tpos\tc*ILR1\tc*ILR2\tD*\tQuartetCnt\n"]
            for r in results:
                output_lines.append(f"{r['file']}\t{r['pos']}\t{r['ilr1']:.6g}\t{r['ilr2']:.6g}\t{r['dstar']:.6g}\t{r['qcnt']:.0f}\n")
        else:
            output_lines = ["file\tpos\tc*ABBA\tc*BABA\tc*AABB\tD*\tQuartetCnt\n"]
            for r in results:
                output_lines.append(f"{r['file']}\t{r['pos']}\t{r['abba']:.6g}\t{r['baba']:.6g}\t{r['aabb']:.6g}\t{r['dstar']:.6g}\t{r['qcnt']:.0f}\n")

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
        plot_scores = "scores" in args.plot or "scatter" in args.plot
        plot_dist = "dist" in args.plot
        plot_correlation = "correlation" in args.plot
        plot_topology_pairs = "topology_pairs" in args.plot
        plot_quartet_counts = "quartet_counts" in args.plot

        if plot_scores or plot_dist or plot_correlation or plot_topology_pairs or plot_quartet_counts:
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
                locus_pattern=locus_pattern,
            )

    return final_output_path

if __name__ == "__main__":
    main()
