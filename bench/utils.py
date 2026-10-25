import argparse
import colorsys
import contextlib
import concurrent.futures
import functools
import hashlib
import io
import itertools
import math
import multiprocessing
import os
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats

from bench.cli.benchmark import (
    ADMIXTURE_BINS,
    BRANCH_LENGTH_BIN_EDGES,
    BRANCH_LENGTH_BINS,
    CASTER_ARG_SPECS,
    CATEGORY_TO_COLUMN,
    COL_ADMIXTURE,
    DEFAULT_CONCAT_PATTERNS,
    FIGURE_COLUMNS,
    FRACTION_BIN_EDGES,
    FRACTION_BINS,
    HD_NUM_BINS,
    PHLAG_ARG_SPECS,
    TOPOLOGY_NAMES,
    BenchmarkStats,
    _RE_STATE_PATH,
    _build_parser,
    _read_args_json,
    assign_bin,
    discover_leaf_dirs,
    get_expected_scores_path,
    nominal_anomaly_fraction,
    parse_report,
    transition_total,
)
from phlag.caster import format_val, int_or_abbrev, parse_ws_from_path
from phlag.utils import (
    ADMIXTURE_DIVERGENCE_THRESHOLD_MYR,
    ECDF_MAX_POINTS,
    PHLAG_SEGMENT_DEFAULTS,
    ecdf_points,
    get_admixture_divergence_time,
    get_cu_branch_length_from_population_info,
    get_out_root,
    get_repo_root,
    get_simulation_clade,
    parse_pattern_string,
    parse_phlag_param_segment,
    read_gt_stats_file,
)

STORE_ROOT = "store/phlag"


def _phlag_orig_root():
    return get_repo_root() / "store" / "phlag_orig"


def _latest_phlag_orig_dir():
    root = _phlag_orig_root()
    dirs = [d for d in root.glob("*/") if next(d.glob("*/*/*/*/report.tsv"), None)]
    return max(dirs, key=lambda d: d.stat().st_mtime, default=root)


PHLAG_SOURCE_ROOTS = {"gtrees": str(_latest_phlag_orig_dir()), "msa": str(get_out_root())}
PAPER_SOURCE = "paper"
PHLAG_SOURCE_ROOTS[PAPER_SOURCE] = PHLAG_SOURCE_ROOTS["gtrees"]
WINDOWLESS_PHLAG_SOURCES = {"gtrees", PAPER_SOURCE}
BRANCH_LENGTH_SCALE = 1.0


def _register_phlag_sources(sources):
    sources = [sources] if isinstance(sources, str) else list(sources)
    for source in sources:
        if source in PHLAG_SOURCE_ROOTS:
            continue
        d = _phlag_orig_root() / source
        if not d.is_dir():
            raise FileNotFoundError(f"phlag source {source!r} is not one of {sorted(PHLAG_SOURCE_ROOTS)} or a dir under {_phlag_orig_root()}")
        PHLAG_SOURCE_ROOTS[source] = str(d)
        WINDOWLESS_PHLAG_SOURCES.add(source)
    return sources


def _phlag_orig_rel(d):
    try:
        return Path(d).relative_to(_phlag_orig_root()).as_posix()
    except ValueError:
        return None
_GT_STATS_COLUMN = re.compile(r"^(null|alt|pooled)_(mean|cov|var|within)_")

SWEEP_METRICS = ["em_hd", "roc_auc", "f1"]

_BETA_FLAGS = {"--beta", "--beta-prime"}

_METRIC_PANEL_COLUMN = {
    "f1": "in_panel_a",
    "em_hd": "in_panel_em_divergence",
    "em_gt_hd": "in_panel_em_divergence",
    "null_kl_agg": "in_panel_relerr",
    "alt_kl_agg": "in_panel_relerr",
}

def _as_list(x):
    return [x] if isinstance(x, str) else x


_NO_METRICS_EXCLUSION = "no TPR/FPR metrics line in report"
_FAILED_RUN_ZERO_METRICS = ["f1", "tpr", "fpr", "precision"]

_SIGNED_METRIC_SUFFIX = "_signed"


def _signed_base_metric(metric):
    """The real runs.tsv metric name for a "_signed" pseudo-metric (e.g.
    "f1_signed" -> "f1"), or None if `metric` doesn't end in that suffix --
    see _plot_heatmap's signed-diff-against-baseline-column handling."""
    if isinstance(metric, str) and metric.endswith(_SIGNED_METRIC_SUFFIX):
        return metric[: -len(_SIGNED_METRIC_SUFFIX)]
    return None


def _null_col_val(col_vals):
    """Which of a heatmap's subplot-column values (_plot_heatmap's col_vals,
    i.e. agg[2]'s display values) counts as the "null"/baseline column for a
    "_signed" metric: the first one that's a flag's off/negated state
    or missing state ("(base)") -- see _combo_label_part --
    since that's the closest thing to a real baseline/control. Falls back to
    col_vals[0] (the first/leftmost column, in whatever order agg produced
    it) when no such state is present, e.g. a tuple axis's realized combos
    (_discover_flag_tuple_values sorts a fully-missing/off combo first when
    one exists on disk, so this still lands on it there too) or a plain
    value list like -w's, which has no notion of "off" at all."""
    for cv in col_vals:
        if cv in ("(none)", "(base)"):
            return cv
    return col_vals[0]

# Metrics that are proportions or bounded distances (never outside [0, 1]) --
# their bar/violin y-axis is pinned to [0, 1] with 0.2-step ticks (see
# _apply_bounded_yaxis) instead of autoscaling/whisker-clipping to whatever a
# given cell's data happens to span, so figures stay comparable to each other.
_BOUNDED_METRICS = {"f1", "tpr", "fpr", "em_hd", "em_gt_hd", "roc_auc", "transition_null_to_alt", "transition_alt_to_null"} | {
    f"{prefix}_transition_{move}"
    for prefix in ("fitted", "gt", "viterbi")
    for move in ("null_to_null", "null_to_alt", "alt_to_null", "alt_to_alt")
}
_BOUNDED_YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# runs.tsv columns resolve_configs_cartesian's agg/axes can categorize by
# alongside args.json flags -- these are per-ROW values (every leaf dir's
# runs.tsv has many rows spanning all of these), not per-directory ones, so
# they're resolved as a row filter applied when a config's dirs get loaded
# (see _load_configs_runs) rather than a find_matching_leaf_dirs constraint.
# The value is the default enumeration used when the axis appears in `agg`
# without its own explicit list in `axes`.
_DATA_COLUMN_AXES = {
    "x_bin": BRANCH_LENGTH_BINS + ADMIXTURE_BINS,
    "fraction_bin": FRACTION_BINS,
    "column": FIGURE_COLUMNS,
}

# Same idea as _DATA_COLUMN_AXES, but these have no fixed enumeration --
# their values are whatever's actually on disk, so the default list (when not
# given explicitly in `axes`) is discovered from runs.tsv via
# _discover_row_values instead of a static constant. em_hd_bin/em_gt_hd_bin
# are a further special case within this: their enumeration comes from
# CrossRunAnalysis._bin_specs (compute_hd_bin_edges), not a fresh runs.tsv
# scan -- see _discover_row_values.
_DYNAMIC_ROW_AXES = {
    "category", "subcategory", "merged_category", "em_hd_bin", "em_gt_hd_bin", "pattern",
    "anomaly_fraction", "outlier_pct",
}


NOMINAL_GENOME_SITES = 6_000_000


def _outlier_pct(patterns):
    pct = {p: nominal_anomaly_fraction(p) for p in patterns.dropna().unique()}
    return patterns.map(lambda p: round(100 * pct[p], 6) if pct.get(p) is not None else None)


def _outlier_sites_label(token):
    pct = _token_to_number(token)
    if pct is None:
        return str(token)
    return f"{_format_param_value(round(pct / 100 * NOMINAL_GENOME_SITES))} ({token})"


def _row_axis_display(name, value):
    if name == "outlier_pct":
        return f"{value:g}%"
    return f"{value:g}" if isinstance(value, float) else str(value)


_ROW_AXIS_TITLE_PREFIX = {
    "fraction_bin": "outlier fraction", "anomaly_fraction": "outlier fraction", "outlier_pct": "outlier",
    "em_hd_bin": "EM HD", "em_gt_hd_bin": "EM GT HD",
}


def _row_axis_title(name, value):
    if value in (None, "(none)"):
        return str(value)
    if name == "x_bin":
        return f"{'admixture' if value in ADMIXTURE_BINS else 'BL'} {value}"
    prefix = _ROW_AXIS_TITLE_PREFIX.get(name)
    return f"{prefix} {value}" if prefix else str(value)


def _merged_category(category, subcategory):
    """category+subcategory space-joined, except admixture -- its
    subcategory is a low/high divergence-time split (see
    get_simulation_categories in phlag/utils.py) that's already captured by
    x_bin's own admixture divergence bins, so merged_category collapses both
    admixture rows down to the bare "admixture" category instead of further
    splitting into "admixture low"/"admixture high"."""
    merged = category + " " + subcategory
    return merged.where(category != "admixture", category)


_CATEGORY_LIKE_AXES = ("category", "merged_category")

_MERGED_CATEGORY_ORDER = ["recombination up", "recombination down", "10X up", "10X down", "admixture"]

PHLAG_PAPER_F1 = {
    "(0,0.1]": {
        "recombination up": {"(0,0.1]": (0.26, 0.015), "(0.1,1]": (0.41, 0.02), "(1,5]": (0.70, 0.01)},
        "recombination down": {"(0,0.1]": (0.08, 0.01), "(0.1,1]": (0.38, 0.015), "(1,5]": (0.53, 0.015)},
        "10X up": {"(0,0.1]": (0.02, 0.005), "(0.1,1]": (0.27, 0.02), "(1,5]": (0.68, 0.01)},
        "10X down": {"(0,0.1]": (0.25, 0.02), "(0.1,1]": (0.32, 0.015), "(1,5]": (0.01, 0.005)},
        "admixture": {"low": (0.03, 0.02), "high": (0.68, 0.09)},
    },
    "(0.1,0.25]": {
        "recombination up": {"(0,0.1]": (0.43, 0.015), "(0.1,1]": (0.66, 0.015), "(1,5]": (0.90, 0.01)},
        "recombination down": {"(0,0.1]": (0.26, 0.02), "(0.1,1]": (0.67, 0.015), "(1,5]": (0.81, 0.01)},
        "10X up": {"(0,0.1]": (0.06, 0.01), "(0.1,1]": (0.69, 0.015), "(1,5]": (0.98, 0.005)},
        "10X down": {"(0,0.1]": (0.46, 0.02), "(0.1,1]": (0.85, 0.01), "(1,5]": (0.09, 0.005)},
        "admixture": {"low": (0.16, 0.07), "high": (0.97, 0.01)},
    },
}
PHLAG_PAPER_REPLICATE_F1 = {
    "(0,0.1]": {
        "recombination up": {"(0,0.1]": (0.25, 0.105), "(0.1,1]": (0.275, 0.095), "(1,5]": (0.66, 0.10)},
        "recombination down": {"(0,0.1]": (0.205, 0.095), "(0.1,1]": (0.40, 0.10), "(1,5]": (0.455, 0.08)},
        "10X up": {"(0,0.1]": (0.065, 0.05), "(0.1,1]": (0.03, 0.03), "(1,5]": (0.645, 0.08)},
        "10X down": {"(0,0.1]": (0.135, 0.075), "(0.1,1]": (0.305, 0.09), "(1,5]": (0.075, 0.015)},
        "admixture": PHLAG_PAPER_F1["(0,0.1]"]["admixture"],
    },
    "(0.1,0.25]": {
        "recombination up": {"(0,0.1]": (0.51, 0.125), "(0.1,1]": (0.50, 0.11), "(1,5]": (0.96, 0.03)},
        "recombination down": {"(0,0.1]": (0.46, 0.10), "(0.1,1]": (0.66, 0.095), "(1,5]": (0.85, 0.055)},
        "10X up": {"(0,0.1]": (0.155, 0.085), "(0.1,1]": (0.38, 0.10), "(1,5]": (0.955, 0.035)},
        "10X down": {"(0,0.1]": (0.60, 0.105), "(0.1,1]": (0.615, 0.115), "(1,5]": (0.225, 0.03)},
        "admixture": PHLAG_PAPER_F1["(0.1,0.25]"]["admixture"],
    },
}
PHLAG_PAPER_FPR_TPR = {
    "(0,0.1]": {
        "recombination up": (0.03, 0.51),
        "recombination down": (0.03, 0.44),
        "10X up": (0.03, 0.45),
        "10X down": (0.07, 0.24),
        "admixture": (0.07, 0.52),
    },
    "(0.1,0.25]": {
        "recombination up": (0.02, 0.70),
        "recombination down": (0.03, 0.69),
        "10X up": (0.05, 0.68),
        "10X down": (0.02, 0.10),
        "admixture": (0.04, 0.67),
    },
}


def _merged_category_sort_key(value):
    try:
        return (0, _MERGED_CATEGORY_ORDER.index(value))
    except ValueError:
        return (1, value)


def _x_bin_category_compatible(names, level_combo):
    """Prunes a resolve_configs_cartesian level_combo where the "x_bin" level's
    value and a crossed "category"/"merged_category" level's value disagree on
    admixture-ness -- a branch-length x_bin (e.g. "(0,0.1]") can only ever
    have rows when category is non-admixture, and "low"/"high" only when
    category=="admixture" (see benchmark.py's x_bin assignment: admixture
    rows get their subcategory directly as x_bin, everything else gets a
    branch-length bin). Without this, agg=["x_bin", "category"] (or
    "merged_category") emits a config for every one of the other, impossible
    pairings too -- always zero rows once row-filtered, just wasted
    find_matching_leaf_dirs/plotting work and empty clutter. names: the
    level names in the same order as level_combo (see resolve_configs_cartesian);
    a no-op (returns True) unless both "x_bin" and a category-like name are
    present. subcategory is intentionally not covered -- for admixture rows
    it already holds the low/high value directly, so an incompatible
    subcategory x x_bin pairing already naturally row-filters to zero without
    needing this prune."""
    by_name = dict(zip(names, level_combo))
    x_bin_entry = by_name.get("x_bin")
    if x_bin_entry is None:
        return True
    is_admix_bin = x_bin_entry[2] in ADMIXTURE_BINS
    for cat_name in _CATEGORY_LIKE_AXES:
        cat_entry = by_name.get(cat_name)
        if cat_entry is None:
            continue
        if is_admix_bin != (cat_entry[2] == "admixture"):
            return False
    return True

_MIRRORED_SPECS = CASTER_ARG_SPECS + PHLAG_ARG_SPECS
_PARSER = _build_parser()
_PARSER.add_argument("--emission-param", dest="emission_param", default="free")
_RECORDED_DEFAULTS = {**PHLAG_SEGMENT_DEFAULTS, "emission_param": "free"}
# Every option string (both short and long form, e.g. "-n" AND "--normalize")
# that the real parser registers for a mirrored dest -- not just the one
# canonical token CASTER_ARG_SPECS/PHLAG_ARG_SPECS happens to list, so a
# config dict can use either alias interchangeably, same as the actual CLI.
_MIRRORED_DESTS = {dest for dest, _, _ in _MIRRORED_SPECS} | {"emission_param"}
_FLAG_TO_DEST = {
    option_string: action.dest
    for option_string, action in _PARSER._option_string_actions.items()
    if action.dest in _MIRRORED_DESTS
}
# dest -> its long ("--...") option string, so a short-form token (e.g. "-w")
# and its long form (e.g. "--window-size") both resolve to the same canonical
# name -- used to always display/store a flag's long form (see
# _canonical_flag_name) regardless of which alias a caller's axes/agg used.
_DEST_TO_LONG_FLAG = {}
for _option_string, _action in _PARSER._option_string_actions.items():
    if _action.dest in _MIRRORED_DESTS and _action.dest not in _DEST_TO_LONG_FLAG:
        _long_opts = [opt for opt in _action.option_strings if opt.startswith("--")]
        if _long_opts:
            _DEST_TO_LONG_FLAG[_action.dest] = _long_opts[0]
del _option_string, _action, _long_opts


def _canonical_flag_name(flag):
    """flag's long-form option string (e.g. "-w" -> "--window-size"), or flag
    itself unchanged if it's not a recognized mirrored CLI flag (e.g. a
    _DATA_COLUMN_AXES/_DYNAMIC_ROW_AXES name like "x_bin") or already has no
    short alias."""
    dest = _FLAG_TO_DEST.get(flag)
    return _DEST_TO_LONG_FLAG.get(dest, flag) if dest is not None else flag
_PARAM_LABELS = {"emission_lambda": "lambda"}

_CONFIG_PALETTE = (
    list(matplotlib.colormaps["tab10"].colors)
    + [matplotlib.colormaps["tab20b"].colors[i] for i in (0, 4, 8, 12, 16)]
    + [matplotlib.colormaps["tab20c"].colors[i] for i in (0, 4, 8, 12)]
)


def _match_cache_key(flags):
    """Canonical, order-independent, hashable key for a find_matching_leaf_dirs
    `flags` dict -- used to memoize its result (see find_matching_leaf_dirs)."""
    return tuple(sorted(flags.items()))


class CrossRunAnalysis:
    """
    Compares finished phlag benchmark runs against each other, reading their
    runs.tsv/args.json straight off disk (no benchmark re-run). A config is a
    set of CLI flags (resolve_configs*), resolved to matching leaf run dirs via
    each run's own args.json, then all rendered via plot: pass "tpr_fpr" in
    `metrics` for the per-config-color TPR/FPR scatter grid, any other metric
    name for a bar/violin subplot -- including the gt_stats.txt-derived ones
    (em_gt_hd, null_/alt_/pooled_mean_<topo>, null_/alt_/pooled_var_<topo>,
    null_/alt_/pooled_cov_<topo1>_<topo2>), which benchmark.py resummarizes
    from caster's gt_stats.txt into every runs.tsv row, so there's no need to
    read gt_stats.txt directly here any more. `root` overrides the default
    store/phlag location.
    """

    def __init__(self, root=None):
        self.root = root if root is not None else STORE_ROOT
        self.metrics = SWEEP_METRICS
        self._leaf_dirs = None
        self._row_filters = {}
        self._config_axes = {}
        self._config_rank = {}
        self._agg_axis_names = {}
        self._restrictions = {}
        self.title_base = {}
        self._beta_axis_idx = None
        self._config_beta = {}
        self._config_implicit = {}
        self._agg_given = True
        self._raw_beta_cache = {}
        self._last_configs = None
        self._color_cache = {}
        self._bin_specs = {}
        self._args_json_cache = {}
        self._match_cache = {}

    def pick_nodes_by_em_gt_hd(self, quantiles=(0.05, 0.35, 0.65, 0.95), dist_type="gaussian",
                                window=1000, step=None):
        """
        Ranks every source_leaf (category/subcategory/node) in
        self.root/<dist_type>/w<window>_s<step>/runs.tsv (the free/free, no
        --rho/--beta base run -- see bench/script.sh) by its mean em_gt_hd
        across patterns, and returns the source_leaf at each requested
        quantile: a small set of nodes spanning easy (low em_gt_hd) to hard
        (high em_gt_hd) test cases. step defaults to window (non-overlapping).
        """
        step = window if step is None else step
        run_dir = Path(self.root) / dist_type / f"w{format_val(window)}_s{format_val(step)}"
        runs = pd.read_csv(run_dir / "runs.tsv", sep="\t")
        node_hd = runs.groupby("source_leaf")["em_gt_hd"].mean().sort_values()
        idx = [int(q * (len(node_hd) - 1)) for q in quantiles]
        return node_hd.iloc[idx]

    # ---- flag/config resolution ----

    def refresh_leaf_dirs(self):
        """Forces the next find_matching_leaf_dirs/_discover_flag_values call to
        rescan disk instead of reusing the cached leaf-dir list -- call this if
        you've run more benchmarks since creating this CrossRunAnalysis. Also
        drops the args.json content cache and find_matching_leaf_dirs' own
        result cache (see _get_args/_match_cache), since both would otherwise
        keep serving stale answers for dirs whose args.json changed or is new."""
        self._leaf_dirs = None
        self._args_json_cache = {}
        self._match_cache = {}
        _windowless_runs_df.cache_clear()

    def _get_args(self, run_dir):
        """_read_args_json(run_dir/"args.json"), cached per run_dir for this
        instance's lifetime -- resolve_configs_cartesian's cartesian expansion
        calls find_matching_leaf_dirs once per (agg-flag combo x pool combo),
        and a multi-axis agg that includes a _DATA_COLUMN_AXES/_DYNAMIC_ROW_AXES
        name (e.g. "x_bin") multiplies that by every value of an axis that
        doesn't even affect which dirs match (it only narrows rows after
        loading -- see resolve_configs_cartesian's docstring), so the same
        args.json files were otherwise being re-read and re-parsed from disk
        dozens to hundreds of times per resolve_configs_cartesian call."""
        cached = self._args_json_cache.get(run_dir)
        if cached is None:
            cached = _read_args_json(run_dir / "args.json")
            self._args_json_cache[run_dir] = cached
        return cached

    def _all_leaf_dirs(self):
        """Every leaf run dir (own args.json + runs.tsv) under self.root, scanned
        once per instance and cached (see refresh_leaf_dirs). Walks args.json
        files directly (Path.rglob("args.json")) rather than
        discover_archived_run_dirs -- that one's rglob("reports") + a nested
        rglob("*.tsv") per match exists to find populated report *archives* for
        --rerun/--copy, which is a much more expensive tree walk than we need
        here and was making every find_matching_leaf_dirs/resolve_configs*
        call (called repeatedly per cartesian combo) rescan the whole store/
        tree from scratch -- args.json presence is all we actually check.

        Gated on analysis.tsv, not runs.tsv, existing -- benchmark.py's
        summarize() writes runs.tsv first and analysis.tsv last (see its own
        _is_finished_run), so a leaf whose run_all() is still in progress
        (e.g. a background --copy/--rerun still working through sibling
        combo dirs like w10_s8/repulsion while w10_s8 itself is already
        done) can have an args.json (written up front) with no runs.tsv yet
        at all, or -- in the narrow window between those two writes -- a
        runs.tsv that isn't actually the final one. Requiring analysis.tsv
        matches benchmark.py's own definition of "done" exactly."""
        if self._leaf_dirs is None:
            self._leaf_dirs = sorted(
                p.parent for p in Path(self.root).rglob("args.json")
                if (p.parent / "analysis.tsv").exists()
            )
        return self._leaf_dirs

    def find_matching_leaf_dirs(self, flags, exclude_keywords=()):
        """All leaf run dirs (own args.json + runs.tsv) anywhere under self.root
        (see _all_leaf_dirs) whose args.json satisfies every flag in `flags`.
        exclude_keywords: skip any run dir whose path contains one of these
        substrings (e.g. "tau", "boost", "gd") -- one-off directory-name
        suffixes for experiment variants that never became a tracked args.json
        flag, so flags-matching alone can't tell them apart from their parent
        config. Result is memoized by (flags, exclude_keywords) for this
        instance's lifetime (see _match_cache_key) -- resolve_configs_cartesian
        calls this once per (agg-flag combo x pool combo), and a multi-axis agg
        including a row-filter axis (e.g. "x_bin", "fraction_bin") repeats the
        exact same flags for every value of that axis (it only changes the
        row_filter, not agg_flags), so this collapses those repeats to one
        real scan."""
        key = (tuple(exclude_keywords), _match_cache_key(flags))
        cached = self._match_cache.get(key)
        if cached is not None:
            return cached
        matches = []
        for run_dir in self._all_leaf_dirs():
            if any(kw in str(run_dir) for kw in exclude_keywords):
                continue
            recorded = self._get_args(run_dir)
            if _matches_flags(recorded, flags):
                matches.append(run_dir)
        self._match_cache[key] = matches
        return matches

    def _discover_flag_values(self, flag, exclude_keywords=()):
        """Every distinct value `flag`'s dest takes across args.json on disk
        (None included, for dirs missing that dest) -- used by
        resolve_configs_cartesian to auto-populate an agg axis that wasn't
        given an explicit value_spec in `axes`."""
        dest = _FLAG_TO_DEST.get(flag)
        if dest is None:
            raise KeyError(f"{flag!r} is not one of benchmark.py's mirrored caster/phlag flags")
        values = set()
        for run_dir in self._all_leaf_dirs():
            if any(kw in str(run_dir) for kw in exclude_keywords):
                continue
            recorded = self._get_args(run_dir)
            values.add(_recorded_value(recorded, dest))
        return sorted(values, key=_param_value_sort_key)

    def _threshold_flag_values(self, flag, spec, exclude_keywords=()):
        """"<5k"/">5k": every on-disk value of `flag` strictly below/above the bound."""
        bound = int_or_abbrev(spec[1:])
        keep = (lambda v: v < bound) if spec[0] == "<" else (lambda v: v > bound)
        return [v for v in self._discover_flag_values(flag, exclude_keywords)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and keep(v)]

    def _unlisted_toggle_flags(self, agg_seq, axes, exclude_keywords=(), values=False):
        """On/off flags (plus value flags if `values`) that aren't pinned in
        `axes` or named anywhere in agg and take more than one value on disk."""
        named = set(axes)
        for name in agg_seq:
            named.update(m for m in name if m != "...") if isinstance(name, tuple) else named.add(name)
        seen = {_FLAG_TO_DEST[f] for f in named if f in _FLAG_TO_DEST}
        extra = []
        for flag, dest in _FLAG_TO_DEST.items():
            if dest in seen:
                continue
            seen.add(dest)
            if (values or isinstance(_PARSER._option_string_actions[flag], argparse._StoreTrueAction)) \
                    and len(self._discover_flag_tuple_values((flag,), exclude_keywords)) > 1:
                extra.append(_canonical_flag_name(flag))
        return extra

    def _expand_tuple_wildcards(self, agg_seq, axes, exclude_keywords=()):
        """Replaces a "..." member of a tuple agg element with
        _unlisted_toggle_flags, so new flags are categorized automatically
        (realized combinations only -- see _discover_flag_tuple_values)."""
        if not any(isinstance(name, tuple) and "..." in name for name in agg_seq):
            return agg_seq
        extra = self._unlisted_toggle_flags(agg_seq, axes, exclude_keywords)
        return [
            tuple(m for member in name for m in (extra if member == "..." else (member,)))
            if isinstance(name, tuple) else name
            for name in agg_seq
        ]

    def _discover_flag_tuple_values(self, flags, exclude_keywords=()):
        """Every distinct joint combination of `flags`' dests that actually
        co-occurs in some leaf dir's args.json on disk (None included, for a
        dest missing from that dir) -- used by resolve_configs_cartesian for
        a tuple agg element (e.g. ("--ap", "--annealing")), so it
        categorizes by only the value combinations real runs actually took
        instead of the full individual cartesian product of each flag's own
        _discover_flag_values (which can include combinations no run ever
        used)."""
        dests = []
        for flag in flags:
            dest = _FLAG_TO_DEST.get(flag)
            if dest is None:
                raise KeyError(f"{flag!r} is not one of benchmark.py's mirrored caster/phlag flags")
            dests.append(dest)
        off_is_missing = [
            isinstance(_PARSER._option_string_actions[flag], argparse._StoreTrueAction) for flag in flags
        ]
        combos = set()
        for run_dir in self._all_leaf_dirs():
            if any(kw in str(run_dir) for kw in exclude_keywords):
                continue
            recorded = self._get_args(run_dir)
            combos.add(tuple(
                False if off and recorded.get(dest) is None else _recorded_value(recorded, dest)
                for dest, off in zip(dests, off_is_missing)
            ))
        return sorted(combos, key=lambda combo: tuple(_param_value_sort_key(v) for v in combo))

    def _discover_row_values(self, col, exclude_keywords=()):
        """Every distinct value `col` takes across runs.tsv rows on disk --
        used by resolve_configs_cartesian to auto-populate a _DYNAMIC_ROW_AXES
        agg axis (e.g. "category") that wasn't given an explicit list in
        `axes`. merged_category isn't itself a runs.tsv column, so it's
        derived here the same way _load_configs_runs derives it. em_hd_bin/
        em_gt_hd_bin aren't runs.tsv columns either, and unlike
        merged_category their enumeration isn't cheap to rederive from a
        single row (it's the bin edges compute_hd_bin_edges already computed
        once, over the whole dataset) -- so this returns those edges'
        labels directly instead of rescanning disk; raises if
        compute_hd_bin_edges hasn't been called yet."""
        if col in ("em_hd_bin", "em_gt_hd_bin"):
            if col not in self._bin_specs:
                raise RuntimeError(f"{col!r} needs cra.compute_hd_bin_edges() called first")
            return [label for label, _, _ in self._bin_specs[col][1]]
        values = set()
        for run_dir in self._all_leaf_dirs():
            if any(kw in str(run_dir) for kw in exclude_keywords):
                continue
            p = run_dir / "runs.tsv"
            if not p.exists():
                continue
            df = _drop_hidden_admixture(pd.read_csv(p, sep="\t"))
            if col == "merged_category":
                if "category" in df.columns and "subcategory" in df.columns:
                    values.update(_merged_category(df["category"], df["subcategory"]).unique())
            elif col == "outlier_pct":
                if "pattern" in df.columns:
                    values.update(_outlier_pct(df["pattern"]).dropna().unique())
            elif col in df.columns:
                values.update(df[col].unique())
        if col == "merged_category":
            return sorted(values, key=_merged_category_sort_key)
        return sorted(values, key=_token_sort_key)

    def _discover_numeric_column(self, col, exclude_keywords=()):
        """Every (non-null) value `col` takes across runs.tsv rows on disk --
        used by compute_hd_bin_edges to get em_hd/em_gt_hd's full observed
        range for fixed bin edges, same disk scan shape as
        _discover_row_values but returning raw values (for min/max), not
        distinct ones."""
        values = []
        for run_dir in self._all_leaf_dirs():
            if any(kw in str(run_dir) for kw in exclude_keywords):
                continue
            p = run_dir / "runs.tsv"
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            if col in df.columns:
                values.extend(df[col].dropna().tolist())
        return values

    def compute_hd_bin_edges(self, exclude_keywords=()):
        """Computes and caches HD_NUM_BINS=6 equal-width bin edges (see
        _compute_hd_bin_edges) for em_hd and em_gt_hd across the WHOLE
        dataset on disk, registering "em_hd_bin"/"em_gt_hd_bin" as
        row-filterable axes usable in plot()'s `agg`/`axes` just like
        x_bin or fraction_bin (see _DYNAMIC_ROW_AXES/_discover_row_values) --
        _load_configs_runs adds the actual columns from these fixed edges
        (self._bin_specs). Fixed once over every run on disk rather than
        recomputed per plot() call from whatever configs that call happens
        to pool (contrast grid_by="hd_bin", which does the latter on
        purpose -- see _plot_heatmap_hd_bin), so the bin boundaries (and
        therefore what's comparable across figures) stay the same across
        different agg/axes choices. Call this once, right after
        construction, before any plot()/agg use of em_hd_bin/em_gt_hd_bin --
        a use before this raises (see _discover_row_values)."""
        for col, bin_col in (("em_hd", "em_hd_bin"), ("em_gt_hd", "em_gt_hd_bin")):
            values = self._discover_numeric_column(col, exclude_keywords)
            self._bin_specs[bin_col] = (col, _compute_hd_bin_edges(values))

    def resolve_configs(self, flags_specs, labels=None, exclude_keywords=()):
        """flags_specs: list of {cli_flag: value} dicts, one per config, e.g.
        [{"-w": "50k"}, {"-w": "100k"}, {"-w": "250k"}]. labels defaults to
        _combo_label(flags, "(base)") per config. exclude_keywords: forwarded
        to find_matching_leaf_dirs."""
        configs = []
        for i, flags in enumerate(flags_specs):
            label = labels[i] if labels else _combo_label(flags, "(base)")
            dirs = self.find_matching_leaf_dirs(flags, exclude_keywords)
            if dirs:
                configs.append((label, dirs))
        return configs

    def resolve_configs_cartesian(self, axes, exclude_keywords=(), agg=()):
        """axes: {flag: value_spec} dict, one entry per flag -- e.g.
        {"-d": "gaussian", "-s": "1k", "--double-variance-init": True, "--lam": [1.0, 1.5]}.
        A list value_spec is an explicit multi-value axis, iterated as-is. A
        bare (non-list) value_spec pins that flag to exactly that one value --
        equivalent to wrapping it in a single-element list -- e.g. -d/-s above
        each match only their given value, no other state. exclude_keywords:
        see resolve_configs.

        agg: the subset of axes' keys to categorize by -- their cartesian
        combinations each become their own config/bar, and their values are all
        that appears in the label/legend. An element of agg may itself be a
        tuple of flags (e.g. ("--ap", "--annealing")) to categorize by that
        combined axis's REALIZED joint value combinations only -- those that
        actually co-occur in some leaf dir's args.json, discovered via
        _discover_flag_tuple_values, rather than the full individual
        cartesian product of each flag's own values (which can include
        combinations no run ever took). This combined axis still
        cartesian-products with every other agg element like any other
        level. Every OTHER axis in `axes` (not in
        agg) is pooled: all of its cartesian states' matching run dirs are
        unioned into each agg-combo's bucket instead of each state producing
        its own config. E.g. axes={"-w": [...4 sizes], "-s": ["1k", "80k"],
        "-d": "gaussian", "--lam": [1.0, 1.5]}, agg=["-w", "-s"] categorizes by
        (window, step) -- one bar per (w, s) pair that has any matching data,
        pooling both lam values (and, since -d's off-state has no matching
        data, staying pinned to gaussian) into each. A bare string is treated
        as a single flag, not iterated character-by-character. Default agg=()
        categorizes by every axis (pools nothing) -- one config per full
        cartesian combo.

        A flag named in `agg` doesn't need its own entry in `axes` -- if it's
        missing there, its distinct recorded values are auto-discovered from
        disk instead (see _discover_flag_values), so agg=["-w", "-s"] alone
        (no "-w"/"-s" keys in axes at all) still categorizes by every (window,
        step) pair actually present on disk, without having to enumerate them
        by hand.

        A name in `agg` (or `axes`) from _DATA_COLUMN_AXES -- "x_bin",
        "fraction_bin", "column" -- or _DYNAMIC_ROW_AXES -- "category",
        "subcategory", "merged_category" (category+subcategory space-joined,
        except admixture -- see _merged_category) -- categorizes by a
        runs.tsv *row* value
        instead of an args.json flag: every leaf dir's runs.tsv has many rows
        spanning all of these, so unlike a flag they can't narrow which dirs
        match -- each resulting config still gets that agg-combo's full dir
        list, but with a row filter attached (applied when the config's rows
        get loaded, e.g. by plot/print_configs -- see
        _load_configs_runs) restricting it to just that x_bin/fraction_bin/
        column value. Its `axes` entry, if given, must be an explicit list
        (a bare value_spec doesn't apply to these -- there's no single-flag
        state to pin, only a set of values to filter rows by). Left out of
        `axes` entirely, a
        _DATA_COLUMN_AXES name uses its fixed default enumeration (already
        known from benchmark.py); a _DYNAMIC_ROW_AXES name instead
        auto-discovers its default list from runs.tsv on disk (like a CLI
        flag's own auto-discovery), since category/subcategory have no fixed
        enumeration.
        Combined with a flag axis in the same agg (e.g. agg=["-w", "x_bin"]),
        every (window, x_bin) pair becomes its own config. Note: the
        fraction_bin x column x x_bin breakdown grids (plot's "tpr_fpr"
        and grid_by="hd_bin") already show these dimensions natively and
        don't apply this row filter -- pooling further by row here mainly
        matters for the flat/heatmap views. A name in `agg` that's neither a
        key of `axes`, a recognized CLI flag, nor a _DATA_COLUMN_AXES name is
        silently skipped (not auto-discovered, not agg'd, no row filter).

        A "-s" value in (0, 1] is a step-size *fraction*, resolved per window
        size (round(fraction * window); 1.0 means step == window -- see
        _as_step_fraction for why this is (0, 1], a wider range than
        caster.py's own step_size_or_fraction at run time) rather than tried
        as a literal step size against every window equally -- args.json's
        recorded step_size is always an already-resolved absolute int, never
        a bare fraction, so e.g. {"-w": ["50k", "100k"], "-s": [0.8]} means
        "step = 80% of window" and, when matching dirs on disk, resolves to
        step_size==40000 for the w=50k combo and step_size==80000 for the
        w=100k combo (see _resolve_step_fraction) -- never tried as a
        literal step_size==0.8. This resolution happens per (window, step)
        combo, so -w and -s stay two independent agg axes/levels even when
        -s is fractional -- agg=["-w", "-s"] cartesian-products them like
        any other two-flag agg (one x value per window, one y value per
        fraction, e.g. for a heatmap). To categorize by only the (window,
        step) pairs actually realized on disk instead, use the combined
        tuple axis agg=[("-w", "-s")] (see the tuple-axis paragraph above)
        -- ordinary tuple-axis behavior, not special-cased for -w/-s. A
        literal "-s" value (>1, or a string like "1k") is left alone and
        still pairs with every window size independently, same as ever.

        Each resulting config also records its own per-agg-axis value, in
        `agg`'s order, on self._config_axes[label] -- plot uses this to
        arrange bars/subplots hierarchically instead of one flat row: agg[0]
        -> bars within a group (hue/color, with a shared legend), agg[1] ->
        x-position (bar groups), agg[2] -> subplot columns, agg[3] -> subplot
        rows. agg=["x_bin", "-w"] (2 levels) therefore draws one bar-group
        per window size, each holding one bar per x_bin; adding a 3rd axis
        makes one column of subplots per its value, a 4th makes one row per
        its value. The axis meant to drive the legend always belongs at
        agg[0] -- put whichever axis you want colored/legended first.

        An empty string "" as one of agg's elements holds that slot's
        position (a constant "(none)" value in self._config_axes, no label
        text, no dir restriction) instead of being dropped and shifting
        every later axis up into its spot -- e.g. agg=["", "x_bin",
        "merged_category", "fraction_bin"] keeps x_bin/merged_category/
        fraction_bin at agg[1]/[2]/[3] (bar/col/row) rather than
        recompacting them into agg[0]/[1]/[2]. plot's _plot_grouped
        collapses the legend to a single pooled entry (the metric averaged
        over every config in the figure) when agg[0] (hue) is left as this
        placeholder while agg[1] (bar) has real values -- see its
        docstring.

        A list of such dicts resolves each one and unions the results: configs
        with the same label pool their dirs (deduplicated), and the title
        restriction joins each dict's own with " | "."""
        if isinstance(axes, (list, tuple)):
            results, beta_idx = [], None
            for a in axes:
                results.append(self.resolve_configs_cartesian(a, exclude_keywords, agg))
                beta_idx = self._beta_axis_idx if beta_idx is None else beta_idx
            configs = self._union_configs(results)
            self._beta_axis_idx = beta_idx
            return configs
        self._agg_given = bool(agg)
        agg_seq = [agg] if isinstance(agg, str) else (list(agg) if agg else list(axes))
        axes = dict(axes)
        given_axes = dict(axes)
        agg_seq = self._expand_tuple_wildcards(agg_seq, axes, exclude_keywords)
        implicit_flags = self._unlisted_toggle_flags(agg_seq, axes, exclude_keywords, values=True)
        unlisted_on = {}
        agg_set = set(agg_seq)
        sources = axes.pop("phlag", None)
        if sources is None:
            sources = ["gtrees", "msa"] if "phlag" in agg_seq else ["msa"]
        sources = _register_phlag_sources(sources)

        unagged = [n for n in axes if (n in _DATA_COLUMN_AXES or n in _DYNAMIC_ROW_AXES) and n not in agg_set]
        if unagged:
            raise ValueError(f"row-level axes {unagged} must also be in agg; outside agg they can't be pooled/restricted")

        data_col_axes = {}
        for name in list(axes.keys()) + agg_seq:
            if name in _DATA_COLUMN_AXES and name not in data_col_axes:
                data_col_axes[name] = axes.pop(name, _DATA_COLUMN_AXES[name])
            elif name in _DYNAMIC_ROW_AXES and name not in data_col_axes:
                data_col_axes[name] = axes.pop(name, None) or self._discover_row_values(name, exclude_keywords)
        agg_set -= set(data_col_axes)

        for flag, spec in list(axes.items()):
            if flag not in _FLAG_TO_DEST:
                continue
            is_threshold = lambda v: isinstance(v, str) and v[:1] in ("<", ">")
            if is_threshold(spec):
                axes[flag] = self._threshold_flag_values(flag, spec, exclude_keywords)
            elif isinstance(spec, list) and any(map(is_threshold, spec)):
                bounds = [set(self._threshold_flag_values(flag, v, exclude_keywords)) for v in spec if is_threshold(v)]
                in_range = [v for v in self._discover_flag_values(flag, exclude_keywords) if v in set.intersection(*bounds)]
                merged = list(dict.fromkeys(in_range + [v for v in spec if not is_threshold(v)]))
                axes[flag] = sorted(merged, key=lambda v: _token_sort_key(str(v)))
        for flag in agg_seq:
            if flag not in axes and flag not in data_col_axes and flag in _FLAG_TO_DEST:
                axes[flag] = self._discover_flag_values(flag, exclude_keywords)
        # A tuple agg element (e.g. ("--rho", "--beta")) is a single agg_set
        # member, not its individual flags -- so excluding pool_axes by
        # membership in agg_set alone would leave "--rho"/"--beta" (if also
        # given their own entries in `axes`) as SEPARATE pool axes. Since a
        # pool combo's values get merged in over the tuple axis's own
        # realized (rho, beta) pair below (see the pool_combo loop), that
        # would silently overwrite/erase the tuple axis's distinguishing
        # combo for every bucket, unioning the same dirs into all of them.
        # Flattening tuple members here keeps them out of pool_axes so only
        # the tuple axis controls their value.
        agg_members = set()
        for name in agg_seq:
            agg_members.update(name) if isinstance(name, tuple) else agg_members.add(name)
        pool_axes = {flag: spec for flag, spec in axes.items() if flag not in agg_members}
        pool_axis_states = [_cartesian_axis_states(flag, spec) for flag, spec in pool_axes.items()]
        pool_combos = list(itertools.product(*pool_axis_states)) if pool_axis_states else [()]
        at_base = {f for f in pool_axes if f in self.title_base and given_axes.get(f) == self.title_base[f]}
        restriction = _pool_axes_label(pool_axes, exclude=at_base)

        # One "level" per (still-existing) name in agg_seq -- a level's states are
        # (contribution_dict, is_flag, display_value) triples. A fractional -s
        # value's display is its raw fraction (see _combo_label_part); the
        # matching absolute step_size is resolved per (window, step) combo
        # below (see _resolve_step_fraction), not here.
        levels = []
        consumed_flags = set()
        beta_level_idx = None
        for name in agg_seq:
            if name in consumed_flags:
                continue
            if name == "":
                # An explicit "skip this axis" placeholder -- keeps its slot
                # (a constant "(none)" value, no label text, no dir
                # restriction) instead of being dropped and shifting every
                # later agg axis into its position -- see plot's
                # docstring/_plot_grouped for what an empty axis 0 (hue) or
                # axis 1 (bar) does to rendering.
                states = [({}, False, "")]
            elif isinstance(name, tuple) and "phlag" in name:
                rest = [n for n in name if n != "phlag"]
                rest_states = [
                    {k: v for fs in combo for k, v in fs.items()}
                    for combo in itertools.product(*(
                        _cartesian_axis_states(f, axes[f] if f in axes else self._discover_flag_values(f, exclude_keywords))
                        for f in rest
                    ))
                ]
                states = []
                for source in sources:
                    if source in WINDOWLESS_PHLAG_SOURCES:
                        states.append(({"phlag": source}, False, source))
                        continue
                    for fs in rest_states:
                        disp = " ".join(part for k, v in fs.items() if (part := _combo_label_part(k, v)) is not None)
                        states.append(({"phlag": source, **fs}, True, f"{source} {disp}".strip()))
                consumed_flags.update(rest)
            elif isinstance(name, tuple):
                # A combined multi-flag axis (e.g. ("--ap", "--annealing"))
                # -- states are the REALIZED joint combinations these flags
                # actually take together on disk (_discover_flag_tuple_values),
                # not the full individual cartesian product of each flag's
                # own values, so it only categorizes by combinations real
                # runs took. This one level still cartesian-products with
                # every other agg element via level_combos below, same as
                # any other level.
                states = []
                combos = [
                    combo for combo in self._discover_flag_tuple_values(name, exclude_keywords)
                    if all(
                        any(_matches_flags({_FLAG_TO_DEST[flag]: v}, fs)
                            for fs in _cartesian_axis_states(flag, axes[flag]))
                        for flag, v in zip(name, combo) if flag in axes
                    )
                ]
                numeric = [
                    flag for i, flag in enumerate(name)
                    if any(isinstance(c[i], (int, float)) and not isinstance(c[i], bool) for c in combos)
                ]
                is_beta = _BETA_FLAGS <= set(name)
                if is_beta:
                    beta_level_idx = len(levels)
                for combo in combos:
                    fs = dict(zip(name, combo))
                    disp = " ".join(
                        part for k, v in fs.items()
                        if not (is_beta and k == "--beta")
                        and (part := _tuple_member_label(k, v) if k in numeric else _combo_label_part(k, v)) is not None
                    )
                    states.append((fs, True, disp))
                consumed_flags.update(name)
            elif name == "phlag":
                states = [({"phlag": v}, False, v) for v in sources]
            elif name in data_col_axes:
                states = [({name: v}, False, _row_axis_display(name, v)) for v in data_col_axes[name]]
            elif name in axes:
                flag_states = _cartesian_axis_states(name, axes[name])
                states = []
                for fs in flag_states:
                    disp = " ".join(
                        part for k, v in fs.items()
                        if (part := _combo_label_part(k, v)) is not None
                    )
                    states.append((fs, True, disp))
                for fs in flag_states:
                    consumed_flags.update(fs.keys())
            else:
                continue  # not a flag, not a data column -- silently skipped, as before
            levels.append((name, states))
        level_names = [_strip_flag(_canonical_flag_name(name)) if name in _FLAG_TO_DEST else name for name, _ in levels]
        level_key_names = [name for name, _ in levels]

        configs = []
        level_combos = list(itertools.product(*[list(enumerate(states)) for _, states in levels])) if levels else [()]
        for indexed_combo in level_combos:
            combo_rank = tuple(i for i, _ in indexed_combo)
            level_combo = tuple(state for _, state in indexed_combo)
            if not _x_bin_category_compatible(level_key_names, level_combo):
                continue
            agg_flags, row_filter, axis_values, label_bits = {}, {}, [], []
            for contribution, is_flag, disp in level_combo:
                axis_values.append(disp if disp else ("(base)" if is_flag else "(none)"))
                # A flag level always contributes something to the label, even
                # "(base)" for its off state, so two configs whose only
                # difference is a flag's on/off state never collide onto the
                # same label text (which self._row_filters/_config_axes key
                # on) -- a data-column level's disp is never empty, so this
                # only changes behavior for flag levels.
                if disp or is_flag:
                    label_bits.append(disp if disp else "(base)")
                if is_flag:
                    if "phlag" in contribution:
                        row_filter["phlag"] = contribution["phlag"]
                    agg_flags.update({k: v for k, v in contribution.items() if k != "phlag"})
                else:
                    row_filter.update(contribution)
            label = " ".join(label_bits) or "(none)"
            combo_sources = [row_filter.pop("phlag")] if "phlag" in row_filter else sources
            dirs = []
            for source in combo_sources:
                if source in WINDOWLESS_PHLAG_SOURCES:
                    if not _windowless_runs_df(source).empty:
                        dirs.append(Path(PHLAG_SOURCE_ROOTS[source]))
                    continue
                for pool_combo in pool_combos:
                    merged = dict(agg_flags)
                    for part in pool_combo:
                        merged.update(part)
                    merged = _resolve_step_fraction(merged)
                    dirs.extend(self.find_matching_leaf_dirs(merged, exclude_keywords))
            if not dirs:
                continue
            splits, windowless_roots = {}, {PHLAG_SOURCE_ROOTS[x] for x in WINDOWLESS_PHLAG_SOURCES}
            for d in dirs:
                recorded = {} if str(d) in windowless_roots else self._get_args(d)
                key = tuple(_implicit_flag_part(f, recorded) for f in implicit_flags)
                if agg:
                    for f, part in zip(implicit_flags, key):
                        if part:
                            unlisted_on.setdefault(f, set()).add(part)
                raw_beta = self._raw_beta(d) if beta_level_idx is not None and str(d) not in windowless_roots else None
                splits.setdefault((key, raw_beta), []).append(d)
            for (key, raw_beta), split_dirs in splits.items():
                text = " ".join(part for part in key if part)
                join = lambda base: (text if base in ("(base)", "(none)") else f"{base} {text}") if text else base
                split_values, base_label = list(axis_values), label
                if beta_level_idx is not None and raw_beta is not None:
                    rest = axis_values[beta_level_idx]
                    beta_disp = f"(beta={raw_beta:g})" if "beta'=" in rest else f"beta={raw_beta:g}"
                    split_values[beta_level_idx] = beta_disp if rest == "(base)" else f"{rest} {beta_disp}"
                    base_label = " ".join(v for v in split_values if v != "(none)")
                split_label = join(base_label)
                split_axes = [join(split_values[0])] + split_values[1:] if split_values else split_values
                existing = next((ds for lbl, ds in configs if lbl == split_label), None)
                if existing is not None:
                    existing.extend(split_dirs)
                    continue
                configs.append((split_label, split_dirs))
                self._row_filters[split_label] = row_filter
                self._config_axes[split_label] = split_axes
                self._config_rank[split_label] = tuple(
                    (i, raw_beta if j == beta_level_idx and raw_beta is not None else float("-inf"))
                    for j, i in enumerate(combo_rank)
                )
                self._agg_axis_names[split_label] = level_names
                self._restrictions[split_label] = restriction
                self._config_beta[split_label] = raw_beta
                self._config_implicit[split_label] = text
        if unlisted_on:
            warnings.warn("unlisted flags split into separate configs: " + ", ".join(
                f"{f} ({', '.join(sorted(parts))})" for f, parts in unlisted_on.items()), stacklevel=2)
            for label, _ in configs:
                vals = self._config_axes[label]
                if vals and vals[0] == "(none)":
                    vals[0] = "(base)"
        self._beta_axis_idx = beta_level_idx
        configs.sort(key=lambda c: self._config_rank[c[0]])
        return configs

    def _union_configs(self, results):
        merged = {}
        restrictions = list(dict.fromkeys(
            self._restrictions.get(label, "") for configs in results for label, _ in configs[:1]
        ))
        for configs in results:
            for label, dirs in configs:
                merged.setdefault(label, {})
                merged[label].update(dict.fromkeys(dirs))
        restriction = " | ".join(r for r in restrictions if r)
        for label in merged:
            self._restrictions[label] = restriction
        configs = [(label, list(dirs)) for label, dirs in merged.items()]
        levels = max((len(self._config_axes.get(label, [])) for label in merged), default=0)
        order = [
            {v: i for i, v in enumerate(_ordered_unique(v for r in results for v in self._axis_values(r, idx)))}
            for idx in range(levels)
        ]
        for label, _ in configs:
            self._config_rank[label] = tuple(
                (order[idx][v], -1) for idx, v in enumerate(self._config_axes.get(label, []))
            )
        configs.sort(key=lambda c: self._config_rank[c[0]])
        return configs

    def _raw_beta(self, run_dir):
        key = str(run_dir)
        if key not in self._raw_beta_cache:
            recorded = self._get_args(run_dir)
            beta, beta_prime = recorded.get("beta"), recorded.get("beta_prime")
            value = None
            if beta is not None:
                value = float(beta)
            elif beta_prime is not None:
                p = Path(run_dir) / "runs.tsv"
                n = pd.read_csv(p, sep="\t", usecols=["n_windows"])["n_windows"].dropna() if p.exists() else pd.Series()
                value = float(beta_prime) * float(n.median()) if not n.empty else None
            self._raw_beta_cache[key] = value
        return self._raw_beta_cache[key]

    def print_configs(self, configs=None, agg=(), metrics=None, exclude_keywords=()):
        """Prints each config's label, run count, and mean of every metric in
        `metrics` (pooled across all its runs -- same aggregate plot
        subplots show), followed by the args.json params that actually vary
        across its matching dirs (see _param_summary_lines). configs: a
        plot()-style {flag: value_spec} dict, resolved via
        resolve_configs_cartesian with `agg`/`exclude_keywords` (so
        print_configs(config, agg) prints the same aggregated configs
        plot(config, agg=agg) draws); an already-resolved [(label, dirs)]
        list; or None (default) for the configs from the most recent plot().
        metrics defaults to self.metrics -- the scalar metrics from the most
        recent plot() call."""
        if configs is None:
            configs = self._last_configs
        elif isinstance(configs, dict) or (isinstance(configs, list) and configs and isinstance(configs[0], dict)):
            configs = self.resolve_configs_cartesian(configs, exclude_keywords=exclude_keywords, agg=agg)
        agg = _as_list(agg)
        metrics = self.metrics if metrics is None else _as_list(metrics)
        metric_values = {metric: self._metric_by_label(configs, metric) for metric in metrics}
        for label, dirs in configs:
            stats = ", ".join(f"{metric}={metric_values[metric][label]:.4f}" for metric in metrics)
            print(f"{label} ({len(dirs)} runs, {stats}):")
            for line in _param_summary_lines(dirs):
                print(f"  {line}")

    def top_configs(self, k=5, metric="f1", configs=None, exclude_keywords=(), ascending=False):
        """Top `k` leaf run dirs under self.root ranked by mean `metric` over
        their runs.tsv (highest first, or lowest with ascending=True; NaN
        last), as a DataFrame of config (path relative to self.root) and
        metric. configs: optional {flag: value_spec} dict restricting which
        dirs are ranked -- a bare value pins the flag, a list allows any of
        its values, and "<10k"/">1k" thresholds (bare or in a list, where
        several thresholds intersect into a range) keep on-disk values past
        the bound, e.g. {"-w": ["<10k"], "-d": "gaussian"}."""
        is_threshold = lambda v: isinstance(v, str) and v[:1] in ("<", ">")
        allowed = {}
        for flag, spec in (configs or {}).items():
            specs = spec if isinstance(spec, list) else [spec]
            bounds = [set(self._threshold_flag_values(flag, s, exclude_keywords)) for s in specs if is_threshold(s)]
            allowed[flag] = [s for s in specs if not is_threshold(s)] + (list(set.intersection(*bounds)) if bounds else [])
        rows = []
        for d in self._all_leaf_dirs():
            if any(kw in str(d) for kw in exclude_keywords):
                continue
            recorded = self._get_args(d)
            if not all(any(_matches_flags(recorded, {f: v}) for v in vs) for f, vs in allowed.items()):
                continue
            df = _load_raw_config_df([d], self._get_args)
            score = df[metric].dropna().mean() if metric in df.columns else float("nan")
            rows.append((str(Path(d).relative_to(self.root)), score))
        df = pd.DataFrame(rows, columns=["config", metric])
        return df.sort_values(metric, ascending=ascending, na_position="last").head(k).reset_index(drop=True)

    # ---- cross-run PNG-metric grids ----

    def _plot_tpr_fpr_grid(self, configs, title="TPR/FPR", full=False):
        colors = {label: _config_color(i) for i, (label, _) in enumerate(configs)}
        auc_by_label = self._metric_by_label(configs, "roc_auc")

        def cell(ax, dfs, column, fraction_bin, x_bins):
            _tpr_fpr_axes(ax, full)
            ax.set_xlabel("FPR", fontsize=8)
            for label, _ in configs:
                d = dfs[label]
                if d.empty:
                    continue
                d = d[_is_true(d["in_panel_b"]) & (d["column"] == column) & (d["fraction_bin"] == fraction_bin)]
                points = []
                for x_bin in x_bins:
                    dd = d[d["x_bin"] == x_bin]
                    if dd.empty:
                        continue
                    points.append((x_bin, dd["fpr"].mean(), dd["tpr"].mean()))
                if not points:
                    continue
                xs = [p[1] for p in points]
                ys = [p[2] for p in points]
                if len(xs) > 1:
                    ax.plot(xs, ys, color=colors[label], alpha=0.45, linewidth=1.0, zorder=2)
                shades = [_shade_color(colors[label], x_bins.index(p[0]) / max(len(x_bins) - 1, 1)) for p in points]
                ax.scatter(xs, ys, color=shades, s=34, edgecolor="white", linewidth=0.7, zorder=3)

        fig = _cross_run_grid(configs, title, cell, categorical_x=False)
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                        markerfacecolor=colors[label], markeredgecolor="white",
                        label=f"{label} (ROC-AUC={auc_by_label[label]:.4f})")
            for label, _ in configs
        ]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.97), fontsize=8, framealpha=0.9)

        # One shared shading legend for the whole figure instead of one per cell --
        # BRANCH_LENGTH_BINS is what 4 of 5 FIGURE_COLUMNS cells actually shade by
        # (the admixture column's 2 divergence-time bins reuse the same light->dark
        # scale, just with fewer steps).
        shade_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                        markerfacecolor=_shade_color("#808080", i / max(len(BRANCH_LENGTH_BINS) - 1, 1)),
                        markeredgecolor="white", markeredgewidth=0.6, label=x_bin)
            for i, x_bin in enumerate(BRANCH_LENGTH_BINS)
        ]
        fig.legend(
            handles=shade_handles, loc="upper left", bbox_to_anchor=(0.005, 0.97), fontsize=7,
            framealpha=0.85, title="shade = bin (light→dark)", title_fontsize=7,
        )
        return fig

    def _axis_title(self, configs, idx, value, skip=None):
        names = self._agg_axis_names.get(configs[0][0], []) if configs else []
        names = [n for i, n in enumerate(names) if i != skip]
        return _row_axis_title(names[idx] if idx < len(names) else None, value)

    def _axis_values(self, configs, idx):
        rank = {}
        for label, _ in configs:
            vals, ranks = self._config_axes.get(label, []), self._config_rank.get(label, ())
            v = vals[idx] if idx < len(vals) else None
            r = ranks[idx] if idx < len(ranks) else (float("inf"), float("inf"))
            rank[v] = min(rank.get(v, r), r)
        return sorted(rank, key=lambda v: (rank[v], _token_sort_key(v)))

    def _plot_tpr_fpr_grouped(self, configs, title, agg_idx=0, full=False):
        """Multi-level agg (2-4 axes, see resolve_configs_cartesian/
        _plot_grouped) version of _plot_tpr_fpr_grid: agg[0] -> one line per
        value (colored, shared legend, different segments), agg[1] -> one
        point per value on each line (shaded light->dark, points along the
        same segment), agg[2] -> subplot columns, agg[3] -> subplot rows.
        Each point is the (mean FPR, mean TPR) of that (row, col, point,
        line) combo's already-row-filtered rows (in_panel_b only). The
        legend's ROC-AUC per line value is pooled across every *real*
        (flag-based) point/col/row value sharing that line value (agg[0]),
        not just one cell -- invariant to any row-filter agg axis, same as
        _plot_grouped's legend -- see _metric_by_axis_value.

        agg_idx: which agg level drives point color + the legend (same as
        _plot_grouped's). agg_idx == 1 swaps agg[0]/agg[1]: one line per
        agg[1] value, points along it per agg[0] value (shaded). For
        agg_idx >= 2 the layout stays fixed and a line is drawn gray when its
        points span more than one agg[agg_idx] value."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        line_idx, point_idx = (1, 0) if agg_idx == 1 else (0, 1)
        line_vals = self._axis_values(configs, line_idx)
        point_vals = self._axis_values(configs, point_idx)
        col_vals = self._axis_values(configs, 2)
        row_vals = self._axis_values(configs, 3)

        by_key = {}
        for label, dirs in configs:
            key = (axis_val(label, 3), axis_val(label, 2), axis_val(label, point_idx), axis_val(label, line_idx))
            by_key[key] = (label, dirs)

        leg_vals = self._axis_values(configs, agg_idx)
        colors = {v: _config_color(i) for i, v in enumerate(leg_vals)}
        shade = True
        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)
        n_rows, n_cols = len(row_vals), len(col_vals)

        # Reserve enough top margin for however many legend rows agg[0]/agg[1]
        # actually produce -- a fixed-height reservation (the old behavior)
        # only fit whatever entry count someone tested with; a longer -w/-s
        # sweep or an extra (--ap, --annealing) combo pushed the legend past
        # the axes into the suptitle. This is only a generous upper-bound
        # estimate to size the canvas -- actual placement below re-measures
        # each legend's real rendered extent instead of trusting this
        # constant, so an under/over estimate here costs at most a
        # slightly bigger-than-necessary canvas, never an overlap.
        show_line_legend = len(leg_vals) > 1 or leg_vals != [None]
        show_shade_legend = shade and len(point_vals) > 1
        legend_rows = (
            (len(leg_vals) if show_line_legend else 0)
            + (len(point_vals) + 1 if show_shade_legend else 0)  # +1: shade legend's own title row
        )
        legend_extra_in = 0.2 * legend_rows

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8 + legend_extra_in), squeeze=False)
        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                ax = axes[ri][ci]
                _tpr_fpr_axes(ax, full)
                ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                ax.set_axisbelow(True)
                if ri == n_rows - 1:
                    ax.set_xlabel("FPR", fontsize=8)
                if ri == 0 and cv not in (None, "(none)"):
                    ax.set_title(self._axis_title(configs, 2, cv), fontsize=9.5, pad=8)
                if ci == 0 and rv not in (None, "(none)"):
                    ax.set_ylabel(self._axis_title(configs, 3, rv), fontsize=8.5)
                for lv in line_vals:
                    points = []
                    for pv in point_vals:
                        entry = by_key.get((rv, cv, pv, lv))
                        if entry is None:
                            continue
                        label, _ = entry
                        d = dfs[label]
                        if d.empty:
                            continue
                        d = d[_is_true(d["in_panel_b"])]
                        if d.empty:
                            continue
                        points.append((pv, d["fpr"].mean(), d["tpr"].mean(), colors[axis_val(label, agg_idx)]))
                    if not points:
                        continue
                    xs = [p[1] for p in points]
                    ys = [p[2] for p in points]
                    if len(xs) > 1:
                        cs = {p[3] for p in points}
                        ax.plot(xs, ys, color=cs.pop() if len(cs) == 1 else "#888888", alpha=0.45, linewidth=1.0, zorder=2)
                    shades = [_shade_color(p[3], point_vals.index(p[0]) / max(len(point_vals) - 1, 1)) if shade and len(point_vals) > 1 else p[3]
                              for p in points]
                    ax.scatter(xs, ys, color=shades, s=34, edgecolor="white", linewidth=0.7, zorder=3)
        _suptitle(fig, title, fontsize=12, y=0.995)

        # The two legends used to sit in opposite top corners (upper-left/
        # upper-right) sharing one row of vertical space -- fine for a wide
        # multi-column grid, but a single-panel figure (the common case here,
        # e.g. agg's col/row axes both empty) is only ~3.3in wide, too narrow
        # for two corner legends with any real text in them to avoid
        # colliding in the middle. Stack them instead (same corner, one atop
        # the other): that only costs vertical space, which is cheap since
        # it's already being reserved via legend_extra_in, and it works
        # regardless of figure width. Each legend's actual bottom is measured
        # (get_window_extent) rather than estimated, so the next legend --
        # and the axes area below both -- is placed exactly, not guessed.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_h_px = fig.bbox.height
        cursor_y = (fig._suptitle.get_window_extent(renderer).y0 - 4) / fig_h_px

        if show_line_legend:
            line_auc_values = self._metric_by_axis_value(configs, agg_idx, "roc_auc")
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                           markerfacecolor=colors[lv], markeredgecolor="white",
                           label=f"{lv} (ROC-AUC={line_auc_values[lv][0]:.4f}±{line_auc_values[lv][1]:.4f})")
                for lv in leg_vals
            ]
            line_legend = fig.legend(
                handles=handles, loc="upper right", bbox_to_anchor=(0.995, cursor_y), fontsize=8, framealpha=0.9)
            fig.canvas.draw()
            cursor_y = (line_legend.get_window_extent(renderer).y0 - 4) / fig_h_px

        # Shared shading legend showing agg[1]'s values (point axis), light->dark,
        # matching the shading drawn on every line's points -- same role as
        # _plot_tpr_fpr_grid's fixed x_bin shading legend, generalized to whatever
        # axis agg[1] actually is.
        if show_shade_legend:
            shade_handles = [
                plt.Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                           markerfacecolor=_shade_color("#808080", i / max(len(point_vals) - 1, 1)),
                           markeredgecolor="white", markeredgewidth=0.6, label=str(pv))
                for i, pv in enumerate(point_vals)
            ]
            shade_title = f"shade = agg[{point_idx}] (light→dark)"
            shade_legend = fig.legend(
                handles=shade_handles, loc="upper right", bbox_to_anchor=(0.995, cursor_y), fontsize=7,
                framealpha=0.85, title=shade_title, title_fontsize=7,
            )
            fig.canvas.draw()
            cursor_y = (shade_legend.get_window_extent(renderer).y0 - 4) / fig_h_px

        fig.tight_layout(rect=[0, 0.04, 1, _below_title_top(fig, cursor_y)])
        return fig

    def _plot_run(self, axes, path, metrics, exclude_keywords=(), logy=False, title="", dir="store"):
        df = _collect_run_df(axes, path, exclude_keywords, dir, self.root)
        panels = [(m,) if isinstance(m, str) else tuple(m) for m in metrics]
        missing = sorted({m for panel in panels for m in panel} - set(df.columns))
        if missing:
            raise ValueError(f"Unknown metric(s) {missing}; available: {[c for c in df.columns if c not in OUT_RUN_ID_COLUMNS]}")

        x_keys = sorted(df[["window", "step"]].drop_duplicates().itertuples(index=False, name=None))
        x_windows = sorted({w for w, _ in x_keys})
        x_labels = [format_val(w) for w in x_windows]
        has_steps = any(w != s for w, s in x_keys)
        configs = sorted(df["config"].unique())
        palette = sns.color_palette("tab10", max(len(configs), 1))
        markers = ["o", "s", "^", "D", "v", "P"]

        fig, axs = plt.subplots(1, len(panels), figsize=(max(4.2, 0.45 * len(x_windows) + 2.4) * len(panels), 3.8), squeeze=False)
        for ax, panel in zip(axs[0], panels):
            for ci, config in enumerate(configs):
                sub = df[df["config"] == config]
                xs = sub["window"].to_numpy(dtype=float)
                tiled = (sub["window"] == sub["step"]).to_numpy()
                order = np.argsort(xs[tiled])
                for mi, metric in enumerate(panel):
                    ys = sub[metric].to_numpy(dtype=float)
                    style = dict(color=palette[ci], marker=markers[mi % len(markers)], markersize=4)
                    ax.plot(xs[tiled][order], ys[tiled][order], linestyle=("-", "--", ":")[mi % 3], linewidth=1.2, **style)
                    ax.plot(xs[~tiled], ys[~tiled], linestyle="none", markerfacecolor="none", **style)
            ax.set_xscale("log")
            ax.set_xticks(x_windows)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
            ax.minorticks_off()
            ax.set_xlabel("window (hollow: step != window)" if has_steps else "window")
            ax.set_title(" vs ".join(panel), fontsize=9)
            if logy:
                ax.set_yscale("log")
            else:
                _apply_bounded_yaxis(ax, panel)
            ax.grid(alpha=0.25)

        handles = [plt.Line2D([], [], color=palette[ci], linewidth=1.5) for ci in range(len(configs))]
        labels = list(configs)
        if any(len(panel) > 1 for panel in panels):
            widest = max(panels, key=len)
            handles += [plt.Line2D([], [], color="0.3", marker=markers[mi % len(markers)], linestyle=("-", "--", ":")[mi % 3],
                                   markersize=4) for mi in range(len(widest))]
            labels += list(widest)
        _suptitle(fig, title or f"{_path_after_out(path)} ({_branch_length_label(df, path)})", fontsize=9)
        if len(handles) > 1:
            _legend_below_axes(fig, handles, labels, ncol=min(len(handles), 4), fontsize=7)
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig

    def _plot_branches(self, axes, runs, metrics, exclude_keywords=(), title="", dir="store"):
        dfs = {label: _collect_run_df(axes, path, exclude_keywords, dir, self.root) for label, path in runs.items()}
        columns = [c for m in metrics for c in (("tpr", "fpr") if m == "tpr_fpr" else (m,))]
        configs = sorted({c for df in dfs.values() for c in df.loc[df[columns].notna().any(axis=1), "config"]})
        configs, palette, legend_labels, legend_ncol = _rho_beta_palette(configs)
        fig, axs = plt.subplots(len(metrics), len(runs), figsize=(3.4 * len(runs), 3.2 * len(metrics)),
                                squeeze=False)
        for ci, (label, path) in enumerate(runs.items()):
            df = dfs[label]
            sized = df.dropna(subset=["window"])
            sizes = sorted({(int(w), int(s)) for w, s in zip(sized["window"], sized["step"])}) or [(None, None)]
            size_labels = ["" if w is None else format_val(w) if w == s else f"{format_val(w)}/{format_val(s)}"
                           for w, s in sizes]

            def rows_at(config, w, s):
                sub = df[df["config"] == config]
                return sub[sub["window"].isna() | ((sub["window"] == w) & (sub["step"] == s))]

            node_pattern = "/".join(Path(path).parts[-2:])
            axs[0][ci].set_title(f"{label}\n{node_pattern}\n{_branch_length_label(df, path)}", fontsize=9)
            for ri, metric in enumerate(metrics):
                ax = axs[ri][ci]
                ax.grid(alpha=0.25)
                if metric == "tpr_fpr":
                    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=0.8)
                    lines = []
                    for si, (w, s) in enumerate(sizes):
                        pts = []
                        for config in configs:
                            sub = rows_at(config, w, s)
                            if sub.empty:
                                continue
                            pts.append((sub["fpr"].mean(), sub["tpr"].mean()))
                            ax.scatter(*pts[-1], color=palette[config], s=36, zorder=3)
                        if len(pts) > 1:
                            lines.append((size_labels[si], pts))
                    window_colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(lines), 1)))
                    for (size_label, pts), color in zip(lines, window_colors):
                        ax.plot(*zip(*pts), color=color, linewidth=1.2, zorder=2, label=size_label)
                    if lines:
                        ax.legend(title="window", fontsize=6, title_fontsize=6, loc="lower right",
                                  ncol=2, framealpha=0.85)
                    ax.set_xlim(-0.02, 1.02)
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_xlabel("FPR")
                    ax.set_ylabel("TPR" if ci == 0 else "")
                    continue
                width = 0.8 / max(len(configs), 1)
                for k, config in enumerate(configs):
                    for si, (w, s) in enumerate(sizes):
                        sub = rows_at(config, w, s)
                        if sub.empty:
                            continue
                        ax.bar(si + (k - (len(configs) - 1) / 2) * width, sub[metric].mean(), width,
                               color=palette[config])
                ax.set_xticks(range(len(sizes)))
                ax.set_xticklabels(size_labels, fontsize=8)
                ax.set_ylabel(metric if ci == 0 else "")
                _apply_bounded_yaxis(ax, metric)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=palette.get(c, "none"), edgecolor="none") for c in legend_labels]
        _suptitle(fig, title or ", ".join(f"{k}={v}" for k, v in axes.items()), fontsize=10)
        _legend_below_axes(fig, handles, [c or "" for c in legend_labels], ncol=legend_ncol, fontsize=8)
        plt.show()
        return fig

    def _ecdf_entries(self, axes, path, exclude_keywords, dir):
        rel = Path(_path_after_out(path))
        if dir is None:
            dests = {_FLAG_TO_DEST.get(k): v for k, v in axes.items()}
            windows = axes.get("-w", axes.get("--window-size"))
            if windows is None:
                raise ValueError("ecdf needs a '-w' axis")
            variants = {d: bool(dests.get(d)) for d in _SCORES_VARIANT_DESTS if d != "chunk_size"}
            is_node = len(rel.parts) == 3
            node_rel = rel if is_node else rel.parent
            entries = []
            for w in windows if isinstance(windows, (list, tuple)) else [windows]:
                w = _typed_value("-w", w)
                node_dir = get_expected_scores_path(Path("simulations", *node_rel.parts, "concat", "_.fa"), node_rel,
                                                    window_size=w, step_size=w, **variants).parent.parent
                size = next(part for part in node_dir.parts if _OUT_SIZE_SEGMENT.match(part))
                patterns = sorted(d.name for d in node_dir.glob("*") if (d / "scores.tsv").exists()) if is_node else [rel.name]
                for pattern in patterns:
                    entries.append((size, (pattern,) if is_node else (), {"window_size": w, "step_size": w, **variants}, None,
                                    node_dir / pattern / "scores.tsv", pattern))
            return entries, pd.DataFrame({"branch_length": []})
        if dir == "store":
            rel = _path_below_source_root(path)
            df, recorded = collect_store_runs(rel, self.root, exclude_keywords)
            if df.empty:
                raise FileNotFoundError(f"No store report found for {path} under {self.root}")
            keep = _filter_out_runs(df.assign(_i=range(len(df))), axes, recorded)[0]["_i"]
            entries = []
            for i in keep:
                report, args = Path(df["report_path"][i]), recorded[i]
                leaf_parts = report.parts[:report.parts.index("reports")]
                size = next(part for part in leaf_parts if _OUT_SIZE_SEGMENT.match(part))
                fasta = Path("simulations", *rel.parts[:-1], "concat", f"{rel.name}.fa")
                scores_path = get_expected_scores_path(
                    fasta, Path(*rel.parts[:-1]), window_size=args.get("window_size"), step_size=args.get("step_size"),
                    **{k: args.get(k) or (None if k == "chunk_size" else False) for k in _SCORES_VARIANT_DESTS})
                entries.append((size, tuple(df["config"][i].split("/")), args, report, scores_path, rel.name))
            return entries, df
        df = _collect_run_df(axes, path, exclude_keywords, "out")
        entries = []
        for report in df["report_path"]:
            parts = Path(report).parent.parts
            size_idx = next(i for i, part in enumerate(parts) if _OUT_SIZE_SEGMENT.match(part))
            entries.append((parts[size_idx], parts[size_idx + 1:], _recorded_from_report_path(report), report, None,
                            parts[size_idx - 1]))
        return entries, df

    def _score_dist_cells(self, metric, axes, runs, agg, exclude_keywords, dir, pattern, by_em):
        assert "phlag" not in axes, f"{metric} doesn't take a 'phlag' source axis"
        assert dir in ("out", "store")
        rest = [a for a in agg if a not in ("em", "gt", "topology", "state")]
        if len(rest) > 1:
            raise ValueError(f"{metric} agg takes 'em' and at most one row flag, got {agg}")
        windows = axes.get("-w", axes.get("--window-size"))
        row_flag = rest[0] if rest else "-w" if isinstance(windows, (list, tuple)) and len(windows) > 1 else None
        source = dir if by_em else None
        row_dest = _FLAG_TO_DEST.get(row_flag)
        if row_flag is not None and row_dest is None:
            raise KeyError(f"{row_flag!r} is not one of benchmark.py's mirrored caster/phlag flags")

        figures, col_titles = {}, []
        for ci, (label, path) in enumerate(runs):
            entries, df = self._ecdf_entries(axes, path, exclude_keywords, source)
            rel = Path(_path_after_out(path))
            category = rel.parts[0] if rel.parts[0] == "admixture" else " ".join(rel.parts[:2])
            col_titles.append(label or f"{category}\n{rel.parts[2]} | {_branch_length_label(df, entries[0][4].parent if source is None and entries else path)}")
            for size, config, recorded, report, scores_path, entry_pattern in entries:
                if pattern is not None and str(entry_pattern) != str(pattern):
                    continue
                if row_dest is not None and row_dest not in recorded:
                    raise ValueError(f"{row_flag} isn't recorded for {report or scores_path}")
                m = _OUT_SIZE_SEGMENT.match(size)
                size_key = m.group(2) if row_dest in ("window_size", "step_size") else size
                tail = tuple(p for p in config if row_dest is None or row_dest not in (parse_phlag_param_segment(p) or {}))
                figures.setdefault((size_key, tail), {})[(recorded.get(row_dest), ci)] = (report, scores_path, entry_pattern)
        return row_flag, row_dest, figures, col_titles

    def _plot_score_dist(self, metric, axes, runs, agg=(), exclude_keywords=(), title="", dir="store", logx=None,
                         normalize=False, topologies=TOPOLOGY_NAMES, pattern=None, linthresh=None, legend=True):
        by_em = "em" in agg
        row_flag, row_dest, figures, col_titles = self._score_dist_cells(metric, axes, runs, agg, exclude_keywords, dir,
                                                                         pattern, by_em)
        panel, what, xlabel, ylabel, legend_items = _SCORE_DIST_METRICS[metric]
        topologies = [topologies] if isinstance(topologies, str) else list(topologies)
        figs = []
        for (size_key, tail), cells in figures.items():
            rows = sorted({v for v, _ in cells}, key=lambda v: (v != _RECORDED_DEFAULTS.get(row_dest),
                                                               not isinstance(v, (int, float)), v if isinstance(v, (int, float)) else str(v)))
            fig, axs = plt.subplots(len(rows), len(runs), figsize=(5 * len(runs), 4.0 * len(rows) + 1.0), squeeze=False)
            drawn = False
            for ri, value in enumerate(rows):
                for ci in range(len(runs)):
                    ax = axs[ri][ci]
                    cell = cells.get((value, ci))
                    try:
                        if cell is None:
                            raise ValueError("no matching report/scores")
                        if cell[0] is None:
                            scores, y_true = _scores_with_truth(*cell[1:])
                            y_pred = None
                        else:
                            scores, y_true, y_pred, _ = _report_window_labels(*cell)
                    except ValueError as e:
                        ax.text(0.5, 0.5, f"missing:\n{e}", transform=ax.transAxes, ha="center", va="center", fontsize=7, wrap=True)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        scores = None
                    if scores is not None:
                        drawn = True
                        if normalize:
                            scores = _minmax_normalize(scores, normalize)
                        panel(ax, scores, y_pred if by_em else y_true, topologies, normalize, logx)
                        if not legend:
                            for text in list(ax.texts):
                                text.remove()
                        if logx and ax.get_xscale() == "linear" and metric in ("ecdf", "fit"):
                            ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                        ax.grid(alpha=0.25)
                        ax.set_xlabel(f"normalized {xlabel}" if normalize and "score" in xlabel else xlabel)
                    if ri == 0:
                        ax.set_title(col_titles[ci], fontsize=10)
                    if ci == 0:
                        row_label = "" if row_flag is None else f"w{format_val(value)}\n" if row_dest == "window_size" \
                            else f"{_strip_flag(row_flag)} = {value}\n"
                        ax.set_ylabel(f"{row_label}{ylabel}")
            if not drawn:
                plt.close(fig)
                continue
            source = "EM state path" if by_em else "GT"
            suffix = "/".join(([size_key] if size_key else []) + list(tail))
            _suptitle(fig, title or f"{source} Null (dashed) vs Outlier (solid) {what}"
                      + (f" — {suffix}" if suffix else "") + (f" — {pattern}" if pattern is not None else ""), fontsize=10)
            items = legend_items(topologies)
            _legend_below_axes(fig, [h for h, _ in items], [l for _, l in items], ncol=min(len(items), 6), fontsize=8)
            figs.append(fig)
        if not figs:
            raise ValueError(f"No {metric} could be drawn for {[p for _, p in runs]} with {axes}")
        plt.show()
        return figs[0] if len(figs) == 1 else figs

    def _plot_ecdf_hmm(self, axes, runs, agg=(), exclude_keywords=(), title="", dir="store", logx=None, normalize=False,
                      topologies=TOPOLOGY_NAMES, pattern=None, linthresh=None, legend=True):
        row_flag, row_dest, figures, _ = self._score_dist_cells("ecdf_hmm", axes, runs, agg, exclude_keywords, dir, pattern, True)
        topologies = [topologies] if isinstance(topologies, str) else list(topologies)
        cols = [TOPOLOGY_NAMES.index(t) for t in topologies]
        figs = []
        for (size_key, tail), cells in figures.items():
            rows = sorted({v for v, _ in cells}, key=lambda v: (v != _RECORDED_DEFAULTS.get(row_dest),
                                                               not isinstance(v, (int, float)), v if isinstance(v, (int, float)) else str(v)))
            for ci, (label, path) in enumerate(runs):
                patterns = sorted({cell[2] for (v, c), cell in cells.items() if c == ci}, key=str)
                for entry_pattern in patterns:
                    fig, axs = plt.subplots(len(rows), len(cols), figsize=(3.8 * len(cols), 3.2 * len(rows) + 1.0), squeeze=False)
                    drawn, bl_label = False, None
                    for ri, value in enumerate(rows):
                        cell = cells.get((value, ci))
                        try:
                            if cell is None or cell[2] != entry_pattern:
                                raise ValueError("no matching report/scores")
                            scores, _, y_pred, _ = _report_window_labels(*cell)
                        except ValueError as e:
                            scores, err = None, e
                        if scores is not None and normalize:
                            scores = _minmax_normalize(scores, normalize)
                        if scores is not None and bl_label is None:
                            bl_label = _branch_length_label(pd.DataFrame({"branch_length": []}), Path(cell[1] or path).parent
                                                            if cell[1] else path)
                        for ci2, col in enumerate(cols):
                            ax = axs[ri][ci2]
                            if scores is None:
                                ax.text(0.5, 0.5, f"missing:\n{err}", transform=ax.transAxes, ha="center", va="center",
                                        fontsize=7, wrap=True)
                                ax.set_xticks([])
                                ax.set_yticks([])
                            else:
                                drawn = True
                                means = []
                                for state, color in enumerate(_HMM_STATE_COLORS):
                                    mean, _ = _draw_ecdf(ax, scores[y_pred == state, col], color=color, linewidth=1.6)
                                    means.append(mean)
                                    if mean is not None:
                                        ax.axvline(mean, color=color, linewidth=1.0, alpha=0.7)
                                if legend:
                                    _text_box(ax, ["Mean"] + [f"{name} HMM: {'n/a' if m is None else f'{m:.3g}'}"
                                                              for name, m in zip(("Null", "Outlier"), means)])
                                _score_xaxis(ax, scores, [topologies[ci2]], normalize, logx)
                                if logx and ax.get_xscale() == "linear":
                                    ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                                ax.grid(alpha=0.25)
                                if ri == len(rows) - 1:
                                    ax.set_xlabel("normalized topology score" if normalize else "topology score")
                            if ri == 0:
                                ax.set_title(topologies[ci2], fontsize=10)
                            if ci2 == 0:
                                row_label = "" if row_flag is None else f"w{format_val(value)}\n" if row_dest == "window_size" \
                                    else f"{_strip_flag(row_flag)} = {value}\n"
                                ax.set_ylabel(f"{row_label}proportion")
                    if not drawn:
                        plt.close(fig)
                        continue
                    rel = _path_after_out(path)
                    if not rel.endswith(f"/{entry_pattern}"):
                        rel = f"{rel}/{entry_pattern}"
                    head = label or f"{rel} ({bl_label})"
                    suffix = " — ".join(x for x in (_size_label(size_key), "/".join(tail)) if x)
                    _suptitle(fig, title or f"{head}" + (f" — {suffix}" if suffix else ""), fontsize=10)
                    _legend_below_axes(fig, [matplotlib.lines.Line2D([], [], color=c, linewidth=1.6) for c in _HMM_STATE_COLORS],
                                       ["Null HMM", "Outlier HMM"], ncol=2, fontsize=8)
                    figs.append(fig)
        if not figs:
            raise ValueError(f"No ecdf_hmm could be drawn for {[p for _, p in runs]} with {axes}")
        plt.show()
        return figs[0] if len(figs) == 1 else figs

    def _plot_ecdf_hmm_pooled(self, axes, agg=(), exclude_keywords=(), title="", topologies=TOPOLOGY_NAMES, pattern=None,
                              logx=None, linthresh=None, normalize=False, legend=True, windows=False):
        name = f"ecdf_hmm{'_all' if windows else ''}"
        agg = [agg] if isinstance(agg, str) else [a for a in agg or () if a]
        by_state = "state" in agg
        agg = [a for a in agg if a not in ("em", "topology", "state")]
        flags = [a for a in agg if not (isinstance(a, str) and a in _POOLED_SCORE_AXES)]
        by = [a for a in agg if a not in flags] or ["-w"]
        if len(by) > 1:
            raise ValueError(f"{name}: agg is at most one of {sorted(_POOLED_SCORE_AXES)} plus config flags, got {agg}")
        by = by[0]
        topologies = [topologies] if isinstance(topologies, str) else list(topologies)
        cols = [TOPOLOGY_NAMES.index(t) for t in topologies]
        series, n_reports = [], 0
        for w, _, _, leaf_label, kept in self._topology_scatter_sources(axes, exclude_keywords, pattern, [], True, "fitted_all", flags):
            n_reports += len(kept)
            groups = {}
            for r in kept:
                groups.setdefault(None if by == "-w" else _POOLED_SCORE_AXES[by](w, r[0]), []).append(r)
            for key in [None] if by == "-w" else _pooled_axis_values(by, set(groups)):
                null, alt, *_ = self._topology_scatter_points(groups[key], [np.nan] * len(groups[key]), "all" if windows else "means")
                series.append((_gt_mean_axis_label("-w", w) + leaf_label + ("" if key is None else f" {_gt_mean_axis_label(by, key)}"),
                               null, alt))
        if not series:
            raise ValueError(f"No {name} could be drawn with {axes}")
        what = "window mean" if windows else "per-run mean window mean"
        title = title or f"HMM (Viterbi) {what} ECDFs pooled over all runs ({'point = window' if windows else 'point = run'}, " \
            f"n reports = {n_reports})" + (f" — {pattern}" if pattern is not None else "")
        symlog = (not normalize) if logx is None else logx
        if by_state:
            fig = _hmm_state_ecdf_figure(series, cols, topologies, what, title, symlog, linthresh, normalize, legend)
        else:
            fig = _window_ecdf_figure(series, cols, topologies, what, title, symlog=symlog, linthresh=linthresh, normalize=normalize,
                                      legend=legend, cmap="Greys" if by in _ORDERED_POOLED_AXES and not flags else "tab10",
                                      state_labels=("Null HMM", "Outlier HMM"))
        plt.show()
        return fig

    def _plot_score_dist_pooled(self, metric, axes, agg=(), exclude_keywords=(), title="", logx=None, normalize=False,
                                topologies=TOPOLOGY_NAMES, pattern=None, linthresh=None, legend=True):
        agg = [a for a in agg if a and a != "gt"] or ["topology"]
        if agg[0] != "topology":
            raise ValueError(f"{metric} without run= overlays topologies: agg is ['topology', cols, rows], got {agg}")
        unknown = [a for a in agg[1:] if a not in _POOLED_SCORE_AXES]
        if unknown or len(agg) > 3:
            raise ValueError(f"{metric} without run= takes agg ['topology', cols, rows] from {sorted(_POOLED_SCORE_AXES)}, got {agg}")
        windows = self._caster_window_roots(axes)
        if "-w" not in agg and len(windows) > 1:
            if len(agg) == 3:
                raise ValueError(f"{metric}: several -w values but no agg slot left for '-w'")
            agg = agg + ["-w"]
        col_axis, row_axis = (agg[1:] + [None, None])[:2]
        panel, what, xlabel, ylabel, legend_items = _SCORE_DIST_METRICS[metric]
        topologies = [topologies] if isinstance(topologies, str) else list(topologies)
        cells = {}
        for w, root in windows:
            for parts, null, alt in _caster_scores_by_run(root, tuple(exclude_keywords), pattern):
                key = tuple(None if a is None else _POOLED_SCORE_AXES[a](w, parts) for a in (col_axis, row_axis))
                cell = cells.setdefault(key, ([], [], set()))
                cell[0].append(null)
                cell[1].append(alt)
                cell[2].add(parts)
        if not cells:
            raise ValueError(f"No {metric} could be drawn: no caster scores for {axes}")
        col_vals = _pooled_axis_values(col_axis, {k[0] for k in cells})
        row_vals = _pooled_axis_values(row_axis, {k[1] for k in cells})
        fig, axs = plt.subplots(len(row_vals), len(col_vals), figsize=(max(9, 5 * len(col_vals)), 4.0 * len(row_vals) + 1.0),
                                squeeze=False)
        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                ax = axs[ri][ci]
                cell = cells.get((cv, rv))
                if cell is None:
                    ax.set_visible(False)
                    continue
                null, alt = np.concatenate(cell[0]), np.concatenate(cell[1])
                scores = np.concatenate([null, alt])
                labels = np.r_[np.zeros(len(null), dtype=int), np.ones(len(alt), dtype=int)]
                if normalize:
                    scores = _minmax_normalize(scores, normalize)
                panel(ax, scores, labels, topologies, normalize, logx)
                if not legend:
                    for text in list(ax.texts):
                        text.remove()
                if logx and ax.get_xscale() == "linear" and metric in ("ecdf", "fit"):
                    ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                ax.grid(alpha=0.25)
                ax.set_xlabel(f"normalized {xlabel}" if normalize and "score" in xlabel else xlabel)
                head = " ".join(_gt_mean_axis_label(a, v) for a, v in ((col_axis, cv), (row_axis, rv)) if a is not None)
                ax.set_title(f"{head} (n runs = {len(cell[2])})".strip(), fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f"{_gt_mean_axis_label(row_axis, rv)}\n{ylabel}" if row_axis else ylabel)
        _suptitle(fig, title or f"GT Null (dashed) vs Outlier (solid) {what}, pooled over all runs"
                  + (f" — {pattern}" if pattern is not None else ""), fontsize=10)
        items = legend_items(topologies)
        _legend_below_axes(fig, [h for h, _ in items], [l for _, l in items], ncol=min(len(items), 6), fontsize=8)
        plt.show()
        return fig

    def _plot_window_ecdf(self, axes, runs=None, exclude_keywords=(), title="", topologies=TOPOLOGY_NAMES, pattern=None,
                          logx=None, linthresh=None, normalize=False, legend=True, color=None, agg=()):
        if color is not None and runs is not None:
            raise ValueError("window_ecdf: color= needs the pooled form (no run=)")
        by = [a for a in agg or () if a] or ["-w"]
        if len(by) > 1 or by[0] not in _POOLED_SCORE_AXES or (by[0] != "-w" and runs is not None):
            raise ValueError(f"window_ecdf: agg is one of {sorted(_POOLED_SCORE_AXES)} (non '-w' only without run=), got {agg}")
        by = by[0]
        topologies = [topologies] if isinstance(topologies, str) else list(topologies)
        cols = [TOPOLOGY_NAMES.index(t) for t in topologies]
        figs = []
        if runs is None:
            per_w = []
            for w, root in self._caster_window_roots(axes):
                runs_w = _caster_scores_by_run(root, tuple(exclude_keywords), pattern)
                if runs_w:
                    per_w.append((w, runs_w, [np.nan] * len(runs_w) if color is None
                                  else self._run_color_values(color, w, root, runs_w, exclude_keywords)))
            spec = None
            if color is not None:
                spec = _color_spec([v for *_, vals in per_w for v in vals], color) if per_w else None
                if per_w and spec is None:
                    raise ValueError(f"window_ecdf: no {color!r} values found for any run")

            def build(label, members):
                rs = [r for r, _ in members]
                s = (label, np.concatenate([r[1] for r in rs]), np.concatenate([r[2] for r in rs]), len(rs))
                if spec is not None:
                    codes = _color_codes(spec, [v for _, v in members])
                    s += tuple(np.concatenate([np.full(len(r[i]), c) for r, c in zip(rs, codes)]) for i in (1, 2))
                return s

            if by == "-w":
                panels = [(None, [build(_gt_mean_axis_label("-w", w), list(zip(runs_w, vals))) for w, runs_w, vals in per_w])]
            else:
                panels = []
                for w, runs_w, vals in per_w:
                    groups = {}
                    for r, v in zip(runs_w, vals):
                        groups.setdefault(_POOLED_SCORE_AXES[by](w, r[0]), []).append((r, v))
                    panels.append((w, [build(_gt_mean_axis_label(by, k), groups[k]) for k in _pooled_axis_values(by, set(groups))]))
            for w, series in panels:
                if not series:
                    continue
                n_runs = series[0][3] if by == "-w" else sum(s[3] for s in series)
                figs.append(_window_ecdf_figure(
                    series, cols, topologies, "window mean",
                    title or f"GT window-mean ECDFs pooled over all runs by {'window size' if by == '-w' else by}"
                    + (f", w{format_val(w)}" if w is not None else "")
                    + (f", points colored by {_color_label(color)}" if spec is not None else "")
                    + f" (n runs = {n_runs})" + (f" — {pattern}" if pattern is not None else ""),
                    symlog=(not normalize) if logx is None else logx, linthresh=linthresh, normalize=normalize,
                    legend=legend, color_by=spec, cmap="Greys" if by in _ORDERED_POOLED_AXES else "tab10"))
        for label, path in runs or ():
            entries, df = self._ecdf_entries(axes, path, exclude_keywords, None)
            by_pattern = {}
            for _, tail, recorded, _, scores_path, entry_pattern in entries:
                if pattern is None or str(entry_pattern) == str(pattern):
                    by_pattern.setdefault(tail, []).append((recorded["window_size"], scores_path, entry_pattern))
            for tail, windows in by_pattern.items():
                series = []
                for w, scores_path, entry_pattern in sorted(windows, key=lambda w: w[0]):
                    try:
                        scores, y_true = _scores_with_truth(scores_path, entry_pattern)
                    except ValueError as e:
                        print(f"Skipping w{format_val(w)}: {e}")
                        continue
                    series.append((_gt_mean_axis_label("-w", w), scores[y_true == 0], scores[y_true == 1]))
                if not series:
                    continue
                rel = "/".join((_path_after_out(path),) + tuple(tail))
                figs.append(_window_ecdf_figure(series, cols, topologies, "window mean", title or
                                                f"{label or rel} ({_branch_length_label(df, windows[0][1].parent)}): GT window-mean ECDFs by window size",
                                                symlog=bool(logx), linthresh=linthresh, normalize=normalize, legend=legend))
        if not figs:
            raise ValueError(f"No window_ecdf could be drawn for {runs and [p for _, p in runs]} with {axes}")
        plt.show()
        return figs[0] if len(figs) == 1 else figs

    def _plot_topology_scatter(self, axes, runs=None, exclude_keywords=(), title="", topologies=TOPOLOGY_NAMES, pattern=None,
                               logx=None, linthresh=None, normalize=False, legend=True, state=None, topology=None, stat="all",
                               color=None, corr="spearman", percentile=None, agg=()):
        agg = [agg] if isinstance(agg, str) else [a for a in agg or () if a]
        flags = [a for a in agg if not (isinstance(a, str) and a in _POOLED_SCORE_AXES)]
        by = [a for a in agg if a not in flags] or ["-w"]
        fitted = stat.startswith("fitted_") or stat in _VITERBI_SPLIT_STATS
        if len(by) > 1 or (flags and not fitted):
            raise ValueError(f"topology_topology_{stat}: agg is at most one of {sorted(_POOLED_SCORE_AXES)} plus standard "
                             f"config agg (flags/tuples, fitted_*/viterbi_* stats only), got {agg}")
        by = by[0]
        if corr not in ("spearman", "pearson"):
            raise ValueError(f"topology_topology_{stat}: corr is 'spearman' or 'pearson', got {corr!r}")
        symbol = "ρ" if corr == "spearman" else "r"
        states = [s for s in _STATE_STYLES if state is None or s[0] in ([state] if isinstance(state, str) else state)]
        picked = TOPOLOGY_NAMES if topology is None else [topology] if isinstance(topology, str) else list(topology)
        if not states or set(picked) - set(TOPOLOGY_NAMES):
            raise ValueError(f"topology_topology_{stat}: state from {[s[0] for s in _STATE_STYLES]}, topology from {TOPOLOGY_NAMES}")
        names = _COVARIANCE_PAIR_NAMES if stat.endswith("covariances") else TOPOLOGY_NAMES
        if names is _COVARIANCE_PAIR_NAMES:
            picked = [n for n, (a, b) in zip(names, _COVARIANCE_PAIRS) if TOPOLOGY_NAMES[a] in picked or TOPOLOGY_NAMES[b] in picked]
        prefixes = [Path(p).parts for p in runs or ()]
        series, n_runs = [], 0
        for w, root, leaf, leaf_label, kept in self._topology_scatter_sources(axes, exclude_keywords, pattern, prefixes, fitted, stat, flags):
            n_runs = max(n_runs, len(kept))
            all_values = self._run_color_values(color, w, root, [(r[0], None, None) for r in kept] if fitted else kept,
                                                exclude_keywords, leaf) if color is not None else [np.nan] * len(kept)
            groups = {}
            for r, v in zip(kept, all_values):
                groups.setdefault(None if by == "-w" else _POOLED_SCORE_AXES[by](w, r[0]), []).append((r, v))
            for key in _pooled_axis_values(by, set(groups)) if by != "-w" else [None]:
                label = _gt_mean_axis_label("-w", w) + leaf_label + ("" if key is None else f" {_gt_mean_axis_label(by, key)}")
                series.append((label, *self._topology_scatter_points([r for r, _ in groups[key]], [v for _, v in groups[key]], stat)))
        if not series:
            raise ValueError(f"No topology_topology_{stat} could be drawn for {runs} with {axes}")
        spec = None
        if color is not None:
            spec = _color_spec([v for s in series for v in s[4][0]], color)
            if spec is None:
                raise ValueError(f"topology_topology_{stat}: no {color!r} values found for any run")
        pairs = [(a, b) for a, b in itertools.combinations(range(len(names)), 2)
                 if (names[a] in picked) + (names[b] in picked) >= min(len(picked), 2)]
        fig, axs = plt.subplots(len(pairs), len(series), figsize=(3.8 * len(series) + (1.2 if spec else 0), 3.5 * len(pairs)),
                                squeeze=False)
        rng = np.random.default_rng(0)
        what = {"all": "window mean", "means": "mean window mean", "variances": "window-mean variance",
                "covariances": "window-mean covariance", "fitted_all": "window mean", "viterbi_variances": "window-mean variance", "fitted_means": "fitted mean", "fitted_variances": "fitted variance",
                "fitted_covariances": "fitted covariance"}[stat]
        markers = ("o", "^") if spec else ("o", "o")
        for ci, (w, null, alt, _, run_values) in enumerate(series):
            if normalize:
                both = np.concatenate([null, alt])
                both = np.column_stack([_minmax_normalize(both[:, c], normalize) for c in range(both.shape[1])])
                null, alt = both[:len(null)], both[len(null):]
            bounds = np.tile([-np.inf, np.inf], (null.shape[1], 1))
            if percentile is not None:
                pooled = np.concatenate([(null, alt)[si] for _, si, _ in states])
                for c in range(pooled.shape[1]):
                    if np.isfinite(pooled[:, c]).any():
                        bounds[c] = np.nanpercentile(pooled[:, c], percentile)
            codes = [_color_codes(spec, v) if spec else None for v in run_values]
            for ri, (a, b) in enumerate(pairs):
                ax = axs[ri][ci]
                lines, kept = [], []
                for name, si, _ in states:
                    xy = (null, alt)[si][:, [a, b]]
                    finite = np.flatnonzero(np.isfinite(xy).all(axis=1) & (xy >= bounds[[a, b], 0]).all(axis=1)
                                            & (xy <= bounds[[a, b], 1]).all(axis=1))
                    kept.append(xy[finite])
                    shown = rng.choice(finite, _SCATTER_MAX_POINTS, replace=False) if len(finite) > _SCATTER_MAX_POINTS else finite
                    point_colors = [_color_of(spec, c) for c in codes[si][shown]] if spec else _HMM_STATE_COLORS[si]
                    ax.scatter(xy[shown, 0], xy[shown, 1], s=2 if stat.endswith("all") else 8, alpha=0.25 if stat.endswith("all") else 0.5,
                               c=point_colors, marker=markers[si], linewidths=0, rasterized=True)
                    lines.append(f"{name} {symbol}={_corr(xy[finite], corr):.3f} (n={len(finite)})")
                if len(states) > 1:
                    lines.append(f"all {symbol}={_corr(np.concatenate(kept), corr):.3f}")
                ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5))
                if logx:
                    ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                    ax.set_yscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                ax.grid(alpha=0.25)
                ax.set_title(f"{w}: {names[a]} vs {names[b]}", fontsize=9)
                ax.set_xlabel(f"{'normalized ' if normalize else ''}{names[a]} {what}")
                if ci == 0:
                    ax.set_ylabel(f"{'normalized ' if normalize else ''}{names[b]} {what}")
        scope = ", ".join("/".join(p) for p in prefixes) or "all runs"
        _suptitle(fig, title or {"all": "GT window-mean", "means": "GT per-run mean window-mean (point = run)",
                                 "variances": "GT per-run window-mean variance (point = run)",
                                 "covariances": "GT per-run window-mean covariance (point = run)",
                                 "fitted_all": "HMM (Viterbi) window-mean",
                                 "viterbi_variances": "HMM (Viterbi) per-run window-mean variance (point = run)",
                                 "fitted_means": "HMM-fitted emission mean (point = run)",
                                 "fitted_variances": "HMM-fitted emission variance (point = run)",
                                 "fitted_covariances": "HMM-fitted emission covariance (point = run)"}[stat]
                  + f" topology vs topology, {scope} (n runs = {n_runs}, {corr.capitalize()} {symbol}"
                  + (f", x/y = p{format_val(percentile[0])}–p{format_val(percentile[1])}" if percentile is not None else "") + ")"
                  + (f", colored by {_color_label(color)}" if spec else "")
                  + (f" — {pattern}" if pattern is not None else ""), fontsize=10)
        right = 1 - 1.1 / fig.get_figwidth() if spec else 1.0
        if legend:
            _legend_below_axes(fig, [matplotlib.lines.Line2D([], [], marker=markers[si], linestyle="",
                                                             color="gray" if spec else _HMM_STATE_COLORS[si]) for _, si, _ in states],
                               [name for name, _, _ in states], ncol=len(states), fontsize=8, right=right)
        else:
            fig.tight_layout(rect=[0, 0, right, 0.95])
        if spec and spec["cats"] is None:
            top, bottom = axs[0][-1].get_position().y1, axs[-1][-1].get_position().y0
            cax = fig.add_axes([right + 0.15 / fig.get_figwidth(), bottom, 0.15 / fig.get_figwidth(), top - bottom])
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=spec["norm"], cmap=spec["cmap"]), cax=cax,
                         label="branch length (CU)" if spec["label"] == "bl" else spec["label"])
        elif spec:
            fig.legend([matplotlib.lines.Line2D([], [], color=_color_of(spec, i), linewidth=2) for i in range(len(spec["cats"]))],
                       spec["cats"], title=spec["label"], loc="center left", bbox_to_anchor=(right, 0.5), fontsize=7, title_fontsize=8)
        plt.show()
        return fig

    def _topology_scatter_sources(self, axes, exclude_keywords, pattern, prefixes, fitted, stat, flags=()):
        roots = self._caster_window_roots(axes)
        keep = lambda parts: not prefixes or any(parts[:len(p)] == p for p in prefixes)
        if not fitted:
            for w, root in roots:
                kept = [r for r in _caster_scores_by_run(root, tuple(exclude_keywords), pattern) if keep(r[0])]
                if kept:
                    yield w, root, None, "", kept
            return
        configs = self.resolve_configs_cartesian(axes, exclude_keywords=exclude_keywords, agg=list(flags))
        if not flags:
            configs = [("", [d for _, ds in configs for d in ds])]
        for w, root in roots:
            for label, dirs in configs:
                at_w = sorted({Path(d) for d in dirs if self._get_args(d).get("window_size") == w})
                for source in self._topology_scatter_leaves(w, root, at_w, keep, pattern, exclude_keywords, stat) if at_w else ():
                    yield (*source[:3], (f" {label}" if label else "") + source[3], source[4])

    def _topology_scatter_leaves(self, w, root, at_w, keep, pattern, exclude_keywords, stat):
        depth = min(len(leaf.parts) for leaf in at_w)
        skipped = [leaf for leaf in at_w if len(leaf.parts) > depth]
        if skipped:
            print(f"w{format_val(w)}: skipping variant runs (pin their flag to plot them): {[str(s) for s in skipped]}")
        at_w = [leaf for leaf in at_w if len(leaf.parts) == depth]
        tails = [[p for p in leaf.parts[next(i for i, p in enumerate(leaf.parts) if _OUT_SIZE_SEGMENT.match(p)) + 1:]]
                 for leaf in at_w]
        common = os.path.commonprefix(tails) if len(tails) > 1 else tails[0]
        for leaf, tail in zip(at_w, tails):
            reports = [r for r in sorted((leaf / "reports").glob("*/*/*/*.tsv"))
                       if keep(r.relative_to(leaf / "reports").with_suffix("").parts)
                       and (pattern is None or r.stem == str(pattern)) and not any(k in str(r) for k in exclude_keywords)]
            if stat in _VITERBI_SPLIT_STATS:
                args = self._get_args(leaf)
                scores_paths = [get_expected_scores_path(
                    Path("simulations", *parts[:3], "concat", f"{parts[3]}.fa"), Path(*parts[:3]),
                    window_size=args.get("window_size"), step_size=args.get("step_size"),
                    **{k: args.get(k) or (None if k == "chunk_size" else False) for k in _SCORES_VARIANT_DESTS})
                    for parts in (r.relative_to(leaf / "reports").with_suffix("").parts for r in reports)]
                loaded = _hmm_split_scores_many(reports, scores_paths)
            else:
                loaded = _fitted_moments_many(reports)
            kept = [(r.relative_to(leaf / "reports").with_suffix("").parts, *m) for r, m in zip(reports, loaded) if m is not None]
            if kept:
                yield w, root, leaf, "".join(f" {p}" for p in tail[len(common):]), kept

    @staticmethod
    def _topology_scatter_points(kept, values, stat):
        if stat in _VITERBI_SPLIT_STATS:
            stat = _VITERBI_SPLIT_STATS[stat]
        elif stat.startswith("fitted_"):
            pick = {"fitted_means": lambda mean, cov: mean, "fitted_variances": lambda mean, cov: np.diagonal(cov),
                    "fitted_covariances": lambda mean, cov: cov[tuple(np.array(_COVARIANCE_PAIRS).T)]}[stat]
            points = [np.array([pick(r[1][i], r[2][i]) for r in kept]) for i in (0, 1)]
            return (*points, len(kept), [values, values])
        if stat == "covariances":
            def offdiag(x):
                return np.cov(x[np.isfinite(x).all(axis=1)].T)[tuple(np.array(_COVARIANCE_PAIRS).T)] \
                    if np.isfinite(x).all(axis=1).sum() >= 2 else np.full(len(_COVARIANCE_PAIRS), np.nan)
            points = [np.array([offdiag(r[i]) for r in kept]) for i in (1, 2)]
            return (*points, len(kept), [values, values])
        if stat == "all":
            points = [np.concatenate([r[i] for r in kept]) for i in (1, 2)]
            run_values = [[v for r, v in zip(kept, values) for _ in range(len(r[i]))] for i in (1, 2)]
        else:
            reduce, min_n = (np.nanvar, 2) if stat == "variances" else (np.nanmean, 1)
            points = [np.array([reduce(r[i], axis=0) if len(r[i]) >= min_n else np.full(len(TOPOLOGY_NAMES), np.nan)
                                for r in kept]) for i in (1, 2)]
            run_values = [values, values]
        return (*points, len(kept), run_values)

    def _plot_ecdf_all(self, axes, runs=None, exclude_keywords=(), title="", topologies=TOPOLOGY_NAMES, pattern=None,
                       logx=None, linthresh=None, normalize=False, legend=True, percentile=(1, 99), vs_bl=False, color=None,
                       logy=False, fit=False):
        prefixes = [Path(p).parts for p in runs or ()]
        figs = []
        for w, root in self._caster_window_roots(axes):
            branches, exps = {}, {}
            for parts, null, alt in _caster_scores_by_run(root, tuple(exclude_keywords), pattern):
                if prefixes and not any(parts[:len(p)] == p for p in prefixes):
                    continue
                branch = branches.setdefault(parts[:3], ([], []))
                branch[0].append(null)
                branch[1].append(alt)
                exps.setdefault(parts[:3], []).append((parts, null, alt))
            if not branches:
                continue
            bls = {k: _sim_cu_branch_length(str(root / Path(*k))) for k in branches}
            colors = self._branch_colors(color, w, root, exps, bls, exclude_keywords)
            scope = ", ".join("/".join(p) for p in prefixes) or "all runs"
            figs.append((functools.partial(_mean_vs_bl_all_figure, logy=logy) if vs_bl else functools.partial(_ecdf_all_figure, fit=fit))(
                branches, bls, topologies, ("Null", "Outlier"), normalize, logx, linthresh, legend, percentile,
                lambda n: title or f"w{format_val(w)} GT {'per-branch window-mean mean vs CU branch length' if vs_bl else 'window-mean ECDFs per branch'}, {scope} (n experiments = {n}, "
                f"branches = {len(branches)}, x = p{format_val(percentile[0])}–p{format_val(percentile[1])})"
                + (f" — {pattern}" if pattern is not None else ""), None if vs_bl and color in (None, "bl") else colors))
        if not figs:
            raise ValueError(f"No {'mean_all_vs_bl' if vs_bl else 'ecdf_all'} could be drawn for {runs} with {axes}")
        plt.show()
        return figs[0] if len(figs) == 1 else figs

    def _plot_ecdf_all_hmm(self, axes, runs=None, exclude_keywords=(), title="", topologies=TOPOLOGY_NAMES, pattern=None,
                          logx=None, linthresh=None, normalize=False, legend=True, percentile=(1, 99), color=None, fit=False):
        prefixes = [Path(p).parts for p in runs or ()]
        dirs = sorted({d for _, ds in self.resolve_configs_cartesian(axes, exclude_keywords=exclude_keywords) for d in ds})
        figs = []
        for leaf in dirs:
            leaf = Path(leaf)
            args = self._get_args(leaf)
            branches, bls, exps, skipped = {}, {}, {}, 0
            for report in sorted((leaf / "reports").glob("*/*/*/*.tsv")):
                parts = report.relative_to(leaf / "reports").with_suffix("").parts
                if prefixes and not any(parts[:len(p)] == p for p in prefixes):
                    continue
                if (pattern is not None and parts[3] != str(pattern)) or any(k in str(report) for k in exclude_keywords):
                    continue
                scores_path = get_expected_scores_path(
                    Path("simulations", *parts[:3], "concat", f"{parts[3]}.fa"), Path(*parts[:3]),
                    window_size=args.get("window_size"), step_size=args.get("step_size"),
                    **{k: args.get(k) or (None if k == "chunk_size" else False) for k in _SCORES_VARIANT_DESTS})
                try:
                    scores, _, y_pred, _ = _report_window_labels(report, scores_path, parts[3])
                except ValueError:
                    skipped += 1
                    continue
                branch = branches.setdefault(parts[:3], ([], []))
                branch[0].append(scores[y_pred == 0])
                branch[1].append(scores[y_pred == 1])
                exps.setdefault(parts[:3], []).append((parts, None, None))
                bls.setdefault(parts[:3], _sim_cu_branch_length(str(scores_path.parent.parent)))
            if skipped:
                print(f"{leaf}: skipped {skipped} reports without a usable state path/scores")
            if not branches:
                continue
            scope = ", ".join("/".join(p) for p in prefixes) or "all runs"
            config = str(leaf.relative_to(self.root)) if leaf.is_relative_to(self.root) else str(leaf)
            figs.append(_ecdf_all_figure(
                branches, bls, topologies, ("Null HMM", "Outlier HMM"), normalize, logx, linthresh, legend, percentile,
                lambda n: title or f"{_size_label(config)} HMM (Viterbi) window-mean ECDFs per branch, {scope} (n experiments = {n}, "
                f"branches = {len(branches)}, x = p{format_val(percentile[0])}–p{format_val(percentile[1])})"
                + (f" — {pattern}" if pattern is not None else ""),
                self._branch_colors(color, args.get("window_size"), None, exps, bls, exclude_keywords, leaf), fit=fit))
        if not figs:
            raise ValueError(f"No ecdf_all_hmm could be drawn for {runs} with {axes}")
        plt.show()
        return figs[0] if len(figs) == 1 else figs

    def _plot_gt_mean_ecdf(self, metric, axes, agg=(), pattern=None, exclude_keywords=(), title="",
                           topologies=TOPOLOGY_NAMES, logx=None, linthresh=None, normalize=False, legend=True,
                           per_window=False):
        logx = (not normalize) if logx is None else logx
        linthresh = linthresh or 1e-2
        topologies = [topologies] if isinstance(topologies, str) else list(topologies)
        if per_window:
            long = self._gt_window_gaps(axes, topologies, exclude_keywords, pattern, normalize)
        else:
            dirs = [d for _, ds in self.resolve_configs_cartesian(axes, exclude_keywords=exclude_keywords) for d in ds]
            df = _load_raw_config_df(dirs, self._get_args)
            if df.empty:
                raise ValueError(f"No runs.tsv found for {axes}")
            if pattern is not None:
                df = df[df["pattern"].astype(str) == str(pattern)]
            df = df.drop_duplicates(["run_id", "_window_size"]).copy()
            df["_cu"] = _cu_branch_lengths(df)
            df["branch_length"] = df["_cu"].apply(lambda v: None if pd.isna(v) else assign_bin(v, BRANCH_LENGTH_BIN_EDGES))
            parts = []
            for t in topologies:
                null_alt = df[[f"null_mean_{t}", f"alt_mean_{t}"]].to_numpy(dtype=float)
                if normalize:
                    null_alt = _minmax_normalize(null_alt, normalize)
                parts.append(df.assign(topology=t, _null=null_alt[:, 0], _alt=null_alt[:, 1]))
            long = pd.concat(parts, ignore_index=True)
            long["_gap"] = long["_alt"] - long["_null"]

        scatter = metric.endswith("_vs_bl")
        value = "_gap" if metric.startswith("gap") else None
        agg = [a for a in agg if a] or (["topology", "branch_length"] if metric == "mean_ecdf" else ["topology", "merged_category"])
        if len(agg) > 3:
            raise ValueError(f"{metric} agg is [curves, cols, rows], got {agg}")
        if "-w" not in agg and long["_window_size"].nunique() > 1:
            if len(agg) == 3:
                raise ValueError(f"{metric}: several -w values but no agg slot left for '-w'")
            agg = agg + ["-w"]
        cols = [{"-w": "_window_size"}.get(a, a) for a in agg]
        for a, c in zip(agg, cols):
            if c not in long.columns:
                raise KeyError(f"{metric}: unknown agg axis {a!r}")
        long = long.dropna(subset=cols + (["_cu"] if scatter else []))
        curve_col, col_col, row_col = (cols + [None, None])[:3]
        curve_vals = _gt_mean_axis_values(long, curve_col)
        col_vals = _gt_mean_axis_values(long, col_col) if col_col else [None]
        row_vals = _gt_mean_axis_values(long, row_col) if row_col else [None]
        colors = {v: _ECDF_TOPO_COLORS[v] if curve_col == "topology" else f"C{i % 10}" for i, v in enumerate(curve_vals)}

        fig, axs = plt.subplots(len(row_vals), len(col_vals), figsize=(4 * len(col_vals), 3.2 * len(row_vals) + 1.0),
                                sharey=True, squeeze=False)
        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                ax = axs[ri][ci]
                mask = pd.Series(True, index=long.index)
                if col_col:
                    mask &= long[col_col] == cv
                if row_col:
                    mask &= long[row_col] == rv
                cell = long[mask]
                entries = []
                for v in curve_vals:
                    sub = cell[cell[curve_col] == v]
                    if not len(sub):
                        continue
                    label = _gt_mean_axis_label(agg[0], v)
                    entries.append((plt.Line2D([], [], color=colors[v], linewidth=1.6),
                                    f"{label}: μ={sub['_gap'].mean():+.3g}" if value else
                                    f"{label}: N {sub['_null'].mean():.3g} / O {sub['_alt'].mean():.3g}"))
                    if scatter:
                        for _, state, _ in _STATE_STYLES if value is None else [(None, None, None)]:
                            y = sub[value or ("_alt" if state else "_null")]
                            ax.scatter(sub["_cu"], y, s=18, linewidths=1, edgecolors=colors[v],
                                       facecolors="none" if state == 0 else colors[v], alpha=0.8)
                    elif metric == "gap_ecdf":
                        mean, _ = _draw_ecdf(ax, sub["_gap"].to_numpy(dtype=float), color=colors[v], linewidth=1.6)
                        if mean is not None:
                            ax.axvline(mean, color=colors[v], linestyle="--", linewidth=1)
                    else:
                        for _, state, linestyle in _STATE_STYLES:
                            _draw_ecdf(ax, sub["_alt" if state else "_null"].to_numpy(dtype=float), color=colors[v],
                                       linestyle=linestyle, linewidth=1.6)
                if scatter:
                    ax.set_xscale("log")
                    if value:
                        ax.axhline(0, color="gray", linewidth=0.5)
                    if logx:
                        ax.set_yscale("symlog", linthresh=linthresh)
                elif metric == "gap_ecdf":
                    ax.axvline(0, color="gray", linewidth=0.5)
                if logx and not scatter:
                    ax.set_xscale("symlog", linthresh=linthresh)
                ax.grid(alpha=0.25)
                if legend and entries:
                    ax.legend([h for h, _ in entries], [l for _, l in entries], fontsize=6, loc="best")
                if ri == 0:
                    ax.set_title(f"{_gt_mean_axis_label(agg[1], cv) if col_col else ''} (n={cell['run_id'].nunique()})", fontsize=10)
                quantity = ("normalized " if normalize else "") + (("Outlier window − Null GT mean" if per_window else
                                                                     "Outlier − Null GT mean") if value else "GT mean")
                if ri == len(row_vals) - 1:
                    ax.set_xlabel("branch length (CU)" if scatter else quantity)
                if ci == 0:
                    ylabel = quantity if scatter else "proportion"
                    ax.set_ylabel(f"{_gt_mean_axis_label(agg[2], rv)}\n{ylabel}" if row_col else ylabel)
        what = {"gap_ecdf": "Outlier window − run Null mean gap ECDFs (dashed = mean)" if per_window else
                "Outlier − Null mean gap ECDFs (dashed = mean)",
                "mean_ecdf": "Null (dashed) vs Outlier (solid) mean ECDFs",
                "gap_vs_bl": "Outlier − Null mean gap vs CU branch length",
                "mean_vs_bl": "Null (hollow) vs Outlier (filled) mean vs CU branch length"}[metric]
        _suptitle(fig, title or f"GT {'per-window' if per_window else 'per-run'} {what}"
                  + (f" — {pattern}" if pattern is not None else ""), fontsize=10)
        if not legend:
            labels = [_gt_mean_axis_label(agg[0], v) for v in curve_vals]
            handles = [plt.Line2D([], [], color=colors[v], linewidth=1.6) for v in curve_vals]
            _legend_below_axes(fig, handles, labels, ncol=min(len(labels), 8), fontsize=8)
        plt.show()
        return fig

    def _gt_window_gaps(self, axes, topologies, exclude_keywords=(), pattern=None, normalize=False):
        rows, ranges = [], {}
        for w, root in self._caster_window_roots(axes):
            for parts, null, alt in _caster_scores_by_run(root, tuple(exclude_keywords), pattern):
                if not len(null) or not len(alt):
                    continue
                cu = _sim_cu_branch_length(str(root / Path(*parts[:3])))
                meta = {"_window_size": w, "category": parts[0], "subcategory": parts[1],
                        "merged_category": _POOLED_SCORE_AXES["merged_category"](w, parts), "pattern": parts[3],
                        "run_id": "/".join(parts), "_cu": cu,
                        "branch_length": None if cu is None else assign_bin(cu, BRANCH_LENGTH_BIN_EDGES)}
                for t in topologies:
                    col = TOPOLOGY_NAMES.index(t)
                    vals = np.concatenate([null[:, col], alt[:, col]])
                    vals = vals[np.isfinite(vals)]
                    if len(vals):
                        lo, hi = ranges.get(t, (np.inf, -np.inf))
                        ranges[t] = (min(lo, vals.min()), max(hi, vals.max()))
                    rows.append(pd.DataFrame({**meta, "topology": t, "_gap": alt[:, col] - np.nanmean(null[:, col])}))
        if not rows:
            raise ValueError(f"No caster scores for {axes}")
        long = pd.concat(rows, ignore_index=True)
        if normalize:
            span = long["topology"].map({t: (hi - lo if hi > lo else 1.0) for t, (lo, hi) in ranges.items()})
            long["_gap"] = (1 if normalize == "0-1" else 2) * long["_gap"] / span
        return long

    def _run_color_values(self, color, w, root, runs, exclude_keywords=(), leaf=None):
        if color == "bl":
            return [_sim_cu_branch_length(str(root / Path(*parts[:3]))) or np.nan for parts, *_ in runs]
        if color in _POOLED_SCORE_AXES and color != "-w":
            return [_POOLED_SCORE_AXES[color](w, parts) for parts, *_ in runs]
        if color in ("anomaly_fraction", "outlier_pct"):
            scale = 100 if color == "outlier_pct" else 1
            fracs = [nominal_anomaly_fraction(parts[3]) for parts, *_ in runs]
            return [scale * (f if f is not None else len(alt) / max(len(null) + len(alt), 1) if null is not None else np.nan)
                    for f, (_, null, alt) in zip(fracs, runs)]
        table = self._runs_tsv_column(color, w, leaf, tuple(exclude_keywords))
        return [table.get(f"{'/'.join(parts[:3])}::{parts[3]}", np.nan) for parts, *_ in runs]

    def _runs_tsv_column(self, col, w, leaf=None, exclude_keywords=()):
        key = (col, w, leaf and str(leaf), exclude_keywords)
        if key not in self._color_cache:
            dirs = [Path(leaf)] if leaf is not None else [
                d for d in self._all_leaf_dirs() if not any(k in str(d) for k in exclude_keywords)
                and self._get_args(d).get("window_size") == w and (self._get_args(d).get("step_size") or w) == w]
            frames = [pd.read_csv(d / "runs.tsv", sep="\t", usecols=lambda c: c in ("run_id", col)) for d in dirs if (d / "runs.tsv").exists()]
            df = pd.concat([f for f in frames if col in f.columns] or [pd.DataFrame(columns=["run_id", col])])
            if df.empty:
                raise ValueError(f"color={col!r}: not a runs.tsv column (or no runs at w{format_val(w)})")
            grouped = df.groupby("run_id")[col]
            self._color_cache[key] = (grouped.mean() if pd.api.types.is_numeric_dtype(df[col])
                                      else grouped.agg(_aggregate_color)).to_dict()
        return self._color_cache[key]

    def _branch_colors(self, color, w, root, exps, bls, exclude_keywords=(), leaf=None):
        if color in (None, "bl"):
            values = {k: bls[k] or np.nan for k in exps}
            spec = _color_spec(list(values.values()), "bl", log=True, cmap="viridis")
        else:
            values = {k: _aggregate_color(self._run_color_values(color, w, root, runs, exclude_keywords, leaf)) for k, runs in exps.items()}
            spec = _color_spec(list(values.values()), color)
        return spec, dict(zip(values, _color_codes(spec, list(values.values())))) if spec else {}

    def _caster_window_roots(self, axes):
        windows = axes.get("-w", axes.get("--window-size"))
        if windows is None:
            raise ValueError("window_ecdf needs a '-w' axis")
        dests = {_FLAG_TO_DEST.get(k): v for k, v in axes.items()}
        variants = {d: bool(dests.get(d)) for d in _SCORES_VARIANT_DESTS if d != "chunk_size"}
        roots = []
        for w in windows if isinstance(windows, (list, tuple)) else [windows]:
            w = _typed_value("-w", w)
            scores = get_expected_scores_path(Path("simulations", "10X", "down", "N0", "concat", "_.fa"), Path("10X", "down", "N0"),
                                              window_size=w, step_size=w, **variants)
            roots.append((w, scores.parents[4]))
        return sorted(roots, key=lambda r: r[0])

    def plot(self, *args, show_axes=True, legend=None, gtrees=False, **kwargs):
        if legend not in (None, "hide", "values"):
            raise ValueError(f"legend must be None, 'hide', or 'values', got {legend!r}")
        args = list(args)
        for i, name in ((2, "metrics"), (3, "agg")):
            if len(args) > i:
                args[i] = _as_list(args[i])
            elif name in kwargs:
                kwargs[name] = _as_list(kwargs[name])
        options = dict(kwargs.get("options") or {})
        admixture = options.pop("admixture", True)
        kwargs["options"] = options or None
        if not admixture:
            kwargs["exclude_keywords"] = (*kwargs.get("exclude_keywords", ()), "admixture")
        with _hiding_admixture(not admixture):
            figs = self._plot_vs_paper(*args, **kwargs) if gtrees else self._plot(*args, **kwargs)
        split = isinstance(figs, list) and any(isinstance(f, list) for f in figs)
        for fig in [f for row in figs for f in row] if split else figs if isinstance(figs, (list, tuple)) else [figs]:
            if fig is None:
                continue
            legends = fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
            if legend == "hide":
                for leg in legends:
                    leg.remove()
            elif legend is None:
                for leg in legends:
                    leg.set_title("")
                    for text in leg.get_texts():
                        text.set_text(re.sub(r"\s*\([^()]*±[^()]*\)$", "", text.get_text()))
                    if all(text.get_text() == "all" for text in leg.get_texts()):
                        leg.remove()
                if getattr(fig, "_legend_params", "") and fig.legends:
                    fig.legends[0].set_title(fig._legend_params, prop={"size": 8})
            if not show_axes:
                if fig._suptitle is not None:
                    fig._suptitle.set_text("")
                for label in (fig._supxlabel, fig._supylabel):
                    if label is not None:
                        label.set_visible(False)
                for ax in fig.axes:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_title("")
        return [_hstack_figs(row) for row in figs] if split else figs

    def _plot_silent(self, config, **kwargs):
        calls, draw, show = [], _draw_stat, plt.show

        def spy(ax, pos, vals, color, width, plot_type, errorbar="default"):
            before = set(ax.get_children())
            out = draw(ax, pos, vals, color, width, plot_type, errorbar)
            calls.append((ax, pos, width, color, [a for a in ax.get_children() if a not in before]))
            return out

        globals()["_draw_stat"], plt.show = spy, lambda *a, **k: None
        try:
            figs = self._plot(config, **kwargs)
        finally:
            globals()["_draw_stat"], plt.show = draw, show
        return (figs[0] if isinstance(figs, list) else figs), calls

    def _plot_vs_paper(self, config, metrics=("f1",), agg=None, **kwargs):
        def with_sources(c):
            sources = c.get("phlag", ["msa"])
            sources = [sources] if isinstance(sources, str) else list(sources)
            return dict(c, phlag=[PAPER_SOURCE] + [x for x in sources if x != PAPER_SOURCE])
        config = [with_sources(c) for c in config] if isinstance(config, (list, tuple)) else with_sources(config)
        figs = [self._tpr_fpr_vs_paper(config, agg, **kwargs) if metric == "tpr_fpr"
                else self._metric_vs_paper(config, metric, agg, **kwargs) for metric in metrics]
        return figs[0] if len(figs) == 1 else figs

    def _metric_vs_paper(self, config, metric, agg=None, plot_type="violin", errorbar="sem", **kwargs):
        from matplotlib.colors import to_hex
        if metric == "f1":
            agg = agg or [("phlag", "-w"), "x_bin", "merged_category", "fraction_bin"]
            paper_of = lambda row, col, label: PHLAG_PAPER_REPLICATE_F1[row][col].get(label, (0.0, 0.0))
        else:
            agg = agg or [("phlag", "-w"), "", "merged_category", "fraction_bin"]
            paper_of = lambda row, col, label: (PHLAG_PAPER_FPR_TPR[row][col][1 if metric == "tpr" else 0], 0.0)
        fig, calls = self._plot_silent(config, metrics=[metric], agg=agg, plot_type=plot_type, errorbar=errorbar, **kwargs)
        legend = fig.legends[0]
        paper_idx = next(i for i, t in enumerate(legend.get_texts()) if t.get_text().startswith(PAPER_SOURCE))
        paper_hex = to_hex(legend.legend_handles[paper_idx].get_facecolor())
        rows = list(PHLAG_PAPER_F1)
        for ax, pos, width, color, artists in calls:
            if to_hex(color) != paper_hex:
                continue
            for a in artists:
                a.remove()
            spec = ax.get_subplotspec()
            ticks = ax.get_xticks()
            label = ax.get_xticklabels()[int(np.argmin(np.abs(ticks - pos)))].get_text()
            mean, err = paper_of(rows[spec.rowspan.start], _MERGED_CATEGORY_ORDER[spec.colspan.start], label)
            ax.bar(pos, mean, width=width * 0.92, color=paper_hex, zorder=2)
            if err and errorbar is not None:
                ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=3)
        legend.get_texts()[paper_idx].set_text("Phlag (paper)")
        spread = {"default": "SEM", "sem": "SEM", "sd": "SD"}.get(errorbar)
        if plot_type == "violin":
            note = ("Violin: KDE of values inside 1.5·IQR, truncated at their min/max; "
                    "dots = outliers beyond it.")
            if errorbar not in (None, "default"):
                note += f" Middle tick = mean, capped ticks = mean ± {spread}."
            else:
                note += " Tick = mean."
        else:
            note = f"Bars: mean ± {spread}." if spread else "Bars: mean."
        if errorbar is not None:
            note += "  Phlag (paper): published mean ± paper's SEM."
        fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=7, color="gray")
        return fig

    def _tpr_fpr_vs_paper(self, config, agg=None, **kwargs):
        from matplotlib.colors import to_hex
        from matplotlib.lines import Line2D
        agg = agg or [("phlag", "-w"), "", "merged_category", "fraction_bin"]
        fig, _ = self._plot_silent(config, metrics=["tpr_fpr"], agg=agg, **kwargs)
        source_legend = next(l for l in fig.legends if any(t.get_text().startswith(PAPER_SOURCE) for t in l.get_texts()))
        entries = [(t.get_text(), to_hex(h.get_markerfacecolor() if isinstance(h, Line2D) else h.get_facecolor()[0]))
                   for t, h in zip(source_legend.get_texts(), source_legend.legend_handles)]
        paper_hex = next(c for label, c in entries if label.startswith(PAPER_SOURCE))
        rows = list(PHLAG_PAPER_FPR_TPR)
        for ax in fig.axes:
            if ax.get_subplotspec() is None:
                continue
            for artist in list(ax.collections):
                if len(artist.get_facecolor()) and all(to_hex(c) == paper_hex for c in artist.get_facecolor()):
                    artist.remove()
            for line in list(ax.lines):
                if to_hex(line.get_color()) == paper_hex:
                    line.remove()
            spec = ax.get_subplotspec()
            fpr, tpr = PHLAG_PAPER_FPR_TPR[rows[spec.rowspan.start]][_MERGED_CATEGORY_ORDER[spec.colspan.start]]
            ax.scatter(fpr, tpr, marker="*", s=180, color=paper_hex, edgecolor="black", linewidth=0.6, zorder=6)
        handles = [Line2D([], [], marker="*", ls="", ms=12, color=paper_hex, mec="black", label="Phlag (paper)")]
        handles += [Line2D([], [], marker="o", ls="", color=c, label=label) for label, c in entries if not label.startswith(PAPER_SOURCE)]
        fig.legend(handles=handles, loc="upper right", fontsize=8)
        source_legend.remove()
        return fig

    def _plot(self, axes, run=None, metrics=None, agg=(), exclude_keywords=(), plot_type="bar", grid_by=None,
              title="", options=None, show_config_suffix=True, logy=False, dir="store", shared=False, agg_idx=0,
              errorbar="default", color=None, scalex=False, scaley=False):
        """`run` (a directory, e.g. "out/10X/down/N276/37-62") makes this a
        run-specific plot: `axes` is still the config dict, but it now only
        FILTERS the reports found under `run` (see collect_out_runs) instead of
        being resolved against the store -- each remaining report is one point,
        x = window[/step], one line per remaining path segment combination
        (dist/variant/rho_beta/...), one subplot per `metrics` element (a
        tuple overlays its members, e.g. ("gt_transition_null_to_alt",
        "fitted_transition_null_to_alt")). An axes value may be a list (any
        of); flags the out/ path doesn't encode (--np, ...) are skipped.
        Pass {} to keep every report. A dict {label: dir} or a list of dirs
        (label = the dir minus its trailing <node>/<pattern>) plots one
        column per run instead.
        `exclude_keywords` drops report paths containing any keyword; `logy`
        log-scales y; agg/grid_by/plot_type don't apply, except
        metrics=["ecdf"] (or "qq": QQ vs Laplace(median, mean abs dev),
        "fit": ECDF vs Gaussian/Laplace MLE CDFs, "fit_qq": QQ vs the
        Laplace fit with the Gaussian fit's curve, each with KS D boxes;
        option "topologies" limits the drawn ones): `run` is one run, a list, or a {label: run} dict
        (one column each, titled category / node / branch length unless
        labeled); a run without a pattern (category/sub/node) draws one
        figure per pattern. Null (dashed) vs Outlier (solid), all
        topologies overlaid, a Delta(O-N) mean box per panel; rows = the
        "-w" values when several, or agg's row flag. By default windows are
        split by ground truth, read straight from store/caster for each
        "-w" (step = window, variant flags from axes, `dir` unused);
        agg=["em"] splits by the EM state path of out/ (or dir="store")
        reports instead. Every distribution metric (window_ecdf, ecdf/qq/
        fit/fit_qq, gap_ecdf/mean_ecdf/*_vs_bl) takes the same options:
        "pattern" (only that pattern), "topologies", "logx" (log when
        all-positive, else symlog at "linthresh"; per-metric default),
        "normalize" (default False: min-max maps each panel's/series'
        values to [-1, 1]; gap/mean metrics map each topology's Null+Outlier
        means jointly, then take the gap; logx then defaults off), "legend"
        (default True: per-subplot legend / box of each curve's mean;
        False = one shared legend, no means). metrics=["window_ecdf"]:
        every "-w" value overlaid (gray, light = small), cols = topologies,
        (pooled only) agg=[axis] from _POOLED_SCORE_AXES (merged_category,
        x_bin, fraction_bin, ...) overlays that axis instead, one figure per -w,
        rows = GT Null/Outlier; with `run`, one figure per run/pattern of
        its window scores, without, one figure of every store/caster run's
        GT mean (gt_stats.txt) for each window; color= (pooled only)
        colors each ECDF's points by their run's value (window lines + mean
        vlines stay gray on top): "bl" = CU branch length, category/
        subcategory/merged_category/pattern/direction (up/down/admixture)
        from the path, anomaly_fraction/
        outlier_pct from the pattern, any other runs.tsv column = mean (or
        mode) over every store run's row at that window (ecdf_all_hmm: its
        own runs.tsv); numeric = rainbow colorbar, else tab legend. ecdf_all/
        ecdf_all_hmm/mean_all_vs_bl take the same color= per branch
        (patterns averaged; default "bl"). metrics=["ecdf_all"]: one
        figure per "-w", one thin GT window-score ECDF per branch (sim,
        patterns pooled) colored by CU branch length, cols = topologies,
        rows = Null/Outlier; `run` (e.g. "10X/down", a list of them)
        restricts branches by store/caster path prefix; option
        "percentile" (default (1, 99)) sets each column's x range from the
        pooled values; option "fit"=True (also ecdf_all_hmm) overlays the
        pooled clipped ECDF (thick black) and its Gaussian MLE CDF (red
        dashed), μ/σ/KS D in the box. metrics=["mean_all_vs_bl"]: same branches, clipping
        and normalize as ecdf_all, but one point per branch = its clipped
        mean vs CU branch length (log x; logx=False = linear; logy=True = log y) with a fitted
        y = a + b·e^(−kx). normalize="0-1" (any dist metric) min-max
        normalizes to [0, 1] instead of [-1, 1].
        metrics=["ecdf_all_hmm"]: ecdf_all split by each
        report's Viterbi state path instead of GT, one figure per store run
        dir matching `axes`. metrics=["ecdf_hmm"]: one figure per run/pattern
        (store/out reports as agg=["em"]), cols = topologies, rows = agg's
        row flag (e.g. ["--np"]) or "-w", Null EM vs Outlier EM ECDFs split
        by the Viterbi path, vlines/box = state means. Without `run`,
        ecdf_hmm pools every store report matching `axes` (point = one
        report's Viterbi Null/Outlier mean) and ecdf_hmm_all pools their
        windows (point = window, Viterbi-labeled); rows = Null HMM/Outlier
        HMM, cols = topologies, one curve per "-w" (plus config flags in
        agg, or one _POOLED_SCORE_AXES axis); "state" in agg overlays both
        states per panel instead, rows = those curves; with `run`,
        ecdf_hmm_all = ecdf_hmm. dir="store" reads
        the same run's reports from self.root's leaf dirs instead of out/
        (reports/<run>.tsv, filtered by each leaf's args.json, config = leaf
        path minus its size segment -- see collect_store_runs). Without `run`, resolves `axes`/`agg`/`exclude_keywords` via resolve_configs_cartesian
        (see its docstring, including its "" placeholder and tuple-axis
        handling) and renders the result -- one call replaces the old
        separate resolve_configs_cartesian(...) + plot_stat(configs, ...)
        pair. The resolved configs are cached on self._last_configs so a
        follow-up print_configs() (no args) can reuse them without
        re-resolving. Renders one bar/violin per config for each metric in
        `metrics` (any
        runs.tsv numeric column -- "f1", "em_hd", "em_gt_hd", "null_kl_agg",
        "alt_kl_agg", "transition_null_to_alt", "roc_auc", ... -- or the
        special name "tpr_fpr", which renders the full fraction_bin x column x
        x_bin TPR/FPR scatter grid as its own figure instead of a bar/violin,
        since ROC shape needs that breakdown).

        grid_by=None (default), `configs` built with a single-level agg
        (resolve_configs_cartesian's agg had 0 or 1 real axes -- see its
        self._config_axes note): every metric shares one row of subplots,
        pooled across FIGURE_COLUMNS/fraction_bin/x_bin/hd_bin -- a single
        shared legend (config -> color) sits in the corner rather than
        repeating config names as x-ticks on every subplot. An element of
        `metrics` may itself be a tuple of metric names (e.g. metrics=["f1",
        ("em_hd", "em_gt_hd")]) -- those metrics share ONE subplot instead
        of each getting its own: one bar/violin per (config, metric) pair,
        grouped by config along x with metric as the sub-bar hue (own
        in-subplot legend), rather than the usual config-colored single bar.
        Not supported with grid_by="hd_bin" or plot_type="heatmap" -- see
        _plot_grouped below for the multi-level-agg equivalent.

        Every bar/violin subplot pins its y-axis to [0, 1] with 0.2-step
        ticks when its metric(s) are all in _BOUNDED_METRICS ("f1", "em_hd",
        "em_gt_hd", "roc_auc", "transition_null_to_alt",
        "transition_alt_to_null" -- proportions/bounded distances), instead
        of autoscaling or (for violin) _clip_axis_to_whiskers' outlier-based
        clipping, so these figures stay comparable to each other -- see
        _apply_bounded_yaxis.

        grid_by=None, `configs` built with a multi-level agg (2-4 axes):
        each metric instead gets its own figure, laid out hierarchically by
        agg's axes instead of one flat bar per config -- agg[0] draws one
        bar per group (colored, with a shared hue legend), agg[1] positions
        bar-groups along x, a 3rd agg axis splits into subplot columns, a
        4th into subplot rows. E.g. agg=["x_bin", "-w"] draws one bar-group
        per window size, each holding one bar per x_bin. If EVERY element of
        `metrics` is instead a same-length tuple (e.g. metrics=[("null_mean_ABBA",
        "alt_mean_ABBA"), ("null_mean_BABA", "alt_mean_BABA")]), agg[0]/[1]
        keep their hue/bar-position roles but `metrics` itself takes over the
        subplot grid role instead of agg[2]/[3] -- one row per `metrics` list
        entry, one column per tuple position, all in ONE shared figure (see
        _plot_grouped's tuple-metrics branch) -- so this only works when agg
        resolves to exactly 2 real levels (hue + bar-position); a 3rd/4th
        real agg level alongside tuple metrics has nowhere left to put the
        grid and raises instead. metrics=["tpr_fpr"]
        with a multi-level agg gets the same hierarchy but as a scatter/line
        plot instead of bars: agg[0] -> one colored line per value (shared
        legend, different segments), agg[1] -> points along that same
        segment (shaded light->dark), agg[2]/agg[3] -> subplot columns/rows
        (see _plot_tpr_fpr_grouped). A single-level (or no) agg keeps the old
        fixed fraction_bin x FIGURE_COLUMNS x_bin grid (_plot_tpr_fpr_grid).

        agg_idx (multi-level agg only, incl. tpr_fpr): colors bars/points and builds the
        legend from agg[agg_idx] instead of agg[0], without moving any axis
        out of its layout slot -- see _plot_grouped.

        grid_by="hd_bin": each metric instead gets its OWN figure, gridded
        fraction_bin (rows, 2) x hd_bin (columns, HD_NUM_BINS=6) -- e.g.
        metrics=["f1"] with 2 configs gives 2 bars/cell x 2 rows x 6 columns.
        hd_bin here is recomputed fresh across ALL of `configs`' pooled
        em_gt_hd values (same equal-width-bins-over-the-observed-range logic
        as BenchmarkStats._compute_hd_bins in benchmark.py) rather than
        trusting each leaf's own runs.tsv hd_bin column -- that column's edges
        are computed per underlying benchmark run, so pooling leaf dirs with
        different em_gt_hd ranges (e.g. a config spanning several lam values)
        would otherwise mix many mutually-inconsistent boundary strings
        instead of one shared 6-bin axis. Ignored when plot_type="heatmap".

        plot_type="heatmap" with grid_by="hd_bin": one figure per metric,
        one subplot per config, cells colored by that metric's mean value on
        a hd_bin (columns) x fraction_bin (rows) grid -- same hd_bin
        recomputation as the bar/violin grid_by="hd_bin" path above, just
        rendered as a heatmap instead of one bar/violin per cell. All
        configs' subplots in a figure share one color scale (min/max across
        every config's grid) so shading is comparable across e.g. dvi
        on/off. See _plot_heatmap_hd_bin.

        plot_type="heatmap" with grid_by set to anything else (a mirrored
        CLI flag token like "--double-variance-init", or a
        _DATA_COLUMN_AXES/_DYNAMIC_ROW_AXES name): splits `configs` into
        groups by that axis's own value (read off self._config_axes /
        self._agg_axis_names -- i.e. it must be one of the axes
        resolve_configs_cartesian's `agg` actually produced for these
        configs) and renders one full hd_bin x fraction_bin heatmap (see
        above) PER GROUP instead of cramming every config into one shared
        figure -- e.g. `configs` built via agg=["--double-variance-init"]
        (2 configs, dvi on/off) with grid_by="--double-variance-init" grouped ==
        each config on its own, so this renders 2 separate figures per
        metric instead of 1 figure with 2 subplots. A config whose agg axes
        don't include the grid_by axis is bucketed under a group named
        None. See _group_configs_by_axis.

        plot_type="heatmap" without grid_by="hd_bin": same agg[0..3] role
        layout as the bar/violin case above, just x/y instead of bar/hue --
        agg[0] -> x axis, agg[1] -> y axis, agg[2] -> subplot columns, agg[3]
        -> subplot rows, each cell colored by that metric's mean value. With
        multiple `metrics` and agg using exactly 2 or 3 real levels, `metrics`
        itself fills the next role after agg's last one (one metric per
        subplot column with a 2-level agg, one per subplot row with a
        3-level agg) instead of each metric getting a separate figure -- see
        _plot_heatmap. Each of x/y whose tokens are all numeric (e.g. -w's
        "w50k", outlier_pct's "25%") is placed at its real values (log if
        positive and spanning >20x, else linear; options logx/logy force
        it), otherwise categorical -- see _heatmap_axis. All of a figure's subplots share one
        color scale. A boolean agg axis in its off/negated state contributes
        no display value at all (see _combo_label_part), so mixing a bool
        toggle into agg[0]/agg[1] collapses that axis to one tick; stick to
        real-valued axes (like -w/-s, or a *_bin/merged_category axis) for
        x/y.

        errorbar: "default" keeps bar=mean+-sem, violin=Tukey whiskers only;
        "sd"/"sem" sets the bar error and overlays mean+-it on violins; None
        draws no error bars or whiskers.

        Returns a list of Figures: the tpr_fpr grid first (if requested),
        then one figure per remaining metric (grid_by or heatmap set) or one
        shared row figure for all of them (grid_by=None, plot_type!="heatmap").
        Also sets self.metrics to the scalar metrics used (i.e. `metrics`
        minus "tpr_fpr") -- print_configs defaults to that instead of
        requiring metrics= on every call.

        options also takes "admixture" (default True; False hides admixture
        rows/columns/x_bins and the admixture path keyword) and "tpr_fpr_full"
        (default False; True shows the tpr_fpr grid's full 0-1 FPR axis
        instead of 0-0.5).

        options=None (default): a dict of {"logx", "logy", "interp"} bools,
        all off by default, applied only in the bar/violin paths (the
        single-level-agg path below and _plot_grouped -- not heatmap, whose
        own numeric-axis logic is separate). "logx" repositions a bar/x
        axis whose category tokens ALL parse as positive numbers (e.g. -w's
        "w50k"/"w1k" tokens, via _token_to_number) onto their real numeric
        values on a log-scaled axis, with per-position width/hue-offsets
        scaled multiplicatively (proportional to that position's value)
        instead of the usual fixed-width additive offsets around an integer
        index -- categorical (non-numeric) axes ignore "logx" and render as
        before. "logy" sets a log y-scale (True) or picks one per subplot
        ("auto": log only when that subplot's pooled values are all > 0 and
        span >= 2 decades, see _resolve_logy) and skips _apply_bounded_yaxis's
        [0, 1] pin (log and a hard 0 floor don't mix). "interp" overlays one
        line per hue/sub-metric connecting that series' mean across bar
        positions, sorted by position -- a visual trend guide alongside the
        bars/violins, not a replacement for them.

        show_config_suffix=True (default): every figure title gets
        _title_suffix's " — <restriction>" text appended, stating which
        fixed (non-agg) axis values these configs were pooled/restricted to
        (e.g. "d gaussian np free ap free rho None beta None s 1.0"). Pass
        False for metrics that don't actually depend on those fixed axes
        (e.g. ground-truth stats read straight from gt_stats.txt, computed
        by caster/benchmark before phlag's EM ever runs) so the title
        doesn't imply a dependency that isn't there."""
        assert metrics is not None, "metrics is required"
        dist_metric = next((m for m in metrics if m in _DIST_METRICS), None)
        if dist_metric is not None:
            if len(metrics) != 1:
                raise ValueError(f"pass metrics=[{dist_metric!r}] on its own")
            unknown = set(options or {}) - _DIST_OPTIONS - ({"percentile"} if dist_metric in ("ecdf_all", "ecdf_all_hmm", "mean_all_vs_bl") else set()) \
                - ({"fit"} if dist_metric in ("ecdf_all", "ecdf_all_hmm") else set()) \
                - ({"logy"} if dist_metric == "mean_all_vs_bl" else set()) \
                - ({"per_window"} if dist_metric == "gap_ecdf" else set()) \
                - ({"state", "topology", "corr", "percentile"} if dist_metric in _TOPOLOGY_SCATTER_METRICS else set())
            if unknown:
                raise ValueError(f"{dist_metric}: unknown options {sorted(unknown)} (takes {sorted(_DIST_OPTIONS)})")
            opts = {"pattern": None, "topologies": TOPOLOGY_NAMES, "logx": None, "linthresh": None, "normalize": False,
                    "legend": True, **(options or {})}
            runs = None if run is None else list(run.items()) if isinstance(run, dict) else [(None, p) for p in run] if isinstance(run, (list, tuple)) else [(None, run)]
            runs = runs and [(label, _resolve_run_path(p)) for label, p in runs]
            if dist_metric in ("ecdf_all", "mean_all_vs_bl"):
                return self._plot_ecdf_all(axes, [run] if isinstance(run, str) else run, exclude_keywords=exclude_keywords,
                                           title=title, vs_bl=dist_metric == "mean_all_vs_bl", color=color, **opts)
            if dist_metric == "ecdf_all_hmm":
                return self._plot_ecdf_all_hmm(axes, [run] if isinstance(run, str) else run, exclude_keywords=exclude_keywords,
                                              title=title, color=color, **opts)
            if dist_metric in ("ecdf_hmm", "ecdf_hmm_all"):
                if run is None:
                    return self._plot_ecdf_hmm_pooled(axes, agg=agg, exclude_keywords=exclude_keywords, title=title,
                                                      windows=dist_metric == "ecdf_hmm_all", **opts)
                return self._plot_ecdf_hmm(axes, runs, agg=agg, exclude_keywords=exclude_keywords, title=title, dir=dir, **opts)
            if (dist_metric in _TOPOLOGY_SCATTER_METRICS) != (plot_type == "scatter"):
                raise ValueError(f"plot_type='scatter' goes with metrics=[one of {sorted(_TOPOLOGY_SCATTER_METRICS)}]")
            if dist_metric in _TOPOLOGY_SCATTER_METRICS:
                return self._plot_topology_scatter(axes, [run] if isinstance(run, str) else run, exclude_keywords=exclude_keywords,
                                                   title=title, stat=_TOPOLOGY_SCATTER_METRICS[dist_metric], color=color, agg=agg, **opts)
            if dist_metric == "window_ecdf":
                return self._plot_window_ecdf(axes, runs, exclude_keywords=exclude_keywords, title=title, color=color, agg=agg, **opts)
            if dist_metric in _SCORE_DIST_METRICS:
                if run is None:
                    return self._plot_score_dist_pooled(dist_metric, axes, agg=agg, exclude_keywords=exclude_keywords,
                                                        title=title, **opts)
                return self._plot_score_dist(dist_metric, axes, runs, agg=agg, exclude_keywords=exclude_keywords,
                                             title=title, dir=dir, **opts)
            if run is not None:
                raise ValueError(f"{dist_metric!r} takes no run=")
            return self._plot_gt_mean_ecdf(dist_metric, axes, agg=agg, exclude_keywords=exclude_keywords, title=title, **opts)
        if isinstance(run, (list, tuple)):
            run = {"/".join(Path(p).parts[:-2]) or str(p): p for p in run}
        if isinstance(run, dict):
            run = {label: _resolve_run_path(p) for label, p in run.items()}
        elif run is not None:
            run = _resolve_run_path(run)
        if isinstance(run, dict):
            return self._plot_branches(axes, run, metrics, exclude_keywords=exclude_keywords, title=title, dir=dir)
        if run is not None:
            return self._plot_run(axes, run, metrics, exclude_keywords=exclude_keywords, logy=logy, title=title, dir=dir)
        assert plot_type in ("bar", "violin", "heatmap", "line")
        assert grid_by in (None, "hd_bin") or plot_type == "heatmap"
        assert grid_by is None or plot_type != "line"
        options = options or {}
        logx, interp = bool(options.get("logx")), bool(options.get("interp"))
        logy = options.get("logy", logy)
        configs = self.resolve_configs_cartesian(axes, exclude_keywords=exclude_keywords, agg=agg)
        if shared is not False and shared is not None:
            configs = self._shared_configs(configs, 1 if shared is True else shared)
        self._last_configs = configs
        suffix = self._title_suffix(configs) if show_config_suffix else ""
        params, suffix = (_compose_title(suffix), "") if title else ("", suffix)
        kw = dict(plot_type=plot_type, grid_by=grid_by, title=title, options=options, logx=logx, logy=logy,
                  interp=interp, agg_idx=agg_idx, errorbar=errorbar, color=color, scalex=scalex, scaley=scaley)
        levels = max((len(self._config_axes.get(label, [])) for label, _ in configs), default=0)
        if levels <= 4:
            return _tag_legend_params(self._plot_resolved(configs, metrics, suffix, **kw), params)
        split_idx = min(range(levels), key=lambda i: (len(self._axis_values(configs, i)), -i))
        split_name = _axis_flag_label(self._agg_axis_names[configs[0][0]][split_idx])
        order = self._axis_values(configs, split_idx)
        groups = {}
        for label, dirs in configs:
            groups.setdefault(self._config_axes[label][split_idx], []).append((label, dirs))
            for store in (self._config_axes, self._config_rank, self._agg_axis_names):
                store[label] = type(store[label])(v for i, v in enumerate(store[label]) if i != split_idx)
        kw["agg_idx"] = agg_idx - (agg_idx > split_idx)
        per_group = [self._plot_resolved(groups[value], metrics, f"{suffix} — {split_name}={value}", **kw)
                     for value in order]
        return _tag_legend_params([list(row) for row in zip(*per_group)], params)

    def _plot_resolved(self, configs, metrics, suffix, plot_type, grid_by, title, options, logx, logy, interp,
                       agg_idx, errorbar, color, scalex, scaley):
        figs = []
        levels = max((len(self._config_axes.get(label, [])) for label, _ in configs), default=0)
        if agg_idx and not 0 <= agg_idx < levels:
            raise ValueError(f"agg_idx={agg_idx} out of range for {levels} agg level(s)")
        if "tpr_fpr" in metrics:
            tpr_fpr_title = _compose_title(suffix, title, "TPR/FPR")
            full = bool(options.get("tpr_fpr_full"))
            if levels >= 2:
                figs.append(self._plot_tpr_fpr_grouped(configs, title=tpr_fpr_title, agg_idx=agg_idx, full=full))
            else:
                figs.append(self._plot_tpr_fpr_grid(configs, title=tpr_fpr_title, full=full))
        metrics = [m for m in metrics if m != "tpr_fpr"]
        self.metrics = [m for panel in metrics for m in (panel if isinstance(panel, tuple) else (panel,))]
        if not metrics:
            return figs

        if plot_type == "line":
            figs.extend(self._plot_beta_line(configs, metrics, title_suffix=suffix,
                                             logx=options.get("logx", True), logy=logy, errorbar=errorbar,
                                             color=color))
            return figs

        if plot_type == "heatmap":
            if grid_by == "hd_bin":
                figs.extend(self._plot_heatmap_hd_bin(configs, metrics, title_suffix=suffix))
            elif grid_by is not None:
                groups = self._group_configs_by_axis(configs, grid_by)
                axis_label = _strip_flag(grid_by) if grid_by in _FLAG_TO_DEST else grid_by
                for group_value, group_configs in groups:
                    group_suffix = f"{suffix} — {axis_label}={group_value}"
                    figs.extend(self._plot_heatmap_hd_bin(group_configs, metrics, title_suffix=group_suffix))
            else:
                figs.extend(self._plot_heatmap(configs, metrics, title_suffix=suffix,
                                               scale=(scalex, scaley)))
            return figs

        if levels >= 2 and grid_by is None:
            figs.extend(self._plot_grouped(configs, metrics, plot_type, title_suffix=suffix, title=title,
                                            logx=logx, logy=logy, interp=interp, agg_idx=agg_idx,
                                            errorbar=errorbar))
            return figs

        colors = {label: _config_color(i) for i, (label, _) in enumerate(configs)}
        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)
        legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[label]) for label, _ in configs]
        legend_labels = [label for label, _ in configs]

        if grid_by == "hd_bin":
            all_gt_hd = [v for d in dfs.values() if not d.empty for v in d["em_gt_hd"].dropna().tolist()]
            edges = _compute_hd_bin_edges(all_gt_hd)
            hd_bins = [label for label, _, _ in edges]
            hd_dfs = {
                label: (d.assign(_hd_bin=d["em_gt_hd"].apply(lambda v: assign_bin(v, edges))) if not d.empty else d)
                for label, d in dfs.items()
            }
            n = len(configs)
            n_rows, n_cols = len(FRACTION_BINS), len(hd_bins)
            # Reserve canvas height for however many config legend rows there
            # are, then re-measure the legend's actual rendered extent below
            # (rather than trusting a fixed bbox_to_anchor/tight_layout
            # margin) -- a long config sweep otherwise pushes the legend past
            # the suptitle or into the axes. See _plot_tpr_fpr_grouped.
            legend_extra_in = 0.24 * n
            for metric in metrics:
                panel_col = _METRIC_PANEL_COLUMN.get(metric)
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(2.0 * n_cols, 3.0 * n_rows + 0.6 + legend_extra_in), squeeze=False)
                for row_idx, fraction_bin in enumerate(FRACTION_BINS):
                    for col_idx, hd_bin in enumerate(hd_bins):
                        ax = axes[row_idx][col_idx]
                        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                        ax.set_axisbelow(True)
                        bounded = _apply_bounded_yaxis(ax, metric)
                        ax.set_xlim(-0.7, n - 0.3)
                        ax.set_xticks([])
                        if row_idx == 0:
                            ax.set_title(hd_bin, fontsize=7.5)
                        if col_idx == 0:
                            ax.set_ylabel(f"alt proportion {fraction_bin}", fontsize=8.5)
                        cell_bounds = []
                        for cfg_idx, (label, _) in enumerate(configs):
                            d = hd_dfs[label]
                            if d.empty:
                                continue
                            dd = d[(d["fraction_bin"] == fraction_bin) & (d["_hd_bin"] == hd_bin)]
                            if panel_col is not None:
                                dd = dd[_is_true(dd[panel_col])]
                            if self._agg_given:
                                _assert_single_run(dd)
                            cell_bounds.append(_draw_stat(ax, cfg_idx, dd[metric], colors[label], 1.0, plot_type, errorbar))
                        if plot_type == "violin" and not bounded:
                            _clip_axis_to_whiskers(ax, cell_bounds)
                metric_part = metric if len(metrics) > 1 else ""
                _suptitle(fig, _compose_title(suffix, metric_part, "by HD bin"), fontsize=12, y=0.995)
                metric_values = self._metric_by_label(configs, metric)
                agg_legend_labels = [f"{label} ({metric}={metric_values[label]:.4f})" for label in legend_labels]
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                fig_h_px = fig.bbox.height
                cursor_y = (fig._suptitle.get_window_extent(renderer).y0 - 4) / fig_h_px
                legend = fig.legend(legend_handles, agg_legend_labels, loc="upper right",
                                     bbox_to_anchor=(0.995, cursor_y), fontsize=8, framealpha=0.9)
                fig.canvas.draw()
                cursor_y = (legend.get_window_extent(renderer).y0 - 4) / fig_h_px
                fig.tight_layout(rect=[0, 0.02, 1, _below_title_top(fig, cursor_y)])
                figs.append(fig)
            return figs

        n = len(configs)
        # options={"logx": True}: only takes effect when every config's own
        # label parses as a positive number (e.g. a single -w agg axis's
        # "w50k"/"w1k" tokens) -- otherwise falls back to the usual
        # evenly-spaced integer index below. See _LOG_SPAN for the
        # multiplicative sub-bar/violin spread this uses in place of
        # sub_width's fixed additive spread in the categorical case.
        bar_nums = [_token_to_number(l) for l in legend_labels]
        use_logx = logx and n > 0 and all(v is not None and v > 0 for v in bar_nums)
        positions = bar_nums if use_logx else list(range(n))
        # Same dynamic-height + remeasured-extent legend placement as the
        # grid_by="hd_bin" branch above -- a fixed bbox/tight_layout margin
        # here overlapped the suptitle or axes once configs/metrics grew
        # long enough to wrap the legend onto extra lines.
        legend_extra_in = 0.24 * n
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(2.8 * len(metrics), 4.0 + legend_extra_in), squeeze=False)
        for ax, panel in zip(axes[0], metrics):
            if use_logx:
                ax.set_xscale("log")
                tick_order = sorted(range(n), key=lambda i: positions[i])
                ax.set_xticks([positions[i] for i in tick_order])
                ax.set_xticklabels([legend_labels[i] for i in tick_order], rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
                ax.set_xlim(min(positions) / 1.5, max(positions) * 1.5)
            else:
                ax.set_xlim(-0.7, n - 0.3)
                ax.set_xticks(range(n))
                ax.set_xticklabels(legend_labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if isinstance(panel, tuple):
                ax.set_title(" / ".join(panel), fontsize=9.5)
                bounded = False
                if _resolve_logy(logy, dfs, panel):
                    ax.set_yscale("log")
                else:
                    bounded = _apply_bounded_yaxis(ax, panel)
                panel_colors = {m: _config_color(i) for i, m in enumerate(panel)}
                sub_width = 0.82 / len(panel)
                cell_bounds = []
                interp_points = {m: [] for m in panel}
                for cfg_idx, (label, _) in enumerate(configs):
                    d = dfs[label]
                    if d.empty:
                        continue
                    pos = positions[cfg_idx]
                    for mi, m in enumerate(panel):
                        panel_col = _METRIC_PANEL_COLUMN.get(m)
                        dd = d[_is_true(d[panel_col])] if panel_col is not None else d
                        frac = (mi - (len(panel) - 1) / 2) / len(panel)
                        x = pos * (1 + frac * _LOG_SPAN) if use_logx else pos + frac * len(panel) * sub_width
                        w = pos * (_LOG_SPAN / len(panel)) if use_logx else sub_width
                        if self._agg_given:
                            _assert_single_run(dd)
                        cell_bounds.append(_draw_stat(ax, x, dd[m], panel_colors[m], w, plot_type, errorbar))
                        if interp:
                            mean = dd[m].dropna().mean()
                            if not pd.isna(mean):
                                interp_points[m].append((pos, mean))
                if plot_type == "violin" and not bounded:
                    _clip_axis_to_whiskers(ax, cell_bounds)
                if interp:
                    for m in panel:
                        pts = sorted(interp_points[m])
                        if len(pts) > 1:
                            xs, ys = zip(*pts)
                            ax.plot(xs, ys, "-", color=panel_colors[m], linewidth=1.3, alpha=0.85, zorder=4)
                handles = [plt.Rectangle((0, 0), 1, 1, facecolor=panel_colors[m]) for m in panel]
                ax.legend(handles, panel, loc="best", fontsize=6.5, framealpha=0.85)
            else:
                metric = panel
                panel_col = _METRIC_PANEL_COLUMN.get(metric)
                ax.set_title(metric, fontsize=9.5)
                bounded = False
                if _resolve_logy(logy, dfs, metric):
                    ax.set_yscale("log")
                else:
                    bounded = _apply_bounded_yaxis(ax, metric)
                cell_bounds = []
                interp_pts = []
                for cfg_idx, (label, _) in enumerate(configs):
                    d = dfs[label]
                    if d.empty:
                        continue
                    if panel_col is not None:
                        d = d[_is_true(d[panel_col])]
                    pos = positions[cfg_idx]
                    w = pos * 0.5 if use_logx else 1.0
                    if self._agg_given:
                        _assert_single_run(d)
                    cell_bounds.append(_draw_stat(ax, pos, d[metric], colors[label], w, plot_type, errorbar))
                    if interp:
                        mean = d[metric].dropna().mean()
                        if not pd.isna(mean):
                            interp_pts.append((pos, mean))
                if plot_type == "violin" and not bounded:
                    _clip_axis_to_whiskers(ax, cell_bounds)
                if interp and len(interp_pts) > 1:
                    interp_pts.sort()
                    xs, ys = zip(*interp_pts)
                    ax.plot(xs, ys, "-", color="black", linewidth=1.2, alpha=0.6, zorder=4)
        _suptitle(fig, _compose_title(suffix, title), fontsize=12, y=0.99)
        flat_metrics = [m for panel in metrics for m in (panel if isinstance(panel, tuple) else (panel,))]
        metric_values = {metric: self._metric_by_label(configs, metric) for metric in flat_metrics}
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_h_px = fig.bbox.height
        cursor_y = (fig._suptitle.get_window_extent(renderer).y0 - 4) / fig_h_px
        # When every metric is a tuple panel, colors[label]/legend_handles above were
        # never used to draw anything -- each panel painted with its own panel_colors
        # (keyed by metric name, not config) and already has its own in-axes legend
        # (see the isinstance(panel, tuple) branch's ax.legend(...) call). Rendering
        # this config-colored legend anyway would show swatches that don't correspond
        # to what's actually on the axes, so skip it in the all-tuple case.
        if not all(isinstance(m, tuple) for m in metrics):
            agg_legend_labels = [
                f"{label} (" + ", ".join(f"{m}={metric_values[m][label]:.4f}" for m in flat_metrics) + ")"
                for label in legend_labels
            ]
            legend = fig.legend(legend_handles, agg_legend_labels, loc="upper right",
                                 bbox_to_anchor=(0.995, cursor_y), fontsize=8, framealpha=0.9)
            fig.canvas.draw()
            cursor_y = (legend.get_window_extent(renderer).y0 - 4) / fig_h_px
        fig.tight_layout(rect=[0, 0, 1, _below_title_top(fig, cursor_y)])
        figs.append(fig)
        return figs

    def _shared_configs(self, configs, axis=1):
        def split(label):
            values = list(self._config_axes.get(label, []))
            own = values.pop(axis) if axis < len(values) else None
            return own, tuple(values)
        present = {}
        for label, _ in configs:
            own, rest = split(label)
            present.setdefault(rest, set()).add(own)
        common = set.intersection(*present.values()) if present else set()
        return [(label, dirs) for label, dirs in configs if split(label)[0] in common]

    def _title_suffix(self, configs):
        """" -- <pooled-axis restriction>" appended to every plot figure
        title, so a figure states what its configs were restricted to (e.g.
        "w 50k s 40k") without needing a separate print_configs call
        alongside it. Restriction text comes from self._restrictions[label]
        (set by resolve_configs_cartesian from its non-agg axes, same for
        every label produced by one call -- read off the first config);
        empty for configs built via resolve_configs (no restriction concept
        there) or a call with nothing pooled."""
        labels = [label for label, _ in configs]
        if not labels:
            return ""
        restriction = self._restrictions.get(labels[0], "")
        orig = _ordered_unique(rel for label, dirs in configs if not label.startswith(PAPER_SOURCE)
                               for d in dirs if (rel := _phlag_orig_rel(d)) is not None)
        text = " — ".join(part for part in [restriction, ", ".join(orig)] if part)
        return f" — {text}" if text else ""

    def _group_configs_by_axis(self, configs, axis_name):
        """Splits `configs` into ordered groups by axis_name's own per-config
        value (self._config_axes, at the index self._agg_axis_names records
        for axis_name -- both set by resolve_configs_cartesian for every
        label produced by one call, see its docstring). axis_name is
        normalized via _strip_flag first, so a raw CLI flag token (e.g.
        "--double-variance-init") matches the stripped name
        resolve_configs_cartesian actually stored ("double-variance-init");
        a _DATA_COLUMN_AXES/_DYNAMIC_ROW_AXES name (e.g. "x_bin") is already
        unstripped and passes through as-is. A config whose agg axes don't
        include axis_name at all lands in the None group. Returns
        [(value, [(label, dirs), ...]), ...] in first-seen value order --
        used by plot's heatmap grid_by generalization to render one
        heatmap figure per group instead of cramming every config into one
        shared figure."""
        norm = _strip_flag(_canonical_flag_name(axis_name)) if axis_name in _FLAG_TO_DEST else axis_name
        order = []
        groups = {}
        for label, dirs in configs:
            names = self._agg_axis_names.get(label, [])
            vals = self._config_axes.get(label, [])
            value = vals[names.index(norm)] if norm in names else None
            if value not in groups:
                groups[value] = []
                order.append(value)
            groups[value].append((label, dirs))
        return [(v, groups[v]) for v in order]

    def _metric_by_label(self, configs, metric):
        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)
        return {
            label: dfs[label][metric].dropna().mean() if not dfs[label].empty else float("nan")
            for label, _ in configs
        }

    def _metric_by_axis_value(self, configs, axis_idx, metric, panel_col=None):
        """Mean+std `metric` for hue/line axis value `axis_idx` (e.g.
        axis_idx=0 pools by the hue/line axis a grouped plot's legend colors
        by -- resolve_configs_cartesian's agg[0], see its docstring),
        averaged in three stages -- NOT one flat mean over every row
        concatenated together, nor a flat mean of per-config means:

        1. Each (config, window_size) pair's own mean first (that config's
           dirs' rows for that window_size, via the "_window_size" column
           _load_raw_config_df attaches -- a config's dirs can span several
           window sizes when -w is a pooled, non-agg axis, see
           resolve_configs_cartesian).
        2. An unweighted mean of those (config, window_size) means across
           every config sharing the target hue value, *within* each window
           size -- "all configs in a window size" pooled first.
        3. An unweighted mean of the resulting per-window-size numbers
           across window sizes.

        A flat pool over rows or over configs would let whichever
        window/config happens to have the most surviving rows (after
        panel/NaN filtering) dominate the legend number; pooling by window
        size first, then across window sizes, gives every window size equal
        say regardless of how many configs or rows it has.

        Loads each config's dirs itself via _load_raw_config_df, deliberately
        ignoring self._row_filters (unlike every other caller of
        _load_configs_runs) -- a row-filter only ever comes from a
        _DATA_COLUMN_AXES/_DYNAMIC_ROW_AXES agg level (e.g. "x_bin",
        "fraction_bin", "em_gt_hd_bin"), which never narrows which dirs
        match (see resolve_configs_cartesian: such a level's contribution
        goes to row_filter, not agg_flags), so every config sharing a given
        hue value AND real (flag-based) bar/col/row value already has
        *identical* dirs regardless of its data-column-axis value(s) --
        row-filtering them apart before pooling would otherwise multiply the
        legend's granularity (and shift its number) every time a purely
        display-oriented row-filter axis gets added to agg, e.g. adding
        "em_gt_hd_bin"/"fraction_bin" as extra subplot col/row axes. Loading
        raw instead means those configs just contribute identical, harmless
        duplicate means (averaging duplicates doesn't change a mean), so the
        legend score stays invariant to how many display axes agg carries --
        only genuine flag-based differences (including which window sizes
        are actually present) affect it. Used to attach an aggregate score
        to each legend entry in _plot_grouped/_plot_tpr_fpr_grouped. Std (not
        IQR, matching _draw_stat's bar errorbar convention) is taken over
        every experiment row within each window size (all configs sharing the
        target value pooled), then averaged across window sizes. Returns
        {axis_value: (mean, std)}."""
        def axis_val(label):
            vals = self._config_axes.get(label, [])
            return vals[axis_idx] if axis_idx < len(vals) else None

        raw_cache = {}
        def raw_df(dirs):
            key = tuple(sorted(str(d) for d in dirs))
            if key not in raw_cache:
                raw_cache[key] = _load_raw_config_df(dirs, self._get_args, self._bin_specs)
            return raw_cache[key]

        values = {}
        for target in _ordered_unique(axis_val(label) for label, _ in configs):
            by_window = {}
            for label, dirs in configs:
                if axis_val(label) != target:
                    continue
                d = raw_df(dirs)
                if d.empty or metric not in d.columns:
                    continue
                if panel_col is not None:
                    d = d[_is_true(d[panel_col])]
                for window_size, sub in d.groupby("_window_size", dropna=False):
                    vals = sub[metric].dropna()
                    if not vals.empty:
                        means, rows = by_window.setdefault(window_size, ([], []))
                        means.append(vals.mean())
                        rows.append(vals)
            window_means = pd.Series(
                [pd.Series(ms, dtype=float).mean() for ms, _ in by_window.values()], dtype=float,
            )
            window_stds = pd.Series(
                [pd.concat(rs).std() for _, rs in by_window.values()], dtype=float,
            ).fillna(0.0)
            if window_means.empty:
                values[target] = (float("nan"), float("nan"))
            else:
                values[target] = (window_means.mean(), window_stds.mean())
        return values

    def _plot_grouped(self, configs, metrics, plot_type, title_suffix="", title="", logx=False, logy=False, interp=False,
                      agg_idx=0, errorbar="default"):
        """logx/logy/interp: see plot()'s options= docstring -- applied per
        subplot here, since each (row, col) cell can have its own
        local_bar_vals (and therefore its own logx numeric-parseability).

        agg_idx: which agg level drives bar color + the legend (default 0,
        the hue axis). Layout roles stay fixed (agg[0] still offsets bars
        within a group, agg[1..3] still bar-group/col/row) -- only coloring
        and the legend's per-value pooled score switch to agg[agg_idx]. An
        interp line (one per hue value) is drawn gray when its points span
        more than one agg[agg_idx] value.

        One figure per metric, laid out by `configs`' per-config axis
        values (self._config_axes, set by resolve_configs_cartesian's agg --
        see its docstring): axis 0 -> one bar per group (hue, shared
        legend), axis 1 -> x-position (bar groups), axis 2 -> subplot
        columns, axis 3 -> subplot rows. A config with fewer than 4 levels
        just gets a constant (None) value for the missing ones, collapsing
        that dimension to a single group/column/row -- same as an explicit
        "" placeholder in agg (see resolve_configs_cartesian's `agg`
        docstring), which also leaves a constant "(none)" value at that axis
        instead of shifting later axes into its slot. When axis 0 (hue) is
        specifically an explicit "(none)" placeholder while axis 1 (bar) has
        real values, there is nothing meaningful to legend by, so the legend
        collapses to a single pooled entry -- the metric averaged over EVERY
        config in the figure (there's no real hue axis left to break it down
        by) -- rather than the usual per-hue-value breakdown; a hue axis that
        merely collapsed to one real (non-placeholder) value on disk keeps
        the normal single-value/colored-hue/legend rendering. Each (row,
        col) subplot only shows the bar categories that actually have data
        for it, not the full bar-value set pooled across every subplot --
        e.g. a "low"/"high" x_bin bar only appears in an admixture column,
        not a recombination one, instead of showing as an empty tick there
        too. The legend's score per hue value is otherwise pooled across
        every *real* (flag-based) bar/col/row value sharing that hue value
        (agg[0]), not just one cell -- but is invariant to any row-filter
        (_DATA_COLUMN_AXES/_DYNAMIC_ROW_AXES, e.g. "x_bin", "fraction_bin",
        "em_gt_hd_bin") axis in agg, since those only ever split the same
        underlying dirs into more display cells without changing what's
        actually being averaged -- see _metric_by_axis_value.

        If every element of `metrics` is instead a same-length tuple, this
        takes a completely different path (see the `tuple_metrics` branch
        below): axis 2/3 (subplot column/row) must both be unused (agg must
        resolve to exactly the 2 real hue/bar-position levels) since
        `metrics` itself fills that grid role instead -- one row per
        `metrics` list entry, one column per tuple position, all in ONE
        shared figure instead of one figure per metric. Each cell's hue
        legend then just names the hue values (no per-hue aggregated score
        annotation, unlike the scalar-metric path's hue_legend_labels --
        there's no single unambiguous metric to score by across a whole
        row/column of different metrics)."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        hue_vals = self._axis_values(configs, 0)
        bar_vals = self._axis_values(configs, 1)
        col_vals = self._axis_values(configs, 2)
        row_vals = self._axis_values(configs, 3)
        agg_names = self._agg_axis_names.get(configs[0][0], []) if configs else []
        bar_axis_label = _axis_flag_label(agg_names[1]) if len(agg_names) > 1 else ""

        hue_idx, bar_idx = 0, 1
        leg_vals = self._axis_values(configs, agg_idx)
        pooled_legend = leg_vals == ["(none)"] and bar_vals != ["(none)"]

        by_key = {}
        for label, dirs in configs:
            key = (axis_val(label, 3), axis_val(label, 2), axis_val(label, bar_idx), axis_val(label, hue_idx))
            by_key[key] = (label, dirs)

        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)
        # Bar categories with no data at all get dropped everywhere below,
        # but which ones are non-empty can differ per (row, col) subplot
        # (e.g. a "low"/"high" x_bin only has rows for an admixture column,
        # not a recombination one) -- so each subplot gets its own x-axis
        # (local_bar_vals) instead of every subplot sharing one global list
        # that would otherwise leave blank ticks in subplots with no data
        # for that category.
        local_bar_vals = {
            (rv, cv): [
                bv for bv in bar_vals
                if any(
                    (rv, cv, bv, hv) in by_key and not dfs[by_key[(rv, cv, bv, hv)][0]].empty
                    for hv in hue_vals
                )
            ]
            for rv in row_vals for cv in col_vals
        }

        colors = {lv: _config_color(i) for i, lv in enumerate(leg_vals)}
        color_of = {label: colors[axis_val(label, agg_idx)] for label, _ in configs}

        def interp_color(hv, pts):
            cs = {c for _, _, c in pts}
            return cs.pop() if len(cs) == 1 else "#888888"

        n_rows, n_cols = len(row_vals), len(col_vals)
        bar_width = 0.82 / max(len(hue_vals), 1)

        figs = []
        tuple_metrics = any(isinstance(m, tuple) for m in metrics)
        if tuple_metrics:
            assert all(isinstance(m, tuple) for m in metrics), (
                "mixing tuple and scalar metrics in one plot() call isn't supported here -- "
                f"got {metrics!r}"
            )
            tuple_len = len(metrics[0])
            assert all(len(m) == tuple_len for m in metrics), (
                f"all tuple metrics must be the same length to form a rectangular grid -- got {metrics!r}"
            )
            assert n_rows == 1 and n_cols == 1, (
                f"tuple metrics need agg to resolve to exactly 2 real levels (hue + bar-position) -- "
                f"they fill the subplot row/column grid role themselves (one row per metrics list "
                f"entry, one column per tuple position), which conflicts with agg's own "
                f"{n_rows}x{n_cols} already-real subplot row/column levels here"
            )
            rv0, cv0 = row_vals[0], col_vals[0]
            cell_bar_vals = local_bar_vals[(rv0, cv0)]
            bar_nums = [_token_to_number(bv) for bv in cell_bar_vals]
            use_logx = logx and len(cell_bar_vals) > 0 and all(v is not None and v > 0 for v in bar_nums)
            positions = bar_nums if use_logx else list(range(len(cell_bar_vals)))

            grid_rows, grid_cols = len(metrics), tuple_len
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3.2 * grid_cols, 3.6 * grid_rows), squeeze=False)
            for ri, panel in enumerate(metrics):
                for ci, this_metric in enumerate(panel):
                    ax = axes[ri][ci]
                    panel_col = _METRIC_PANEL_COLUMN.get(this_metric)
                    if use_logx:
                        ax.set_xscale("log")
                        ax.set_xticks(positions)
                        ax.set_xticklabels(cell_bar_vals, rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
                        ax.set_xlim(min(positions) / 1.5, max(positions) * 1.5)
                    else:
                        ax.set_xlim(-0.7, len(cell_bar_vals) - 0.3)
                        ax.set_xticks(range(len(cell_bar_vals)))
                        ax.set_xticklabels(cell_bar_vals, rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
                    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                    ax.set_axisbelow(True)
                    bounded = False
                    if _resolve_logy(logy, dfs, this_metric):
                        ax.set_yscale("log")
                    else:
                        bounded = _apply_bounded_yaxis(ax, this_metric)
                    ax.set_title(this_metric, fontsize=9)
                    cell_bounds = []
                    interp_points = {hv: [] for hv in hue_vals}
                    for bi, bv in enumerate(cell_bar_vals):
                        pos = positions[bi]
                        for hi, hv in enumerate(hue_vals):
                            entry = by_key.get((rv0, cv0, bv, hv))
                            if entry is None:
                                continue
                            label, _ = entry
                            d = dfs[label]
                            if d.empty or this_metric not in d.columns:
                                continue
                            if panel_col is not None:
                                d = d[_is_true(d[panel_col])]
                            frac = (hi - (len(hue_vals) - 1) / 2) / max(len(hue_vals), 1)
                            x = pos * (1 + frac * _LOG_SPAN) if use_logx else bi + frac * len(hue_vals) * bar_width
                            w = pos * (_LOG_SPAN / max(len(hue_vals), 1)) if use_logx else bar_width
                            if self._agg_given:
                                _assert_single_run(d)
                            cell_bounds.append(_draw_stat(ax, x, d[this_metric], color_of[label], w, plot_type, errorbar))
                            if interp:
                                mean = d[this_metric].dropna().mean()
                                if not pd.isna(mean):
                                    interp_points[hv].append((pos, mean, color_of[label]))
                    if plot_type == "violin" and not bounded:
                        _clip_axis_to_whiskers(ax, cell_bounds)
                    if interp:
                        for hv in hue_vals:
                            pts = sorted(interp_points[hv], key=lambda p: p[0])
                            if len(pts) > 1:
                                xs, ys, _ = zip(*pts)
                                ax.plot(xs, ys, "-", color=interp_color(hv, pts), linewidth=1.3, alpha=0.85, zorder=4)
            _suptitle(fig, _compose_title(title_suffix, title), fontsize=12, y=0.99)
            if bar_axis_label and grid_rows * grid_cols == 1:
                fig.supxlabel(bar_axis_label, fontsize=9)
            if pooled_legend or len(leg_vals) > 1 or leg_vals not in ([None], ["(none)"]):
                handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[lv]) for lv in leg_vals]
                legend = fig.legend(handles, [str(lv) for lv in leg_vals], loc="center left",
                                    bbox_to_anchor=(1.0, _title_bottom(fig) / 2), fontsize=8, framealpha=0.9)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            figs.append(fig)
            return figs

        for metric in metrics:
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            cell_w = min(16.0, max(3.2, 0.07 * len(hue_vals) * max((len(v) for v in local_bar_vals.values()), default=1)))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_w * n_cols, 3.6 * n_rows), squeeze=False)
            for ri, rv in enumerate(row_vals):
                for ci, cv in enumerate(col_vals):
                    ax = axes[ri][ci]
                    cell_bar_vals = local_bar_vals[(rv, cv)]
                    bar_nums = [_token_to_number(bv) for bv in cell_bar_vals]
                    use_logx = logx and len(cell_bar_vals) > 0 and all(v is not None and v > 0 for v in bar_nums)
                    positions = bar_nums if use_logx else list(range(len(cell_bar_vals)))
                    if use_logx:
                        ax.set_xscale("log")
                        ax.set_xticks(positions)
                        ax.set_xticklabels(cell_bar_vals, rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
                        ax.set_xlim(min(positions) / 1.5, max(positions) * 1.5)
                    else:
                        ax.set_xlim(-0.7, len(cell_bar_vals) - 0.3)
                        ax.set_xticks(range(len(cell_bar_vals)))
                        ax.set_xticklabels(cell_bar_vals, rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
                    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                    ax.set_axisbelow(True)
                    bounded = False
                    if _resolve_logy(logy, dfs, metric):
                        ax.set_yscale("log")
                    else:
                        bounded = _apply_bounded_yaxis(ax, metric)
                    if ri == 0 and cv not in (None, "(none)"):
                        ax.set_title(self._axis_title(configs, 2, cv), fontsize=9)
                    if ci == 0 and rv not in (None, "(none)"):
                        ax.set_ylabel(self._axis_title(configs, 3, rv), fontsize=9)
                    cell_bounds = []
                    interp_points = {hv: [] for hv in hue_vals}
                    for bi, bv in enumerate(cell_bar_vals):
                        pos = positions[bi]
                        for hi, hv in enumerate(hue_vals):
                            entry = by_key.get((rv, cv, bv, hv))
                            if entry is None:
                                continue
                            label, _ = entry
                            d = dfs[label]
                            # metric can be entirely absent from d's columns (not just
                            # NaN-valued) when every one of this config's dirs' runs.tsv
                            # predates it -- see _plot_heatmap's identical guard.
                            if d.empty or metric not in d.columns:
                                continue
                            if panel_col is not None:
                                d = d[_is_true(d[panel_col])]
                            frac = (hi - (len(hue_vals) - 1) / 2) / max(len(hue_vals), 1)
                            x = pos * (1 + frac * _LOG_SPAN) if use_logx else bi + frac * len(hue_vals) * bar_width
                            w = pos * (_LOG_SPAN / max(len(hue_vals), 1)) if use_logx else bar_width
                            if self._agg_given:
                                _assert_single_run(d)
                            cell_bounds.append(_draw_stat(ax, x, d[metric], color_of[label], w, plot_type, errorbar))
                            if interp:
                                mean = d[metric].dropna().mean()
                                if not pd.isna(mean):
                                    interp_points[hv].append((pos, mean, color_of[label]))
                    if plot_type == "violin" and not bounded:
                        _clip_axis_to_whiskers(ax, cell_bounds)
                    if interp:
                        for hv in hue_vals:
                            pts = sorted(interp_points[hv], key=lambda p: p[0])
                            if len(pts) > 1:
                                xs, ys, _ = zip(*pts)
                                ax.plot(xs, ys, "-", color=interp_color(hv, pts), linewidth=1.3, alpha=0.85, zorder=4)
            _suptitle(fig, _compose_title(title_suffix, title, metric if len(metrics) > 1 else ""), fontsize=12, y=0.99)
            # A single subplot's bar categories aren't otherwise labeled
            # anywhere, so the shared axis label carries real information
            # there -- but a multi-subplot grid repeats the same bar
            # categories (and now, since _combo_label_part canonicalizes bool
            # flags to their long form, self-explanatory tick labels like
            # "ilr"/"(base)") in every column, making one more copy centered
            # under the whole figure redundant.
            if bar_axis_label and n_rows * n_cols == 1:
                fig.supxlabel(bar_axis_label, fontsize=9)
            fig.supylabel(metric, fontsize=9)
            if pooled_legend or len(leg_vals) > 1 or leg_vals not in ([None], ["(none)"]):
                hue_metric_values = self._metric_by_axis_value(configs, agg_idx, metric, panel_col)
                legend_hue_vals = [hv for hv in leg_vals if not pd.isna(hue_metric_values[hv][0])]
                handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[hv]) for hv in legend_hue_vals]
                if pooled_legend:
                    hue_legend_labels = [
                        f"all ({metric}={hue_metric_values[hv][0]:.4f}±{hue_metric_values[hv][1]:.4f})"
                        for hv in legend_hue_vals
                    ]
                else:
                    hue_legend_labels = [
                        f"{hv} ({metric}={hue_metric_values[hv][0]:.4f}±{hue_metric_values[hv][1]:.4f})"
                        for hv in legend_hue_vals
                    ]
                # Outside the axes to the right, not stacked above them --
                # a long hue sweep (e.g. every -w value) makes a tall legend
                # that used to push down into a fixed 15%-of-height top
                # reservation regardless of how many entries there were,
                # overlapping the axes. Jupyter's inline backend renders with
                # bbox_inches="tight" by default, so a legend anchored past
                # the figure's right edge just grows the saved/displayed
                # canvas sideways instead of colliding with anything.
                legend = fig.legend(handles, hue_legend_labels, loc="center left",
                                    bbox_to_anchor=(1.0, _title_bottom(fig) / 2), fontsize=8, framealpha=0.9)
            _fit_fig_to_axes(fig, axes, cell_w - 0.8, 2.6, rect=[0, 0, 1, 0.95])
            figs.append(fig)
        return figs

    def _plot_beta_line(self, configs, metrics, title_suffix="", logx=True, logy=False, errorbar="default", color=None):
        bi = self._beta_axis_idx
        if bi is None:
            raise ValueError('plot_type="line" needs a ("--beta", "--beta-prime") tuple in agg')

        def layout_vals(label):
            vals = [v for i, v in enumerate(self._config_axes.get(label, [])) if i != bi] + [None] * 3
            implicit = self._config_implicit.get(label) if bi == 0 else ""
            if implicit:
                vals[0] = implicit if vals[0] in (None, "(none)", "(base)") else f"{vals[0]} {implicit}"
            return tuple(vals[:3])

        layout = {label: layout_vals(label) for label, _ in configs}
        hue_vals = _ordered_unique(h for h, _, _ in layout.values())
        col_vals = sorted(_ordered_unique(c for _, c, _ in layout.values()), key=_token_sort_key)
        row_vals = sorted(_ordered_unique(r for _, _, r in layout.values()), key=_token_sort_key)
        if color is None:
            colors = {hv: _config_color(i) for i, hv in enumerate(hue_vals)}
        else:
            colors = {hv: _gradient_color(color, i / max(len(hue_vals) - 1, 1)) for i, hv in enumerate(hue_vals)}
        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)
        windowless_roots = {PHLAG_SOURCE_ROOTS[x] for x in WINDOWLESS_PHLAG_SOURCES}
        is_dm = {
            label: bool(dirs) and all(
                str(d) not in windowless_roots and bool(self._get_args(d).get("dirichlet_mean")) for d in dirs
            )
            for label, dirs in configs
        }
        dm_vals = sorted(set(is_dm.values()))
        split_dm = len(dm_vals) > 1

        def line_color(hv, dm):
            return _shade_color(colors[hv], 0.7 if dm else 0.1) if split_dm else colors[hv]

        figs = []
        for metric in metrics:
            assert isinstance(metric, str), f'plot_type="line" takes scalar metrics, got {metric!r}'
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            points, baselines = {}, {}
            for label, _ in configs:
                d = dfs[label]
                if d.empty or metric not in d.columns:
                    continue
                if panel_col is not None:
                    d = d[_is_true(d[panel_col])]
                if self._agg_given:
                    _assert_single_run(d)
                vals = d[metric].dropna()
                if vals.empty:
                    continue
                hv, cv, rv = layout[label]
                hv = (hv, is_dm[label])
                stat = (vals.mean(), 0.0 if errorbar is None else _spread(vals, "sd" if errorbar == "default" else errorbar))
                beta = self._config_beta.get(label)
                if beta is None:
                    baselines[(rv, cv, hv)] = stat
                else:
                    points.setdefault((rv, cv, hv), []).append((beta, *stat))
            n_rows, n_cols = len(row_vals), len(col_vals)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows), squeeze=False)
            for ri, rv in enumerate(row_vals):
                for ci, cv in enumerate(col_vals):
                    ax = axes[ri][ci]
                    line_keys = [(hv, dm) for hv in hue_vals for dm in dm_vals]
                    xs_all = [p[0] for hv in line_keys for p in points.get((rv, cv, hv), [])]
                    if logx and xs_all and min(xs_all) > 0:
                        ax.set_xscale("log")
                    if _resolve_logy(logy, dfs, metric):
                        ax.set_yscale("log")
                    else:
                        _apply_bounded_yaxis(ax, metric)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.set_axisbelow(True)
                    for hv in line_keys:
                        pts = sorted(points.get((rv, cv, hv), []))
                        if pts:
                            xs, ys, es = zip(*pts)
                            ax.errorbar(xs, ys, yerr=None if errorbar is None else es, fmt="-o",
                                        markersize=2.5, color=line_color(*hv),
                                        linewidth=1.3, elinewidth=0.8, capsize=2, zorder=3)
                        if (rv, cv, hv) in baselines:
                            ax.axhline(baselines[(rv, cv, hv)][0], color=line_color(*hv), linestyle=":",
                                       linewidth=1.2, alpha=0.8, zorder=2)
                    if ri == 0 and cv not in (None, "(none)"):
                        ax.set_title(self._axis_title(configs, 1, cv, skip=bi), fontsize=9)
                    if ci == 0 and rv not in (None, "(none)"):
                        ax.set_ylabel(self._axis_title(configs, 2, rv, skip=bi), fontsize=9)
            _suptitle(fig, _compose_title(title_suffix, metric if len(metrics) > 1 else ""), fontsize=12, y=0.99)
            fig.supxlabel("beta (raw; beta' x n_windows)", fontsize=9)
            fig.supylabel(metric, fontsize=9)
            if split_dm or len(hue_vals) > 1 or hue_vals not in ([None], ["(none)"]):
                handles = [matplotlib.lines.Line2D([], [], color=colors[hv]) for hv in hue_vals]
                labels = [str(hv) for hv in hue_vals]
                if split_dm:
                    handles += [matplotlib.lines.Line2D([], [], color=_shade_color("#808080", f)) for f in (0.7, 0.1)]
                    labels += ["dirichlet-mean", "dirichlet-mode"]
                if baselines:
                    handles.append(matplotlib.lines.Line2D([], [], color="black", linestyle=":"))
                    labels.append("no prior")
                fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, _title_bottom(fig) / 2),
                           fontsize=8, framealpha=0.9)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            figs.append(fig)
        return figs

    def _plot_heatmap(self, configs, metrics, title_suffix="", scale=(False, False)):
        """Cells colored by metric's mean value on an x (agg[0]) by y
        (agg[1]) grid; a 3rd agg axis (agg[2]) splits into subplot columns,
        a 4th (agg[3]) into subplot rows -- same agg[0..3] role layout as
        _plot_grouped's bar/hue/col/row, just x/y instead of bar/hue.

        With multiple `metrics` and agg using exactly 2 or 3 real levels
        (agg[0]/agg[1] for x/y, optionally agg[2] for columns), `metrics`
        fills the next role after agg's last one instead of each metric
        getting its own separate figure -- agg with 2 real levels puts one
        metric per subplot COLUMN, agg with 3 real levels puts one metric
        per subplot ROW (agg[3] itself, if given as a real axis, would take
        priority over metrics for that role -- but agg[3] only ever exists
        when agg already has 4 real levels, so this never conflicts with
        agg's own 3rd/4th levels). With 4 real agg levels already filling
        every role, or a single metric, each metric gets its own figure as
        before. All of one figure's subplots share one color scale so
        shading is comparable across e.g. merged_category columns, or
        across metrics when metrics fills a role.

        x and y are scaled independently (see _heatmap_axis): an axis whose
        tokens all parse as numbers (_token_to_number, e.g. "w50k", "25%")
        is placed at its real values, cell size proportional to the gap --
        log when positive and spanning >20x, else linear; scale=(logx, logy)
        forces either (None = auto). A non-numeric axis (em_hd_bin/
        merged_category labels) stays an evenly-spaced categorical grid.

        A metric name ending in "_signed" (e.g. "f1_signed" for the real
        column "f1" -- see _signed_base_metric) renders that real metric as
        usual, then subtracts its "null" column's grid from every OTHER
        column's grid, cell by cell, within the same row -- i.e. agg[2]
        (the 3rd agg axis, which this method puts on subplot columns) gets
        treated as "baseline vs. variants": the null/baseline column is
        whichever col_vals entry is a flag's off/negated or missing
        ("(base)") display state if one is present, else
        col_vals[0] (its first/leftmost value, in whatever order `agg`
        or _discover_flag_tuple_values' sort produced -- e.g. the "no
        special flags" combo for a tuple axis) -- see _null_col_val. The
        baseline column ITSELF keeps its raw (undiffed) value -- diffing it
        against itself would just be a trivial all-0 column -- and renders
        on its own ordinary sequential "Blues" scale/colorbar (shared with any
        cell whose row isn't signed at all), while every diffed cell renders
        on its own zero-centered diverging "RdBu" scale/colorbar, so blue
        (positive, above baseline) vs. red (negative, below baseline) reads
        at a glance without washing out the baseline's own real magnitude.
        Requires agg[2] to be a real axis (not
        already filled by `metrics` itself, i.e. metric_role != 2 below) --
        asserts otherwise, since there'd be no baseline column to diff
        against."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        x_vals = _ordered_unique(axis_val(label, 0) for label, _ in configs)
        y_vals = _ordered_unique(axis_val(label, 1) for label, _ in configs)
        col_vals = _ordered_unique(axis_val(label, 2) for label, _ in configs)
        row_vals = _ordered_unique(axis_val(label, 3) for label, _ in configs)
        agg_names = self._agg_axis_names.get(configs[0][0], []) if configs else []
        x_axis_label = _axis_flag_label(agg_names[0]) if len(agg_names) > 0 else ""
        y_axis_label = _axis_flag_label(agg_names[1]) if len(agg_names) > 1 else ""
        vs_window = any("window size" in _axis_flag_label(n) for n in agg_names[:2])
        x_tick_fmt = _outlier_sites_label if vs_window and agg_names[:1] == ["outlier_pct"] else str
        y_tick_fmt = _outlier_sites_label if vs_window and agg_names[1:2] == ["outlier_pct"] else str
        if x_tick_fmt is not str:
            x_axis_label = "outlier sites (%)"
        if y_tick_fmt is not str:
            y_axis_label = "outlier sites (%)"

        n_agg_levels = max((len(self._config_axes.get(label, [])) for label, _ in configs), default=0)
        metric_role = n_agg_levels if len(metrics) > 1 and n_agg_levels in (2, 3) else None
        if metric_role == 2:
            col_vals = list(metrics)
        elif metric_role == 3:
            row_vals = list(metrics)

        by_key = {}
        for label, dirs in configs:
            key = (axis_val(label, 3), axis_val(label, 2), axis_val(label, 1), axis_val(label, 0))
            by_key[key] = (label, dirs)

        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)

        x_vals, x_edges, tick_x, x_scale = _heatmap_axis(x_vals, scale[0])
        y_vals, y_edges, tick_y, y_scale = _heatmap_axis(y_vals, scale[1])

        n_rows, n_cols = len(row_vals), len(col_vals)
        outer_metrics = [None] if metric_role else metrics

        figs = []
        for outer_metric in outer_metrics:
            grids = {}
            signed = {}
            for rv in row_vals:
                for cv in col_vals:
                    cell_metric = rv if metric_role == 3 else (cv if metric_role == 2 else outer_metric)
                    base_metric = _signed_base_metric(cell_metric)
                    if base_metric is not None:
                        assert metric_role != 2, (
                            f"{cell_metric!r}: a signed metric needs agg[2] as real subplot columns to "
                            "diff against a baseline column, but metrics itself fills that role here"
                        )
                    real_metric = base_metric or cell_metric
                    signed[(rv, cv)] = base_metric is not None
                    lookup_rv = None if metric_role == 3 else rv
                    lookup_cv = None if metric_role == 2 else cv
                    panel_col = _METRIC_PANEL_COLUMN.get(real_metric)
                    grid = [[float("nan")] * len(x_vals) for _ in y_vals]
                    for yi, yv in enumerate(y_vals):
                        for xi, xv in enumerate(x_vals):
                            entry = by_key.get((lookup_rv, lookup_cv, yv, xv))
                            if entry is None:
                                continue
                            label, _ = entry
                            d = dfs[label]
                            if panel_col is not None and not d.empty:
                                d = d[_is_true(d[panel_col])]
                            # real_metric can be entirely absent from d's columns, not just
                            # NaN-valued -- pd.concat drops a column that's missing from
                            # every one of a config's dirs' runs.tsv (e.g. legacy runs
                            # predating null_mean_*/alt_mean_*, see CLAUDE.md's caster
                            # scores note), so indexing d[real_metric] unconditionally
                            # raises KeyError instead of just having no data to average.
                            grid[yi][xi] = (
                                d[real_metric].dropna().mean()
                                if not d.empty and real_metric in d.columns else float("nan")
                            )
                    grids[(rv, cv)] = grid
            any_signed = any(signed.values())
            is_diff = {}
            if any_signed:
                baseline_cv = _null_col_val(col_vals)
                for rv in row_vals:
                    base_grid = grids[(rv, baseline_cv)]
                    for cv in col_vals:
                        is_diff[(rv, cv)] = signed[(rv, cv)] and cv != baseline_cv
                        if not is_diff[(rv, cv)]:
                            continue
                        grid = grids[(rv, cv)]
                        grids[(rv, cv)] = [
                            [
                                (grid[yi][xi] - base_grid[yi][xi])
                                if not (pd.isna(grid[yi][xi]) or pd.isna(base_grid[yi][xi])) else float("nan")
                                for xi in range(len(x_vals))
                            ]
                            for yi in range(len(y_vals))
                        ]
            # A signed metric's baseline column keeps its raw (undiffed) value -- it
            # shares the ordinary sequential "Blues" scale (raw_vmin/vmax) with any
            # genuinely-unsigned cell, while every diffed cell gets its own
            # zero-centered diverging "RdBu" scale (diff_vmin/vmax) -- two separate
            # meshes/colorbars per figure instead of cramming both kinds of value
            # onto one scale (which would either wash out the baseline's real
            # magnitude or wreck the diff scale's zero-centering).
            diff_vals = [
                v for (rv, cv), grid in grids.items() if is_diff.get((rv, cv))
                for row in grid for v in row if not pd.isna(v)
            ]
            raw_vals = [
                v for (rv, cv), grid in grids.items() if not is_diff.get((rv, cv))
                for row in grid for v in row if not pd.isna(v)
            ]
            diff_vmin = diff_vmax = None
            if diff_vals:
                vabs = max(abs(v) for v in diff_vals) or 1.0
                diff_vmin, diff_vmax = -vabs, vabs
            raw_vmin, raw_vmax = (min(raw_vals), max(raw_vals)) if raw_vals else (0.0, 1.0)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(max(3.2, 0.55 * len(x_vals) + 1.5) * n_cols, max(3.0, 0.45 * len(y_vals) + 1.5) * n_rows),
                squeeze=False,
            )
            diff_mesh = raw_mesh = None
            for ri, rv in enumerate(row_vals):
                for ci, cv in enumerate(col_vals):
                    ax = axes[ri][ci]
                    grid = grids[(rv, cv)]
                    if is_diff.get((rv, cv)):
                        cmap, vmin, vmax = "RdBu", diff_vmin, diff_vmax
                    else:
                        cmap, vmin, vmax = "Blues", raw_vmin, raw_vmax
                    ax.set_xscale(x_scale)
                    ax.set_yscale(y_scale)
                    mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
                    if is_diff.get((rv, cv)):
                        diff_mesh = mesh
                    else:
                        raw_mesh = mesh
                    ax.set_xticks(tick_x)
                    ax.set_xticklabels([x_tick_fmt(v) for v in x_vals], rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
                    ax.set_yticks(tick_y)
                    # Only the leftmost column in each row needs y-tick text --
                    # every subplot got its own labels before, which rendered
                    # on top of the previous (left-neighbor) subplot's data
                    # cells once panels were packed close together.
                    if ci == 0:
                        ax.set_yticklabels([y_tick_fmt(v) for v in y_vals], fontsize=7.5)
                    else:
                        ax.set_yticklabels([])
                    for yi, yn in enumerate(tick_y):
                        for xi, xn in enumerate(tick_x):
                            v = grid[yi][xi]
                            if not pd.isna(v):
                                color = "white" if (v - vmin) / (vmax - vmin or 1) > 0.6 else "black"
                                ax.text(xn, yn, f"{v:.3f}", ha="center", va="center", color=color, fontsize=7)
                    if ri == 0 and cv not in (None, "(none)"):
                        ax.set_title(_row_axis_title(agg_names[2] if len(agg_names) > 2 else None, cv), fontsize=9.5)
                    if ci == 0 and rv not in (None, "(none)"):
                        ax.set_ylabel(_row_axis_title(agg_names[3] if len(agg_names) > 3 else None, rv), fontsize=9, labelpad=28)
            title_metric = "/".join(metrics) if metric_role else outer_metric
            base_label = "value" if metric_role else outer_metric
            multi_metric = metric_role is None and len(metrics) > 1
            _suptitle(fig, _compose_title(title_suffix, title_metric if multi_metric else "", "heatmap"), fontsize=12)
            fig.subplots_adjust(top=_title_bottom(fig, 0.88))
            if x_axis_label:
                fig.supxlabel(x_axis_label, fontsize=9)
            if y_axis_label:
                fig.supylabel(y_axis_label, fontsize=9)
            all_axes = list(axes.flat)
            if diff_mesh is None:
                # Ordinary (unsigned) heatmap, or a signed one whose baseline
                # column is the only thing being shown -- one colorbar, right
                # side, exactly as before.
                fig.subplots_adjust(left=max(0.12, 0.9 / fig.get_figwidth()), bottom=0.2, right=0.88, top=_title_bottom(fig, 0.88), wspace=0.3, hspace=0.35)
                if raw_mesh is not None:
                    fig.colorbar(raw_mesh, ax=all_axes, label=base_label, fraction=0.05, pad=0.02, location="right")
            else:
                # Split, rather than stack, when both are present: the diff
                # colorbar keeps the usual right side (unaffected below), the
                # raw/baseline one goes on the left -- stacking both on the same
                # side left no room for the row-label text and y-tick bin
                # labels between them and the subplot grid.
                # A fixed left pad guess doesn't scale to whatever the leftmost
                # column's y-tick labels actually render as -- a bin-range
                # label like "(0.833,1.000]" is much wider than a short numeric
                # one, and the row's own ylabel (labelpad=28) adds more on top
                # of that. So reserve a generous baseline margin first, THEN
                # measure the leftmost column's REAL rendered extent (tick
                # labels + ylabel, via get_tightbbox) once that layout is
                # final, and place the colorbar's own axes (via add_axes, which
                # -- unlike ax=/location=, doesn't reflow other axes) snug
                # against whatever that measured extent turns out to be.
                fig.subplots_adjust(left=0.32, bottom=0.2, right=0.85, top=_title_bottom(fig, 0.88), wspace=0.3, hspace=0.35)
                fig.colorbar(diff_mesh, ax=all_axes, label=f"Δ {base_label}", fraction=0.05, pad=0.02, location="right")
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                fig_w, fig_h = fig.bbox.width, fig.bbox.height
                left_col_axes = [axes[ri][0] for ri in range(n_rows)]
                label_x0 = min(ax.get_tightbbox(renderer).x0 for ax in left_col_axes)
                top = axes[0][0].get_window_extent(renderer).y1
                bottom = axes[-1][0].get_window_extent(renderer).y0
                gap, cbar_width = 0.012 * fig_w, 0.02 * fig_w
                cbar_x1 = label_x0 - gap
                cbar_x0 = max(0.01 * fig_w, cbar_x1 - cbar_width)
                cax = fig.add_axes([cbar_x0 / fig_w, bottom / fig_h, (cbar_x1 - cbar_x0) / fig_w, (top - bottom) / fig_h])
                fig.colorbar(raw_mesh, cax=cax, label=f"{base_label} (baseline)")
                cax.yaxis.set_ticks_position("left")
                cax.yaxis.set_label_position("left")
            figs.append(fig)
        return figs

    def _plot_heatmap_hd_bin(self, configs, metrics, title_suffix=""):
        """One figure per metric, one subplot per config: cells colored by
        that metric's mean over a hd_bin (columns) x fraction_bin (rows)
        grid -- hd_bin recomputed fresh across all configs' pooled em_gt_hd
        values, same as plot's grid_by="hd_bin" bar/violin path (see
        _compute_hd_bin_edges). All of a figure's subplots share one color
        scale (min/max across every config's grid) so e.g. a dvi-on vs
        dvi-off pair of subplots is visually comparable."""
        dfs = _load_configs_runs(configs, self._row_filters, self._bin_specs, self._get_args)
        all_gt_hd = [v for d in dfs.values() if not d.empty for v in d["em_gt_hd"].dropna().tolist()]
        edges = _compute_hd_bin_edges(all_gt_hd)
        hd_bins = [label for label, _, _ in edges]
        hd_dfs = {
            label: (d.assign(_hd_bin=d["em_gt_hd"].apply(lambda v: assign_bin(v, edges))) if not d.empty else d)
            for label, d in dfs.items()
        }
        n_configs = len(configs)
        n_rows, n_cols = len(FRACTION_BINS), len(hd_bins)

        figs = []
        for metric in metrics:
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            grids = []
            for label, _ in configs:
                d = hd_dfs[label]
                grid = [[float("nan")] * n_cols for _ in range(n_rows)]
                for ri, fraction_bin in enumerate(FRACTION_BINS):
                    for ci, hd_bin in enumerate(hd_bins):
                        dd = d[(d["fraction_bin"] == fraction_bin) & (d["_hd_bin"] == hd_bin)] if not d.empty else d
                        if panel_col is not None and not dd.empty:
                            dd = dd[_is_true(dd[panel_col])]
                        grid[ri][ci] = dd[metric].dropna().mean() if not dd.empty else float("nan")
                grids.append(grid)
            flat = [v for g in grids for row in g for v in row if not pd.isna(v)]
            vmin, vmax = (min(flat), max(flat)) if flat else (0.0, 1.0)

            fig, axes = plt.subplots(1, n_configs, figsize=(2.6 * n_configs + 1.0, 3.6), squeeze=False)
            mesh = None
            for idx, (label, _) in enumerate(configs):
                ax = axes[0][idx]
                mesh = ax.pcolormesh(grids[idx], cmap="Blues", vmin=vmin, vmax=vmax, shading="flat")
                ax.set_xticks([i + 0.5 for i in range(n_cols)])
                ax.set_xticklabels(hd_bins, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.5)
                if idx == 0:
                    ax.set_yticks([i + 0.5 for i in range(n_rows)])
                    ax.set_yticklabels(FRACTION_BINS, fontsize=7.5)
                else:
                    ax.set_yticks([])
                ax.set_title(label, fontsize=9)
                for ri in range(n_rows):
                    for ci in range(n_cols):
                        v = grids[idx][ri][ci]
                        if not pd.isna(v):
                            color = "white" if (v - vmin) / (vmax - vmin or 1) > 0.6 else "black"
                            ax.text(ci + 0.5, ri + 0.5, f"{v:.3f}", ha="center", va="center",
                                     color=color, fontsize=6.5)
            metric_part = metric if len(metrics) > 1 else ""
            _suptitle(fig, _compose_title(title_suffix, metric_part, "by HD bin x alt proportion"), fontsize=11, y=1.0)
            fig.text(0.5, 0.01, "HD bin (em_gt_hd)", ha="center", fontsize=8.5)
            fig.text(0.005, 0.5, "alt proportion", va="center", rotation="vertical", fontsize=8.5)
            fig.colorbar(mesh, ax=axes[0].tolist(), label=metric, fraction=0.05, pad=0.02)
            fig.subplots_adjust(left=0.09, bottom=0.28, right=0.9, top=0.85, wspace=0.15)
            figs.append(fig)
        return figs


# ---- module-private helpers: pure functions, no instance state needed ----

OUT_RUN_ID_COLUMNS = ["report_path", "config", "mode", "window", "step"]

_OUT_RUN_METRIC_KEYS = {
    "tpr": "tpr", "fpr": "fpr", "precision": "precision", "f1": "f1", "accuracy": "accuracy",
    "roc_auc": "auc", "em_hd": "em_hd", "em_gt_hd": "em_gt_hd", "bic": "bic",
    "log_likelihood": "log_likelihood", "n_windows": "n_windows", "branch_length": "branch_length",
    **{f"{prefix}_transition_{move}": f"{key}_{move}"
       for prefix, key in (("fitted", "transition"), ("gt", "gt_transition"), ("viterbi", "viterbi_transition"))
       for move in ("null_to_null", "null_to_alt", "alt_to_null", "alt_to_alt")},
}
_OUT_GT_STATS_LABELS = (("Null", "null"), ("Alt", "alt"), ("Overall", "pooled"))
_OUT_GT_STATS_COLUMNS = [
    f"{prefix}_{stat}"
    for _, prefix in _OUT_GT_STATS_LABELS
    for stat in (
        ["mean_norm", "cov_norm"]
        + [f"mean_{t}" for t in TOPOLOGY_NAMES]
        + [f"var_{t}" for t in TOPOLOGY_NAMES]
        + [f"cov_{TOPOLOGY_NAMES[i]}_{TOPOLOGY_NAMES[j]}" for i in range(3) for j in range(i + 1, 3)]
        + [f"within_mean_{t}" for t in TOPOLOGY_NAMES]
        + [f"within_var_{t}" for t in TOPOLOGY_NAMES]
    )
]
_OUT_TRANSITION_TOTAL_COLUMNS = [f"{prefix}_transition_total" for prefix in ("fitted", "gt", "viterbi")]
_OUT_RUN_COLUMNS = OUT_RUN_ID_COLUMNS + list(_OUT_RUN_METRIC_KEYS) + _OUT_TRANSITION_TOTAL_COLUMNS + _OUT_GT_STATS_COLUMNS


_ECDF_TOPO_COLORS = {"ABBA": "#1F77B4", "BABA": "#D62728", "AABB": "#2CA02C"}
_RE_STATE_OCCUPANCY = re.compile(r"^State occupancy \(posterior, windows\): Null ([-\d.eE+na]+), (?:Alt|Outlier) ([-\d.eE+na]+)")


_SCORES_VARIANT_DESTS = ("pair", "site", "chunk_size", "normalize", "zscale", "ilr", "norm_eps", "exp_minus")


def _scores_with_truth(scores_path, pattern):
    if scores_path is None or not Path(scores_path).exists():
        raise ValueError(f"no scores.tsv ({scores_path or 'above the report'})")
    scores_path = Path(scores_path)
    df = pd.read_csv(scores_path, sep="\t").drop_duplicates("pos").sort_values("pos")
    columns = [next((c for c in df.columns if t.lower() in c.lower()), None) for t in TOPOLOGY_NAMES]
    if None in columns:
        raise ValueError(f"{scores_path.name} lacks the {'/'.join(TOPOLOGY_NAMES)} columns")
    positions = df["pos"].to_numpy()
    _, anomaly_intervals, _ = parse_pattern_string(pattern, block_size_bp=500000, total_span=positions.max())
    y_true = np.zeros(len(positions), dtype=int)
    for start_bp, end_bp in anomaly_intervals:
        y_true[(positions >= start_bp) & (positions <= end_bp)] = 1
    return df[columns].to_numpy(dtype=float), y_true


def _report_window_labels(report, scores_path, pattern):
    report = Path(report)
    for d in report.parents if scores_path is None else ():
        if (d / "scores.tsv").exists():
            scores_path = d / "scores.tsv"
            break
        if _OUT_SIZE_SEGMENT.match(d.name):
            break
    scores, y_true = _scores_with_truth(scores_path, pattern)
    occupancy = path_line = None
    with open(report) as handle:
        for line in handle:
            m = _RE_STATE_OCCUPANCY.match(line)
            if m:
                occupancy = (float(m.group(1)), float(m.group(2)))
            if _RE_STATE_PATH.match(line.strip()):
                path_line = line.strip()
                break
    if path_line is None:
        raise ValueError("no state path in the report")
    y_pred = np.array(path_line.split(","), dtype=int)
    if len(y_pred) != len(y_true):
        raise ValueError(f"state path has {len(y_pred)} windows, scores.tsv {len(y_true)}")
    return scores, y_true, y_pred, occupancy


_CASTER_VARIANT_DIRS = {"site", "pair", "ilr", "normalize", "norm-eps", "exp-minus"}
_WINDOW_ECDF_LINTHRESH = 1e-2


def _minmax_normalize(values, mode=True):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return values
    lo, hi = finite.min(), finite.max()
    unit = (values - lo) / (hi - lo if hi > lo else 1.0)
    return unit if mode == "0-1" else 2 * unit - 1


_GT_SCORES_MAX_ROWS = 20000
_GT_SCORES_CACHE = Path("store") / "cache" / "gt_scores"
_GT_SCORES_WORKERS = 16


def _gt_scores_cache_path(scores_path):
    st = scores_path.stat()
    key = hashlib.sha1(f"{scores_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{_GT_SCORES_MAX_ROWS}".encode()).hexdigest()
    return _gt_scores_cache_root() / key[:2] / f"{key}.npz"


@functools.lru_cache(maxsize=None)
def _gt_scores_cache_root():
    return get_repo_root() / _GT_SCORES_CACHE


def _cached_gt_scores(scores_path):
    cache = _gt_scores_cache_path(scores_path)
    try:
        with np.load(cache) as z:
            return z["null"], z["alt"]
    except (OSError, KeyError, ValueError):
        pass
    try:
        scores, y_true = _scores_with_truth(scores_path, scores_path.parent.name)
    except ValueError:
        return None
    if len(scores) > _GT_SCORES_MAX_ROWS:
        keep = np.sort(np.random.default_rng(0).choice(len(scores), _GT_SCORES_MAX_ROWS, replace=False))
        scores, y_true = scores[keep], y_true[keep]
    null, alt = scores[y_true == 0], scores[y_true == 1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, null=null, alt=alt)
    os.replace(tmp, cache)
    return null, alt


@functools.lru_cache(maxsize=32)
def _caster_scores_by_run(root, exclude_keywords=(), pattern=None):
    paths = [p for p in sorted(root.glob("*/*/*/*/scores.tsv"))
             if p.relative_to(root).parts[0] not in _CASTER_VARIANT_DIRS and not any(k in str(p) for k in exclude_keywords)
             and (pattern is None or p.parent.name == str(pattern))]
    if sum(not _gt_scores_cache_path(p).exists() for p in paths) > 1:
        with concurrent.futures.ProcessPoolExecutor(_GT_SCORES_WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
            loaded = list(pool.map(_cached_gt_scores, paths, chunksize=4))
    else:
        loaded = [_cached_gt_scores(p) for p in paths]
    return tuple((p.relative_to(root).parts[:4], *r) for p, r in zip(paths, loaded) if r is not None)


_COVARIANCE_PAIRS = tuple(itertools.combinations(range(len(TOPOLOGY_NAMES)), 2))
_COVARIANCE_PAIR_NAMES = [f"{TOPOLOGY_NAMES[a]}·{TOPOLOGY_NAMES[b]}" for a, b in _COVARIANCE_PAIRS]
_FIT_MOMENTS_CACHE = Path("store") / "cache" / "fit_moments"


def _fit_moments_cache_path(report):
    fit = report.with_suffix(".fit.pkl")
    stamp = "|".join(f"{p.stat().st_size}|{p.stat().st_mtime_ns}" for p in (report, fit) if p.exists())
    key = hashlib.sha1(f"{report.resolve()}|{stamp}".encode()).hexdigest()
    return get_repo_root() / _FIT_MOMENTS_CACHE / key[:2] / f"{key}.npz"


def _read_fitted_moments(report):
    import pickle
    lines = report.read_text().splitlines()
    flipped = any(line.strip() == "Label polarity flipped for evaluation: True" for line in lines)
    fit = report.with_suffix(".fit.pkl")
    if fit.exists():
        with open(fit, "rb") as f:
            e = pickle.load(f)["params"].emissions
        means, covs = np.asarray(e.means, dtype=float), np.asarray(e.covariances, dtype=float)
        if hasattr(e, "mixture_weights"):
            w = np.asarray(e.mixture_weights, dtype=float)
            mix = np.einsum("sm,smd->sd", w, means)
            covs = np.einsum("sm,smde->sde", w, covs + np.einsum("smd,sme->smde", means, means)) - np.einsum("sd,se->sde", mix, mix)
            means = mix
        if means.shape != (2, len(TOPOLOGY_NAMES)):
            return None
        order = [1, 0] if flipped else [0, 1]
        return means[order], covs[order]
    header = next((i for i, line in enumerate(lines) if line.startswith("Topology\tState\t")), None)
    if header is None:
        return None
    cols = lines[header].split("\t")
    means, covs = np.full((2, 3), np.nan), np.full((2, 3, 3), np.nan)
    for line in lines[header + 1:]:
        cells = line.split("\t")
        if len(cells) != len(cols) or cells[0] not in TOPOLOGY_NAMES:
            break
        row = dict(zip(cols, cells))
        d, s = TOPOLOGY_NAMES.index(row["Topology"]), 0 if row["State"] == "Null" else 1
        if "FittedMean" in row:
            means[s, d], covs[s, d, d] = float(row["FittedMean"]), float(row["FittedStd"]) ** 2
        elif row["Statistic"] == "mean":
            means[s, d] = float(row["Fitted"])
        elif row["Statistic"] == "std":
            covs[s, d, d] = float(row["Fitted"]) ** 2
    return means, covs


def _cached_fitted_moments(report):
    cache = _fit_moments_cache_path(report)
    try:
        with np.load(cache) as z:
            return (z["means"], z["covs"]) if z["ok"] else None
    except (OSError, KeyError, ValueError):
        pass
    try:
        moments = _read_fitted_moments(report)
    except (OSError, ValueError, KeyError, AttributeError, EOFError):
        moments = None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, ok=moments is not None, means=moments[0] if moments else np.empty(0), covs=moments[1] if moments else np.empty(0))
    os.replace(tmp, cache)
    return moments


def _fitted_moments_many(reports):
    if sum(not _fit_moments_cache_path(r).exists() for r in reports) > 1:
        with concurrent.futures.ProcessPoolExecutor(_GT_SCORES_WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
            return list(pool.map(_cached_fitted_moments, reports, chunksize=8))
    return [_cached_fitted_moments(r) for r in reports]


_HMM_SPLIT_CACHE = Path("store") / "cache" / "hmm_split_scores"
_VITERBI_SPLIT_STATS = {"fitted_all": "all", "viterbi_variances": "variances"}


def _hmm_split_cache_path(report, scores_path):
    stamp = "|".join(f"{p.resolve()}|{p.stat().st_size}|{p.stat().st_mtime_ns}" for p in (report, scores_path) if p.exists())
    key = hashlib.sha1(f"{stamp}|{_GT_SCORES_MAX_ROWS}".encode()).hexdigest()
    return get_repo_root() / _HMM_SPLIT_CACHE / key[:2] / f"{key}.npz"


def _cached_hmm_split_scores(report, scores_path):
    cache = _hmm_split_cache_path(report, scores_path)
    try:
        with np.load(cache) as z:
            return (z["null"], z["alt"]) if z["ok"] else None
    except (OSError, KeyError, ValueError):
        pass
    try:
        scores, _, y_pred, _ = _report_window_labels(report, scores_path, report.stem)
        if len(scores) > _GT_SCORES_MAX_ROWS:
            keep = np.sort(np.random.default_rng(0).choice(len(scores), _GT_SCORES_MAX_ROWS, replace=False))
            scores, y_pred = scores[keep], y_pred[keep]
        split = scores[y_pred == 0], scores[y_pred == 1]
    except (OSError, ValueError):
        split = None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, ok=split is not None, null=split[0] if split else np.empty(0), alt=split[1] if split else np.empty(0))
    os.replace(tmp, cache)
    return split


def _hmm_split_scores_many(reports, scores_paths):
    if sum(not _hmm_split_cache_path(r, s).exists() for r, s in zip(reports, scores_paths)) > 1:
        with concurrent.futures.ProcessPoolExecutor(_GT_SCORES_WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
            return list(pool.map(_cached_hmm_split_scores, reports, scores_paths, chunksize=8))
    return [_cached_hmm_split_scores(r, s) for r, s in zip(reports, scores_paths)]


def _pooled_caster_scores(root, exclude_keywords=(), pattern=None):
    runs = _caster_scores_by_run(root, exclude_keywords, pattern)
    if not runs:
        return None, None, 0
    return np.concatenate([r[1] for r in runs]), np.concatenate([r[2] for r in runs]), len(runs)


def _window_ecdf_figure(series, cols, topologies, xlabel, title, symlog=False, linthresh=None, normalize=False, legend=True,
                        color_by=None, cmap="Greys", state_labels=("Null", "Outlier")):
    if normalize:
        series = [(w, *np.split(_minmax_normalize(np.concatenate([null, alt]), normalize), [len(null)]), *rest)
                  for w, null, alt, *rest in series]
        xlabel = f"normalized {xlabel}"
    colors = plt.cm.Greys(np.linspace(0.35, 1.0, len(series))) if cmap == "Greys" else [plt.cm.tab10(i % 10) for i in range(len(series))]
    norm = color_by
    fig, axs = plt.subplots(2, len(cols), figsize=(3.8 * len(cols), 6.5), sharey=True, squeeze=False)
    handles, entries = [], {}
    for s, color in zip(series, colors):
        w, null, alt = s[:3]
        handles.append(matplotlib.lines.Line2D([], [], color=color, linewidth=1.6))
        for state, vals in enumerate((null, alt)):
            for ci, col in enumerate(cols):
                mean, _ = _draw_ecdf(axs[state][ci], vals[:, col], color=color, linewidth=0.6 if norm else 1.3, zorder=4 if norm else 2)
                if mean is not None:
                    entries.setdefault((state, ci), []).append((handles[-1], f"{w}: μ={mean:.3g}"))
                    if norm is not None:
                        axs[state][ci].axvline(mean, color=color, linewidth=1.0, linestyle="--", zorder=5)
                if norm is not None:
                    _scatter_ecdf_colored(axs[state][ci], vals[:, col], s[4 + state], norm)
    for ci, col in enumerate(cols):
        pooled = np.concatenate([v[:, col] for _, null, alt, *_ in series for v in (null, alt)])
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, [0.5, 99.5] if symlog else [1, 99]) if len(pooled) else (0, 0)
        for state, state_label in enumerate(state_labels):
            ax = axs[state][ci]
            if symlog:
                ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                ax.tick_params(axis="x", labelsize=6)
            if hi > lo:
                ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.set_title(f"{topologies[ci]} ({state_label})", fontsize=9)
            if state == 1:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel("proportion")
            if legend and (state, ci) in entries:
                ax.legend(*zip(*entries[state, ci]), fontsize=6, loc="best")
    _suptitle(fig, title, fontsize=10)
    if norm is not None:
        _add_color_key(fig, axs, norm)
    if not legend:
        _legend_below_axes(fig, handles, [s[0] for s in series], ncol=min(len(series), 10), fontsize=8)
    return fig


_HMM_STATE_COLORS = ("#1F3B73", "#E4573D")


def _hmm_state_ecdf_figure(series, cols, topologies, xlabel, title, symlog, linthresh, normalize, legend):
    if normalize:
        series = [(label, *np.split(_minmax_normalize(np.concatenate([null, alt]), normalize), [len(null)]))
                  for label, null, alt in series]
        xlabel = f"normalized {xlabel}"
    fig, axs = plt.subplots(len(series), len(cols), figsize=(3.8 * len(cols), 3.2 * len(series) + 1.0), sharex="col", squeeze=False)
    for ci, col in enumerate(cols):
        pooled = np.concatenate([v[:, col] for _, null, alt in series for v in (null, alt)])
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, [0.5, 99.5] if symlog else [1, 99]) if len(pooled) else (0, 0)
        for ri, (label, null, alt) in enumerate(series):
            ax = axs[ri][ci]
            means = []
            for vals, color in zip((null, alt), _HMM_STATE_COLORS):
                mean, n = _draw_ecdf(ax, vals[:, col], color=color, linewidth=1.6)
                means.append((mean, n))
                if mean is not None:
                    ax.axvline(mean, color=color, linewidth=1.0, alpha=0.7)
            if legend:
                _text_box(ax, ["Mean"] + [f"{name} HMM: {'n/a' if m is None else f'{m:.3g}'} (n={n})"
                                          for name, (m, n) in zip(("Null", "Outlier"), means)])
            if symlog:
                ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                ax.tick_params(axis="x", labelsize=6)
            if hi > lo:
                ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if ri == 0:
                ax.set_title(topologies[ci], fontsize=10)
            if ri == len(series) - 1:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel(f"{label}\nproportion")
    _suptitle(fig, title, fontsize=10)
    _legend_below_axes(fig, [matplotlib.lines.Line2D([], [], color=c, linewidth=1.6) for c in _HMM_STATE_COLORS],
                       ["Null HMM", "Outlier HMM"], ncol=2, fontsize=8)
    return fig
_SCATTER_MAX_POINTS = 20000


def _corr(xy, method="spearman"):
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2 or np.ptp(xy[:, 0]) == 0 or np.ptp(xy[:, 1]) == 0:
        return float("nan")
    if method == "spearman":
        xy = np.column_stack([stats.rankdata(xy[:, 0]), stats.rankdata(xy[:, 1])])
    return float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1])


_BL_POINT_CMAP = "rainbow"


def _scatter_ecdf_colored(ax, vals, codes, spec):
    keep = np.isfinite(vals)
    order = np.argsort(vals[keep], kind="stable")
    vals, codes = vals[keep][order], codes[keep][order]
    if not len(vals):
        return
    idx = np.unique(np.linspace(0, len(vals) - 1, min(len(vals), ECDF_MAX_POINTS)).astype(int))
    known = np.isfinite(codes[idx])
    ax.scatter(vals[idx][known], (idx[known] + 1) / len(vals), c=codes[idx][known], cmap=spec["cmap"], norm=spec["norm"], s=8,
               linewidths=0, zorder=3, rasterized=True)


def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _aggregate_color(values):
    present = [v for v in values if not pd.isna(v)]
    if not present:
        return np.nan
    if all(_is_number(v) for v in present):
        return float(np.mean(present))
    labels = [str(v) for v in present]
    return max(dict.fromkeys(labels), key=labels.count)


def _color_label(color):
    return "CU branch length" if color == "bl" else color


def _color_spec(values, label, log=False, cmap=_BL_POINT_CMAP):
    present = [v for v in values if not pd.isna(v)]
    if all(_is_number(v) for v in present):
        known = np.array(present, dtype=float)
        known = known[np.isfinite(known) & ((known > 0) if log else True)]
        if not len(known):
            return None
        return {"label": label, "cats": None, "cmap": matplotlib.colormaps[cmap],
                "norm": (matplotlib.colors.LogNorm if log else matplotlib.colors.Normalize)(known.min(), known.max())}
    cats = sorted(set(map(str, present)), key=_merged_category_sort_key if label == "merged_category" else
                  (lambda v: (_DIRECTION_ORDER.index(v) if v in _DIRECTION_ORDER else len(_DIRECTION_ORDER), v)) if label == "direction"
                  else _token_sort_key)
    return {"label": label, "cats": cats, "cmap": matplotlib.colors.ListedColormap([_config_color(i) for i in range(len(cats))]),
            "norm": matplotlib.colors.BoundaryNorm(np.arange(len(cats) + 1) - 0.5, len(cats))}


_DIRECTION_ORDER = ("up", "down", "admixture")


def _color_codes(spec, values):
    if spec["cats"] is None:
        return np.array([np.nan if pd.isna(v) or not _is_number(v) else v for v in values], dtype=float)
    idx = {c: i for i, c in enumerate(spec["cats"])}
    return np.array([np.nan if pd.isna(v) else idx.get(str(v), np.nan) for v in values], dtype=float)


def _color_of(spec, code):
    if spec is None or not np.isfinite(code) or (isinstance(spec["norm"], matplotlib.colors.LogNorm) and code <= 0):
        return "gray"
    return spec["cmap"](spec["norm"](code))


def _add_color_key(fig, axs, spec):
    if spec["cats"] is None:
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=spec["norm"], cmap=spec["cmap"]), ax=axs,
                     label="branch length (CU)" if spec["label"] == "bl" else spec["label"], fraction=0.025, pad=0.02)
    else:
        fig.legend([matplotlib.lines.Line2D([], [], color=_color_of(spec, i), linewidth=2) for i in range(len(spec["cats"]))],
                   spec["cats"], title=spec["label"], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, title_fontsize=8)


def _size_label(text):
    return re.sub(r"\b([wc])(\d+[km]?)_s(\d+[km]?)\b", lambda m: f"{m[1]}{m[2]}" if m[2] == m[3] else m[0], str(text))


def _ecdf_all_series(branches, topologies, normalize, percentile):
    topologies = [topologies] if isinstance(topologies, str) else list(topologies)
    cols = [TOPOLOGY_NAMES.index(t) for t in topologies]
    series = {k: [np.concatenate(v) for v in vals] for k, vals in branches.items()}
    if normalize:
        series = {k: np.split(_minmax_normalize(np.concatenate(v), normalize), [len(v[0])]) for k, v in series.items()}
    ranges, means = [], {}
    for ci, col in enumerate(cols):
        pooled = np.concatenate([v[:, col] for null, alt in series.values() for v in (null, alt)])
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, percentile) if len(pooled) else (0, 0)
        ranges.append((lo, hi))
        for key, states in series.items():
            for state in (0, 1):
                x = states[state][:, col]
                x = x[(x >= lo) & (x <= hi)] if hi > lo else x
                means[key, state, ci] = np.nanmean(x) if len(x) else np.nan
    return topologies, cols, series, ranges, means


def _mean_vs_bl_all_figure(branches, bls, topologies, state_labels, normalize, logx, linthresh, legend, percentile, title, colors=None,
                           logy=False):
    topologies, cols, series, ranges, means = _ecdf_all_series(branches, topologies, normalize, percentile)
    n_experiments = sum(len(null) for null, _ in branches.values())
    keys = [k for k in series if bls[k]]
    x = np.array([bls[k] for k in keys], dtype=float)
    grid = np.geomspace(x.min(), x.max(), 200) if logx is not False else np.linspace(x.min(), x.max(), 200)
    fig, axs = plt.subplots(2, len(cols), figsize=(3.8 * len(cols) + 1.2, 6.5), sharex=True, sharey="col", squeeze=False)
    for ci in range(len(cols)):
        for state, state_label in enumerate(state_labels):
            ax = axs[state][ci]
            y = np.array([means[k, state, ci] for k in keys], dtype=float)
            ok = np.isfinite(y)
            ax.scatter(x[ok], y[ok], s=14, color=[_color_of(colors[0], colors[1].get(k, np.nan)) for k, o in zip(keys, ok) if o]
                       if colors and colors[0] else "C0", alpha=0.6, linewidths=0)
            fit = _fit_exp_decay(x[ok], y[ok])
            if fit is not None:
                a, b, k = fit
                ax.plot(grid, a + b * np.exp(-k * grid), color="black", linewidth=1.4)
                ax.text(0.98, 0.98, f"y = {a:.3g} {'+' if b >= 0 else '−'} {abs(b):.3g}·e^(−{k:.3g}x)",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5))
            if logx is not False:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(f"{topologies[ci]} ({state_label})", fontsize=9)
            if state == 1:
                ax.set_xlabel("branch length (CU)")
            if ci == 0:
                ax.set_ylabel("normalized mean" if normalize else "mean")
    if colors and colors[0]:
        _add_color_key(fig, axs, colors[0])
    _suptitle(fig, title(n_experiments), fontsize=10)
    return fig


def _fit_exp_decay(x, y):
    if len(x) < 4:
        return None
    try:
        params, _ = optimize.curve_fit(lambda t, a, b, k: a + b * np.exp(-k * t), x, y,
                                             p0=(y[np.argmax(x)], y[np.argmin(x)] - y[np.argmax(x)], 1.0), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return params


def _ecdf_all_figure(branches, bls, topologies, state_labels, normalize, logx, linthresh, legend, percentile, title, colors,
                     fit=False):
    topologies, cols, series, ranges, means = _ecdf_all_series(branches, topologies, normalize, percentile)
    n_experiments = sum(len(null) for null, _ in branches.values())
    spec, codes = colors
    fig, axs = plt.subplots(2, len(cols), figsize=(3.8 * len(cols) + 1.2, 6.5), sharey=True, squeeze=False)
    for key in sorted(series, key=lambda k: (not np.isfinite(codes.get(k, np.nan)), codes.get(k, 0))):
        color = _color_of(spec, codes.get(key, np.nan))
        for state in (0, 1):
            for ci, col in enumerate(cols):
                _draw_ecdf(axs[state][ci], series[key][state][:, col], color=color, linewidth=0.7, alpha=0.6)
    symlog = (not normalize) if logx is None else logx
    for ci, col in enumerate(cols):
        lo, hi = ranges[ci]
        for state, state_label in enumerate(state_labels):
            ax = axs[state][ci]
            if symlog:
                ax.set_xscale("symlog", linthresh=linthresh or _WINDOW_ECDF_LINTHRESH)
                ax.tick_params(axis="x", labelsize=6)
            if hi > lo:
                ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            mean = np.nanmean([means[k, state, ci] for k in series])
            ax.axvline(mean, color="black", linewidth=1.4)
            box = [f"mean = {mean:.3g}"]
            if fit:
                pooled = np.concatenate([series[k][state][:, col] for k in series])
                pooled = pooled[np.isfinite(pooled)]
                pooled = pooled[(pooled >= lo) & (pooled <= hi)] if hi > lo else pooled
                if len(pooled) > 1:
                    mu, sigma = stats.norm.fit(pooled)
                    _draw_ecdf(ax, pooled, color="black", linewidth=2.2)
                    grid = np.union1d(np.linspace(pooled.min(), pooled.max(), 400), np.quantile(pooled, np.linspace(0, 1, 400)))
                    ax.plot(grid, stats.norm.cdf(grid, mu, sigma), color="red", linestyle="--", linewidth=1.8)
                    box.append(f"N(μ={mu:.3g}, σ={sigma:.3g}) D={stats.kstest(pooled, 'norm', args=(mu, sigma)).statistic:.3f}")
            if legend:
                ax.text(0.02, 0.98, "\n".join(box), transform=ax.transAxes, ha="left", va="top", fontsize=7,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5))
            ax.set_title(f"{topologies[ci]} ({state_label})", fontsize=9)
            if state == 1:
                ax.set_xlabel("normalized window mean" if normalize else "window mean")
            if ci == 0:
                ax.set_ylabel("proportion")
    if spec:
        _add_color_key(fig, axs, spec)
    if fit:
        fig.legend([matplotlib.lines.Line2D([], [], color="black", linewidth=1.4),
                    matplotlib.lines.Line2D([], [], color="black", linewidth=2.2),
                    matplotlib.lines.Line2D([], [], color="red", linestyle="--", linewidth=1.8)],
                   ["mean of per-branch means", "pooled ECDF (all branches, clipped)", "Gaussian MLE fit to pooled"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    _suptitle(fig, title(n_experiments), fontsize=10)
    return fig


@functools.lru_cache(maxsize=None)
def _sim_cu_branch_length(sim_dir):
    _, clade, _ = get_simulation_clade(Path(sim_dir) / "_" / "scores.tsv")
    bl = get_cu_branch_length_from_population_info(clade) if clade else None
    return None if bl is None else bl * BRANCH_LENGTH_SCALE


@functools.lru_cache(maxsize=None)
def _admixture_cu_branch_length(source_leaf, pattern):
    _, clade, _ = get_simulation_clade(Path(source_leaf) / str(pattern) / "scores.tsv")
    return get_cu_branch_length_from_population_info(clade) if clade else None


def _cu_branch_lengths(df):
    cu = df["branch_length_cu"].astype(float).copy()
    missing = cu.isna() & (df["category"] == "admixture")
    cu[missing] = [_admixture_cu_branch_length(l, p) for l, p in zip(df.loc[missing, "source_leaf"], df.loc[missing, "pattern"])]
    return cu * BRANCH_LENGTH_SCALE


def _gt_mean_axis_values(df, col):
    vals = df[col].unique().tolist()
    if col == "merged_category":
        return sorted(vals, key=_merged_category_sort_key)
    if col == "branch_length":
        return [b for b, _, _ in BRANCH_LENGTH_BIN_EDGES if b in vals]
    if col == "topology":
        return [t for t in TOPOLOGY_NAMES if t in vals]
    return sorted(vals, key=lambda v: (not isinstance(v, (int, float)), v if isinstance(v, (int, float)) else str(v)))


_POOLED_SCORE_AXES = {
    "-w": lambda w, parts: w,
    "category": lambda w, parts: parts[0],
    "subcategory": lambda w, parts: parts[1],
    "merged_category": lambda w, parts: parts[0] if parts[0] == "admixture" else f"{parts[0]} {parts[1]}",
    "direction": lambda w, parts: parts[0] if parts[0] == "admixture" else parts[1],
    "pattern": lambda w, parts: parts[3],
    "x_bin": lambda w, parts: parts[1] if parts[0] == "admixture" else _sim_branch_length_bin(parts),
    "fraction_bin": lambda w, parts: assign_bin(nominal_anomaly_fraction(parts[3]), FRACTION_BIN_EDGES),
}
_ORDERED_POOLED_AXES = ("-w", "fraction_bin")


def _sim_branch_length_bin(parts):
    cu = _sim_cu_branch_length(str(Path("store", "caster", *parts[:3])))
    return None if cu is None else assign_bin(cu, BRANCH_LENGTH_BIN_EDGES)


def _pooled_axis_values(axis, vals):
    if axis is None:
        return [None]
    if axis == "merged_category":
        return sorted(vals, key=_merged_category_sort_key)
    if axis in _DATA_COLUMN_AXES:
        order = _DATA_COLUMN_AXES[axis]
        return sorted(vals, key=lambda v: (v not in order, order.index(v) if v in order else str(v)))
    return sorted(vals, key=lambda v: (not isinstance(v, (int, float)), v if isinstance(v, (int, float)) else str(v)))


def _gt_mean_axis_label(axis, value):
    if axis == "-w":
        return f"w{format_val(value)}"
    if axis in ("x_bin", "fraction_bin"):
        return _row_axis_title(axis, value)
    if axis == "branch_length":
        return f"BL {value} CU"
    return str(value)


def _draw_ecdf(ax, vals, **style):
    finite = vals[np.isfinite(vals)]
    if not len(finite):
        return None, 0
    ax.plot(*ecdf_points(finite), drawstyle="steps-post", **style)
    return float(finite.mean()), len(finite)


_STATE_STYLES = (("Null", 0, "--"), ("Outlier", 1, "-"))
_QQ_PROBS = np.linspace(0.005, 0.995, 199)
_QQ_FITS = {"gaussian": (stats.norm, 1.2, 0.75), "laplace": (stats.laplace, 0.8, 0.45)}


def _dist_series(scores, labels, topologies):
    for topology in topologies:
        vals = scores[:, TOPOLOGY_NAMES.index(topology)]
        for state_name, state, linestyle in _STATE_STYLES:
            v = vals[(labels == state) & np.isfinite(vals)]
            if len(v):
                yield topology, state_name, state, linestyle, v


def _text_box(ax, lines):
    ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", multialignment="left",
            fontsize=7, family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5))


def _score_xaxis(ax, scores, topologies, normalize, logx):
    finite = scores[:, [TOPOLOGY_NAMES.index(t) for t in topologies]]
    finite = finite[np.isfinite(finite)]
    lo, hi = ((0, 1) if normalize == "0-1" else (-1, 1)) if normalize else np.percentile(finite, [0.5, 99.5]) if len(finite) else (0, 0)
    if hi > lo:
        ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    if logx and not normalize and len(finite) and finite.min() > 0 and _resolve_logy(logx, {"": pd.DataFrame({"v": finite})}, "v"):
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))


def _panel_ecdf(ax, scores, labels, topologies, normalize, logx):
    means = {}
    for topology, state_name, state, linestyle, v in _dist_series(scores, labels, topologies):
        means[topology, state], _ = _draw_ecdf(ax, v, color=_ECDF_TOPO_COLORS[topology], linestyle=linestyle, linewidth=1.6)
    _text_box(ax, ["Δ(O−N)" + (", normalized" if normalize else "")]
              + [f"{t}: {means.get((t, 1), np.nan) - means.get((t, 0), np.nan):+.3g}" for t in topologies])
    _score_xaxis(ax, scores, topologies, normalize, logx)


def _panel_qq(ax, scores, labels, topologies, normalize, logx):
    theo = stats.laplace.ppf(_QQ_PROBS)
    lines = []
    for topology, state_name, state, linestyle, v in _dist_series(scores, labels, topologies):
        loc = np.median(v)
        scale = np.mean(np.abs(v - loc)) or 1.0
        ax.plot(theo, (np.quantile(v, _QQ_PROBS) - loc) / scale, color=_ECDF_TOPO_COLORS[topology], linestyle=linestyle, linewidth=1.6)
        lines.append(f"{topology} {state_name[0]}: D={stats.kstest(v, 'laplace', args=(loc, scale)).statistic:.3f}")
    ax.plot(theo, theo, color="black", linewidth=1)
    ax.plot(theo, stats.norm.ppf(_QQ_PROBS) / np.sqrt(2 / np.pi), color="gray", linestyle=":", linewidth=1.2)
    _text_box(ax, lines)


def _fit_series(scores, labels, topologies):
    for topology, state_name, state, linestyle, v in _dist_series(scores, labels, topologies):
        fits = {name: rv.fit(v) for name, (rv, _, _) in _QQ_FITS.items()}
        ds = " ".join(f"{name[0]}={stats.kstest(v, rv.cdf, args=fits[name]).statistic:.3f}" for name, (rv, _, _) in _QQ_FITS.items())
        yield topology, linestyle, v, fits, f"{topology} {state_name[0]}: D {ds}"


def _panel_fit(ax, scores, labels, topologies, normalize, logx):
    lines = []
    for topology, linestyle, v, fits, line in _fit_series(scores, labels, topologies):
        color = _ECDF_TOPO_COLORS[topology]
        _draw_ecdf(ax, v, color=color, linestyle=linestyle, linewidth=2.2)
        grid = np.linspace(v.min(), v.max(), 400)
        for name, (rv, lw, alpha) in _QQ_FITS.items():
            ax.plot(grid, rv.cdf(grid, *fits[name]), color=color, linestyle=linestyle, linewidth=lw, alpha=alpha)
        lines.append(line)
    _text_box(ax, lines)
    _score_xaxis(ax, scores, topologies, normalize, logx)


def _panel_fit_qq(ax, scores, labels, topologies, normalize, logx):
    theo = stats.laplace.ppf(_QQ_PROBS)
    lines = []
    for topology, linestyle, v, fits, line in _fit_series(scores, labels, topologies):
        color = _ECDF_TOPO_COLORS[topology]
        loc, scale = fits["laplace"]
        ax.plot(theo, (np.quantile(v, _QQ_PROBS) - loc) / scale, color=color, linestyle=linestyle, linewidth=1.8)
        ax.plot(theo, (stats.norm.ppf(_QQ_PROBS, *fits["gaussian"]) - loc) / scale, color=color, linestyle=linestyle,
                linewidth=1.0, alpha=0.55)
        lines.append(line)
    ax.plot(theo, theo, color="black", linewidth=1)
    _text_box(ax, lines)


def _series_handles(topologies, kinds):
    return [(matplotlib.lines.Line2D([], [], color=_ECDF_TOPO_COLORS[t], linestyle=ls, linewidth=lw, alpha=alpha),
             f"{t} {name}{' ' + kind if kind else ''}")
            for t in topologies for kind, lw, alpha in kinds for name, _, ls in _STATE_STYLES]


_DIST_OPTIONS = {"pattern", "topologies", "logx", "linthresh", "normalize", "legend"}

_SCORE_DIST_METRICS = {
    "ecdf": (_panel_ecdf, "topology-score ECDFs", "topology score", "proportion",
             lambda ts: _series_handles(ts, [("", 1.6, 1.0)])),
    "qq": (_panel_qq, "topology-score QQ vs fitted Laplace (median, mean abs dev)", "Laplace(0,1) quantile",
           "standardized sample quantile",
           lambda ts: _series_handles(ts, [("", 1.6, 1.0)])
           + [(matplotlib.lines.Line2D([], [], color="black", linewidth=1), "Laplace (y=x)"),
              (matplotlib.lines.Line2D([], [], color="gray", linestyle=":", linewidth=1.2), "Gaussian ref")]),
    "fit": (_panel_fit, "topology-score ECDF (thick) vs Gaussian/Laplace MLE CDFs", "topology score", "proportion",
            lambda ts: _series_handles(ts, [("observed ECDF", 2.2, 1.0)]
                                       + [(f"{name} fit", lw, alpha) for name, (_, lw, alpha) in _QQ_FITS.items()])),
    "fit_qq": (_panel_fit_qq, "topology-score QQ vs Laplace fit (y=x) and Gaussian fit (curve)", "Laplace(0,1) quantile",
               "Laplace-standardized quantile",
               lambda ts: _series_handles(ts, [("observed", 1.8, 1.0), ("gaussian fit", 1.0, 0.55)])
               + [(matplotlib.lines.Line2D([], [], color="black", linewidth=1), "laplace fit (y=x)")]),
}

_TOPOLOGY_SCATTER_METRICS = {f"topology_topology_{s}": s for s in ("all", "means", "variances", "covariances", "fitted_all", "viterbi_variances", "fitted_means",
                                                                    "fitted_variances", "fitted_covariances")}

_DIST_METRICS = {*_TOPOLOGY_SCATTER_METRICS, "window_ecdf", "ecdf_all", "mean_all_vs_bl", "ecdf_hmm", "ecdf_hmm_all", "ecdf_all_hmm", "gap_ecdf", "mean_ecdf", "gap_vs_bl", "mean_vs_bl", *_SCORE_DIST_METRICS}


def _out_transition_totals(parsed):
    totals = {"fitted_transition_total": transition_total(parsed)}
    for prefix in ("gt", "viterbi"):
        to_alt, to_null = parsed[f"{prefix}_transition_null_to_alt"], parsed[f"{prefix}_transition_alt_to_null"]
        totals[f"{prefix}_transition_total"] = None if to_alt is None or to_null is None else to_alt + to_null
    return totals


_OUT_SIZE_SEGMENT = re.compile(r"^([wc])\d+[km]?_s\d+[km]?(.*)$", re.IGNORECASE)


_DEST_TO_FLAG = {}
for _flag, _dest in _FLAG_TO_DEST.items():
    _DEST_TO_FLAG.setdefault(_dest, _flag)


def _path_after_out(path):
    from phlag.utils import get_out_relative_parts
    rel = get_out_relative_parts(Path(path).parts)
    return "/".join(rel) if rel is not None else str(path)


def _recorded_from_report_path(report):
    """args.json-shaped, dest-keyed dict recovered from an out/ report's own
    path (parse_ws_from_path for window/step/site/ilr/normalize/norm-eps/z,
    plus get_phlag_param_segments' dist/rho_beta/flag
    segments after the size dir). Only what the path encodes is present, so
    callers can tell "not recorded" (key absent) from "off" (False/None)."""
    report = Path(report)
    mode, window, step, is_site, is_z, is_ilr, is_norm, norm_eps = parse_ws_from_path(report)
    parts = report.parent.parts
    size_idx = next(i for i, part in enumerate(parts) if _OUT_SIZE_SEGMENT.match(part))
    rest = [p for p in parts[size_idx + 1:] if p not in ("site", "ilr", "normalize", "norm-eps", "exp-minus")]
    recorded = {
        "window_size": window, "step_size": step, "site": is_site, "zscale": is_z, "ilr": is_ilr,
        "normalize": is_norm, "norm_eps": norm_eps, "exp_minus": "exp-minus" in parts, "pair": mode == "c" and not is_site,
        "rho": None, "beta": None, "beta_prime": None, **PHLAG_SEGMENT_DEFAULTS, "emission_lambda": None,
    }
    if rest:
        recorded["dist_type"] = rest[0]
    for seg in rest[1:]:
        recorded.update(parse_phlag_param_segment(seg) or {})
    return recorded


def _filter_out_runs(df, axes, recorded=None):
    """Keeps the rows of collect_out_runs' `df` whose path-derived flags
    (_recorded_from_report_path) satisfy every entry of `axes` -- a
    {cli_flag: requirement} dict as in the store-mode plot(), each requirement
    optionally a list/tuple meaning any-of, matched via _matches_flags (same
    None/False/typed-value rules); "-s" as a (0, 1] fraction is resolved
    against each row's own window. Flags the path can't tell (e.g. --np) are
    returned as `skipped` instead of silently passing or failing every row.
    `recorded` (one args.json dict per row) replaces the path-derived flags,
    as for collect_store_runs."""
    if recorded is None:
        recorded = [_recorded_from_report_path(p) for p in df["report_path"]]
    known = set(recorded[0]) | {"dist_type"} if recorded else set()
    skipped, checks = [], []
    for flag, requirement in axes.items():
        dest = _FLAG_TO_DEST.get(flag)
        if dest is None:
            raise KeyError(f"{flag!r} is not one of benchmark.py's mirrored caster/phlag flags")
        if dest not in known:
            skipped.append(_DEST_TO_FLAG[dest])
        else:
            checks.append((flag, requirement if isinstance(requirement, (list, tuple, set)) else [requirement]))

    def row_ok(rec):
        for flag, options in checks:
            hit = False
            for req in options:
                if flag in ("-s", "--step-size") and _as_step_fraction(req) is not None:
                    req = max(1, round(_as_step_fraction(req) * rec["window_size"]))
                hit = hit or _matches_flags(rec, {flag: req})
            if not hit:
                return False
        return True

    keep = [row_ok(r) for r in recorded]
    return df[keep].reset_index(drop=True), skipped


def _out_gt_stats_columns(report):
    """null_/alt_/pooled_ mean/var/cov(_norm)/within_mean/within_var columns,
    named as in store runs.tsv (benchmark.py's summarize), from the gt_stats.txt caster wrote
    next to the report's scores.tsv -- the nearest one walking up from the
    report, since phlag's own dist/rho_beta/... dirs sit below the size dir.
    Empty if none is found."""
    for d in report.parents:
        if (d / "gt_stats.txt").exists():
            stats = read_gt_stats_file(d / "gt_stats.txt")
            break
        if _OUT_SIZE_SEGMENT.match(d.name):
            return {}
    else:
        return {}
    out = {}
    for label, prefix in _OUT_GT_STATS_LABELS:
        if label not in stats:
            continue
        mean, cov = np.asarray(stats[label][0], dtype=float), np.asarray(stats[label][1], dtype=float)
        out[f"{prefix}_mean_norm"], out[f"{prefix}_cov_norm"] = float(np.linalg.norm(mean)), float(np.linalg.norm(cov))
        for i, ti in enumerate(TOPOLOGY_NAMES):
            out[f"{prefix}_mean_{ti}"] = float(mean[i])
            out[f"{prefix}_var_{ti}"] = float(cov[i, i])
            for j in range(i + 1, 3):
                out[f"{prefix}_cov_{ti}_{TOPOLOGY_NAMES[j]}"] = float(cov[i, j])
        within_mean, within_var = stats.get(f"{label}WithinMean"), stats.get(f"{label}WithinVariance")
        if within_mean is not None:
            for ti, v in zip(TOPOLOGY_NAMES, within_mean):
                out[f"{prefix}_within_mean_{ti}"] = float(v)
        if within_var is not None:
            for ti, v in zip(TOPOLOGY_NAMES, within_var):
                out[f"{prefix}_within_var_{ti}"] = float(v)
    return out


def collect_out_runs(path, exclude_keywords=(), windowless=False):
    """One row per report.tsv under `path` (a runs.tsv-shaped table for the
    ad-hoc out/ tree, which has no runs.tsv of its own), columns: OUT_RUN_ID_COLUMNS
    plus every _OUT_RUN_METRIC_KEYS name (runs.tsv's f1/roc_auc/em_hd/... and
    the fitted_/gt_/viterbi_transition_<from>_to_<to> probabilities parsed by
    benchmark.parse_report). also the null_/alt_/pooled_ gt_stats columns (`_out_gt_stats_columns`). `config` is the report's directory relative to
    `path` minus its w<W>_s<S> size segment (a pair-mode 'c' size or flat
    '_site'-style suffix is kept, as it changes what's being compared),
    with leading segments shared by every report dropped. windowless=True
    keeps reports with no size segment (window/step/mode None, no gt_stats)."""
    path = Path(path)
    records = []
    for report in sorted(path.rglob("report.tsv")):
        if any(k in str(report) for k in exclude_keywords):
            continue
        ws = None if windowless else parse_ws_from_path(report)
        if ws is None and not windowless:
            continue
        mode, window, step = ws[:3] if ws else (None, None, None)
        parts = list(report.relative_to(path).parent.parts)
        tokens = []
        for part in parts:
            m = _OUT_SIZE_SEGMENT.match(part)
            if m:
                tokens.append(("c" if m.group(1).lower() == "c" else "") + m.group(2))
            else:
                tokens.append(part)
        parsed = parse_report(report)
        row = {"report_path": str(report), "mode": mode, "window": window, "step": step, "_tokens": [t for t in tokens if t]}
        row.update({col: parsed[key] for col, key in _OUT_RUN_METRIC_KEYS.items()})
        row.update(_out_transition_totals(parsed))
        if not windowless:
            row.update(_out_gt_stats_columns(report))
        records.append(row)
    if not records:
        return pd.DataFrame(columns=OUT_RUN_ID_COLUMNS)
    depth = min(len(r["_tokens"]) for r in records)
    shared = 0
    while shared < depth and len({r["_tokens"][shared] for r in records}) == 1:
        shared += 1
    for r in records:
        r["config"] = "/".join(r.pop("_tokens")[shared:]) or "(none)"
    df = pd.DataFrame(records)
    return df.reindex(columns=_OUT_RUN_COLUMNS)


def collect_store_runs(rel, root=STORE_ROOT, exclude_keywords=()):
    """collect_out_runs' store/ counterpart: one row per finished leaf run dir
    under `root` (own args.json + analysis.tsv) holding reports/<rel>.tsv,
    `rel` being <category>/<subcategory>/<node>/<pattern>. window/step come
    from the leaf's args.json, gt_stats columns from its runs.tsv row, and
    `config` is the leaf's path below `root` minus its size segment, shared
    leading segments dropped. Returns (df, recorded args.json per row)."""
    rel = Path(rel)
    source_leaf, pattern = "/".join(rel.parts[:-1]), rel.name
    records, recorded = [], []
    for args_path in sorted(Path(root).rglob("args.json")):
        leaf = args_path.parent
        report = leaf / "reports" / rel.parent / f"{pattern}.tsv"
        if not report.exists() or not (leaf / "analysis.tsv").exists():
            continue
        if any(k in str(report) for k in exclude_keywords):
            continue
        args = _read_args_json(args_path)
        ws = parse_ws_from_path(leaf)
        parsed = parse_report(report)
        row = {"report_path": str(report), "mode": ws[0] if ws else None,
               "window": args.get("window_size"), "step": args.get("step_size"),
               "_tokens": [p for p in leaf.relative_to(root).parts if not _OUT_SIZE_SEGMENT.match(p)]}
        row.update({col: parsed[key] for col, key in _OUT_RUN_METRIC_KEYS.items()})
        row.update(_out_transition_totals(parsed))
        runs = pd.read_csv(leaf / "runs.tsv", sep="\t")
        match = runs[(runs["source_leaf"] == source_leaf) & (runs["pattern"].astype(str) == pattern)]
        if not match.empty:
            row.update({c: match.iloc[0][c] for c in _OUT_GT_STATS_COLUMNS if c in match.columns})
        records.append(row)
        recorded.append(args)
    if not records:
        return pd.DataFrame(columns=OUT_RUN_ID_COLUMNS), []
    depth = min(len(r["_tokens"]) for r in records)
    shared = 0
    while shared < depth and len({r["_tokens"][shared] for r in records}) == 1:
        shared += 1
    for r in records:
        r["config"] = "/".join(r.pop("_tokens")[shared:]) or "(none)"
    df = pd.DataFrame(records)
    return df.reindex(columns=_OUT_RUN_COLUMNS), recorded


def _windowless_source_of(d):
    for source in WINDOWLESS_PHLAG_SOURCES:
        if Path(d) == Path(PHLAG_SOURCE_ROOTS[source]):
            return source
    return None


def _path_below_source_root(path):
    path = Path(path)
    parts = (path if path.is_absolute() else get_repo_root() / path).parts
    for root in PHLAG_SOURCE_ROOTS.values():
        root_parts = Path(root).parts
        if parts[:len(root_parts)] == root_parts:
            return Path(*parts[len(root_parts):])
    return path


def _resolve_run_path(path):
    path = Path(path)
    if path.is_absolute() or (get_repo_root() / path).exists():
        return str(path)
    return str(Path(get_out_root()) / path)


def _collect_run_df(axes, path, exclude_keywords=(), dir="store", store_root=STORE_ROOT):
    """collect_out_runs + _filter_out_runs for one run dir. An axes "phlag"
    entry (a PHLAG_SOURCE_ROOTS key, or a list of them) instead reads `path`
    (relative to a source root, or with one as prefix) under each named root,
    prefixing every row's config with its source name. A
    WINDOWLESS_PHLAG_SOURCES source has no w<W>_s<S> dirs, so its reports
    are kept unfiltered by `axes`, with window/step None. dir="store" reads
    the same run's reports from `store_root` instead (collect_store_runs),
    filtered by each leaf's args.json."""
    assert dir in ("out", "store")
    axes = dict(axes)
    sources = axes.pop("phlag", None)
    if dir == "store" and sources is None:
        df, recorded = collect_store_runs(_path_below_source_root(path), store_root, exclude_keywords)
        if df.empty:
            raise FileNotFoundError(f"No store report found for {path} under {store_root}")
        df, _ = _filter_out_runs(df, axes, recorded)
    elif sources is None:
        df = collect_out_runs(path, exclude_keywords)
        if df.empty:
            raise FileNotFoundError(f"No report.tsv found under {path}")
        df, _ = _filter_out_runs(df, axes)
    else:
        rel = _path_below_source_root(path)
        frames = []
        for source in _register_phlag_sources(sources):
            windowless = source in WINDOWLESS_PHLAG_SOURCES
            d = collect_out_runs(Path(PHLAG_SOURCE_ROOTS[source]) / rel, exclude_keywords, windowless=windowless)
            if d.empty:
                continue
            if not windowless:
                d, _ = _filter_out_runs(d, axes)
            d = d.assign(config=[source if c == "(none)" else f"{source}/{c}" for c in d["config"]])
            frames.append(d.dropna(axis=1, how="all"))
        columns = _OUT_RUN_COLUMNS
        df = pd.concat(frames, ignore_index=True).reindex(columns=columns) if frames else pd.DataFrame()
    if df.empty:
        raise ValueError(f"No report under {path} matches {axes} (phlag={sources})")
    return df


_RHO_BETA_CONFIG = re.compile(r"rho([^_/]+)_beta(?:prime)?([^_/]+)")


def _srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=float)[..., :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _separate_colors(palette, order, group_of, min_delta_e=20, min_shade_delta_e=10):
    pool = [tuple(c) for name in ("tab10", "Dark2", "Set1", "tab20", "Set2") for c in sns.color_palette(name)]
    pool += [(0.5, 0.5, 0.5), (0.75, 0.75, 0.75)]
    pool_lab = _srgb_to_lab(pool)
    out, used, used_groups = {}, [], []
    for c in order:
        color = palette[c]
        lab = _srgb_to_lab(color)
        same = np.array([g is not None and g == group_of.get(c) for g in used_groups], dtype=bool)
        dists = np.linalg.norm(np.array(used) - lab, axis=1) if used else np.array([])
        if (dists < np.where(same, min_shade_delta_e, min_delta_e)).any():
            gaps = np.linalg.norm(pool_lab[:, None] - np.array(used)[None], axis=2).min(axis=1)
            color = pool[int(gaps.argmax())]
            lab = pool_lab[int(gaps.argmax())]
        out[c] = color
        used.append(lab)
        used_groups.append(group_of.get(c))
    return out


def _rho_beta_palette(configs):
    """One tab10 hue per rho, shades of it from light to slightly-darker-than-
    base (never near black) by ascending beta;
    configs without a rho<X>_beta[prime]<Y> segment get black/greys. Any color
    too close (CIELAB) to an earlier one is swapped for a distinct one. Returns
    configs reordered (rho, beta ascending, others last), the palette, legend
    labels padded with None so each rho group fills one column, and ncol."""
    groups, others = {}, []
    for c in configs:
        m = _RHO_BETA_CONFIG.search(c)
        if m:
            groups.setdefault(float(m.group(1)), []).append((float(m.group(2)), c))
        else:
            others.append(c)
    tab10 = sns.color_palette("tab10")
    hues = [tab10[i] for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]
    palette, ordered = {}, []
    for hi, rho in enumerate(sorted(groups)):
        members = [c for _, c in sorted(groups[rho])]
        base = np.array(hues[hi % len(hues)])
        for t, c in zip(np.linspace(0.6, -0.25, len(members)) if len(members) > 1 else [0], members):
            palette[c] = tuple(base + (1 - base) * t if t >= 0 else base * (1 + t))
            ordered.append(c)
    for i, c in enumerate(others):
        palette[c] = (0.6 * i / max(len(others) - 1, 1),) * 3
    ordered += others
    group_of = {c: rho for rho, members in groups.items() for _, c in members}
    palette = _separate_colors(palette, ordered, group_of)
    if not groups:
        return ordered, palette, others, max(min(len(others), 6), 1)
    rows = max(max(len(v) for v in groups.values()), len(others))
    legend = [c for rho in sorted(groups) for c in ([c for _, c in sorted(groups[rho])] + [None] * rows)[:rows]]
    legend += (others + [None] * rows)[:rows] if others else []
    return ordered, palette, legend, len(legend) // rows


def _branch_length_label(df, path):
    if "admixture" in Path(path).parts:
        t = get_admixture_divergence_time(path)
        return "admixture" if t is None else f"admixture t={t:.2f} Myr"
    lengths = df["branch_length"].dropna()
    if lengths.empty:
        _, clade, _ = get_simulation_clade(Path(path) / "scores.tsv")
        bl = get_cu_branch_length_from_population_info(clade) if clade else None
    else:
        bl = lengths.iloc[0]
    return "BL N/A" if bl is None else f"BL {bl:.3g} CU"


def _typed_value(flag, raw_value):
    """Coerces a raw flags-dict value (e.g. "50k") through FLAG's own argparse
    type (e.g. int_or_abbrev), matching how args.json stores it (e.g. 50000).
    Bools (store_true flags) pass through unchanged. A whole-number float
    string (e.g. "1.0") is retried as its int() after the type_fn's first
    attempt fails -- int_or_abbrev itself only handles bare int() for a
    non-k/m suffix, so "1.0" would raise even on a real CLI invocation;
    this only smooths over that same gap for axis literals, not a fix to
    int_or_abbrev itself."""
    if isinstance(raw_value, bool):
        return raw_value
    action = _PARSER._option_string_actions.get(flag)
    type_fn = action.type if action and action.type else (lambda v: v)
    try:
        return type_fn(raw_value)
    except (TypeError, ValueError):
        try:
            f = float(raw_value)
        except (TypeError, ValueError):
            raise
        if f != int(f):
            raise
        return type_fn(str(int(f)))


def _matches_flags(recorded, flags):
    """flags: {cli_flag_token: requirement}, using literal CLI flag tokens exactly as
    they appear on benchmark.py's parser (e.g. "-d", "-w", "-s", "--ap", "--annealing",
    "--double-variance-init"). requirement=True/False requires that dest's recorded
    value be truthy/falsy (store_true flags); a plain value requires the recorded value
    to equal that value (typed through the flag's own argparse type); requirement=None
    requires the recorded value be missing/null -- e.g. an older args.json written
    before that flag was added to CASTER_ARG_SPECS/PHLAG_ARG_SPECS, so the dest is
    simply absent. A missing dest (recorded.get(dest) is None) also satisfies an
    explicit requirement=False, AND a recorded explicit False also satisfies
    requirement=None -- both directions of the same "missing == off" equivalence for
    a store_true flag that some args.json predate and others record explicitly (e.g.
    --site, added in b0c28fa: some leaf dirs' args.json were written before it existed
    at all, recording nothing, while sibling dirs re-run since then record `"site":
    false` explicitly for the exact same off state) -- treating those as distinct
    would wrongly exclude one sibling or the other from a pool/agg axis pinned to
    that flag's off state (requirement=False) or its "never recorded" state
    (requirement=None), even though on disk they mean the same thing. Safe for any
    flag type, not just booleans: an `is False` identity check only ever matches a
    literal recorded False, never a falsy non-bool value (0, ""), which a numeric/
    string flag could otherwise (rarely, and likely a data bug if so) have recorded.
    recorded: one leaf's args.json, dest-keyed."""
    for flag, requirement in flags.items():
        dest = _FLAG_TO_DEST.get(flag)
        if dest is None:
            raise KeyError(f"{flag!r} is not one of benchmark.py's mirrored caster/phlag flags")
        value = _recorded_value(recorded, dest)
        if requirement is None:
            if value is not None and value is not False and value != _RECORDED_DEFAULTS.get(dest):
                return False
        else:
            typed_requirement = _typed_value(flag, requirement)
            if value != typed_requirement and not (value is None and typed_requirement is False):
                return False
    return True


def _strip_flag(flag):
    return flag.lstrip("-")


def _tag_legend_params(figs, params):
    for fig in figs if isinstance(figs, list) else [figs]:
        for f in fig if isinstance(fig, list) else [fig]:
            if f is not None:
                f._legend_params = params
    return figs


def _compose_title(suffix, *parts):
    """Joins optional leading title parts (e.g. a metric name, gated on
    whether this figure is one of several -- see plot()'s docstring) with
    _title_suffix's " — <restriction>" text (bare, no leading " — ") using a
    single " — " separator, dropping any empty piece -- so a figure's title
    never carries boilerplate like "Cross-run"/"comparison", just whatever
    identifies its content (metric, if given) and the configs it covers
    (the restriction)."""
    bare_suffix = suffix[len(" — "):] if suffix.startswith(" — ") else suffix.strip()
    pieces = [p for p in parts if p] + ([bare_suffix] if bare_suffix else [])
    return " — ".join(pieces)


def _axis_flag_label(name):
    """Display text for an agg axis's own name (self._agg_axis_names, already
    canonicalized to a CLI flag's long form and _strip_flag-ed by
    resolve_configs_cartesian -- see _canonical_flag_name) as an x/y axis
    label -- dashes/underscores become spaces (e.g. "window-size" -> "window
    size", "x_bin" -> "x bin"), a tuple axis's raw member tokens (e.g.
    ("--ap", "--annealing"), NOT yet canonicalized/stripped -- unlike a plain
    flag axis's name, resolve_configs_cartesian stores a tuple axis's name
    as-is) are each canonicalized and joined with " / ", and the ""
    placeholder axis (see resolve_configs_cartesian's agg docstring) has no
    label at all."""
    if isinstance(name, tuple):
        return " / ".join(_axis_flag_label(_canonical_flag_name(n)) for n in name)
    if not name:
        return ""
    return _strip_flag(name).replace("-", " ").replace("_", " ")


def _tuple_member_label(flag, value):
    if value is None or value is False:
        return None
    name = "beta'" if flag == "--beta-prime" else _strip_flag(_canonical_flag_name(flag))
    return f"{name}={value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else f"{name}={value}"


def _pool_axes_label(pool_axes, exclude=()):
    """Display text for resolve_configs_cartesian's pooled (non-agg) axes --
    e.g. {"-w": ["50k"], "-s": ["40k"]} -> "w 50k s 40k" -- used as
    plot's title suffix so a figure states what its configs were
    restricted to instead of how many there are. Flags in `exclude`
    (those still at CrossRunAnalysis.title_base's value) are left out."""
    parts = []
    for flag, spec in pool_axes.items():
        if flag in exclude:
            continue
        dest = _FLAG_TO_DEST.get(flag)
        if spec is None or spec is False or (dest in _RECORDED_DEFAULTS and spec == _RECORDED_DEFAULTS[dest]):
            continue
        name = _strip_flag(flag) if flag in _FLAG_TO_DEST else flag
        val_str = ",".join(str(v) for v in spec) if isinstance(spec, list) else str(spec)
        parts.append(f"{name} {val_str}")
    return " ".join(parts)


def _combo_label_part(flag, requirement):
    """Display text for one axis's state in a suffix combo label: a True value renders as
    its bare value (value flags, e.g. "repulsion" for {"--ap": "repulsion"}) or bare flag
    name stripped of leading dashes (bool flags, e.g. "annealing"); a False or None (missing
    from args.json) value is dropped entirely (returns None), since it's the
    default/off state -- a config with nothing left renders as "(base)". -w/-s
    specifically render as "w"/"s" glued directly onto their value, shortened to a
    k/m suffix like the rest of the codebase (e.g. {"-w": "50000"} -> "w50k"). A
    fractional -s value (e.g. 0.8, meaning 80% of whatever window it ends up
    paired with -- see _resolve_step_fraction) renders as its bare fraction
    ("s0.8") instead, since it has no single absolute step size to format
    until matched against a specific window."""
    if requirement is True:
        return _strip_flag(_canonical_flag_name(flag))
    if requirement is False or requirement is None:
        return None
    if flag == "-s" and (frac := _as_step_fraction(requirement)) is not None:
        return f"s{frac:g}"
    if flag in ("-w", "-s"):
        return f"{_strip_flag(flag)}{_format_param_value(_typed_value(flag, requirement))}"
    return str(requirement)


def _implicit_flag_part(flag, recorded):
    dest = _FLAG_TO_DEST[flag]
    value = recorded.get(dest)
    if dest == "window_size":
        return _combo_label_part("-w", value)
    if dest == "step_size":
        return None if value is None or value == recorded.get("window_size") else _combo_label_part("-s", value)
    if value is True:
        return _strip_flag(flag)
    if value is None or value is False or value == _PARSER._option_string_actions[flag].default:
        return None
    return _tuple_member_label(flag, value)


def _combo_label(flags, empty_label):
    parts = [part for flag, req in flags.items() if (part := _combo_label_part(flag, req)) is not None]
    return " ".join(parts) if parts else empty_label


def _cartesian_axis_states(flag, value_spec):
    """A list value_spec is an explicit multi-value axis, used as-is -- e.g.
    {"-w": ["50k", "100k", "250k"]}. A bare (non-list) value_spec pins the axis to
    exactly that one value, equivalent to a single-element list -- e.g.
    {"--annealing": True} matches only annealing=True, never pooling in
    annealing=False too. None is a valid list element (or bare value): it means
    the flag is missing from args.json entirely -- e.g. {"--lam": [None, 1.0,
    1.5]} includes runs with no recorded lambda alongside 1.0 and 1.5."""
    if isinstance(value_spec, list):
        return [{flag: v} for v in value_spec]
    return [{flag: value_spec}]


def _as_step_fraction(value):
    """value (int/float/numeric string) as a float if it's in (0, 1] -- a
    step-size fraction, else None (including for bools and anything
    unparseable). caster.py's own step_size_or_fraction only treats (0, 1)
    (exclusive) as a fraction -- 1.0 falls through to int_or_abbrev there,
    which can't parse a bare "1.0" (no k/m suffix) and raises -- but this
    analysis tool's fraction resolution (_resolve_step_fraction) has no such
    parsing conflict, so 1.0 is included here as a convenience meaning
    "step == window"."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        frac = float(value)
    except (TypeError, ValueError):
        return None
    return frac if 0 < frac <= 1 else None


def _resolve_step_fraction(flags):
    """If `flags` (a {cli_flag: requirement} dict, e.g. resolve_configs_
    cartesian's per-(window, step)-combo `merged`) holds "-s" as a bare
    fraction in (0, 1] alongside a concrete "-w" value, returns a copy with
    "-s" replaced by the resolved absolute step (round(fraction * window),
    floored at 1) -- matching args.json's always-absolute recorded
    step_size, since a fraction is never itself a real recorded value.
    Returns `flags` unchanged otherwise (including when -s isn't a
    fraction or -w is missing)."""
    if "-s" not in flags or "-w" not in flags:
        return flags
    frac = _as_step_fraction(flags["-s"])
    if frac is None:
        return flags
    w_int = _typed_value("-w", flags["-w"])
    flags = dict(flags)
    flags["-s"] = max(1, round(frac * w_int))
    return flags


def _assert_single_run(df):
    """Raises if `df` (the frame about to be sliced into one _draw_stat
    bar/violin call -- i.e. one hue) spans more than one distinct
    benchmark run directory ("_source_dir", tagged per-row by
    _load_raw_config_df). A config's `dirs` list can hold 2+ directories
    when -w or another axis is pooled rather than agg'd (see
    resolve_configs_cartesian) -- silently averaging rows from separate
    runs into one bar/violin can hide real run-to-run variance or mix
    incompatible runs without anyone noticing. Checked AFTER row_filter/
    panel_col filtering (on the same frame _draw_stat's vals column gets
    pulled from), not just against len(dirs) -- either filter can drop an
    entire run's rows out of this specific hue, which len(dirs) alone
    wouldn't catch. No-op when df is empty or predates "_source_dir"."""
    if df.empty or "_source_dir" not in df.columns:
        return
    dirs = df["_source_dir"].unique()
    assert len(dirs) <= 1, (
        f"one bar/violin is pooling rows from {len(dirs)} distinct benchmark run "
        f"directories: {sorted(dirs)} -- averaging them together into a single "
        "hue hides real run-to-run variance; add the pooled axis to `agg` "
        "instead of leaving it pooled"
    )


def _spread(vals, errorbar):
    assert errorbar in ("sd", "sem"), errorbar
    if len(vals) < 2:
        return 0.0
    return vals.std() / np.sqrt(len(vals)) if errorbar == "sem" else vals.std()


def _draw_stat(ax, pos, vals, color, width, plot_type, errorbar="default"):
    """Draws one config's distribution at x=pos: plot_type="bar" is mean+sem
    errorbar; "violin" is the
    distribution as a standard box-and-whisker would show it -- the body's
    KDE is built only from values inside the Tukey whisker
    [Q1-1.5*IQR, Q3+1.5*IQR] (matplotlib's own showextrema draws the data's
    raw min/max instead, which is what made the tails so long with any real
    outliers), with the whisker itself drawn as a line and anything beyond it
    scattered as faint outlier dots so it isn't silently dropped. Falls back
    to a single dot when there's only one value (violinplot's KDE needs at
    least 2). errorbar: "default" keeps those (bar: sem; violin: whiskers
    only); "sd"/"sem" sets the bar's error and overlays mean+-that on violins;
    None draws neither.
    Returns the (lo, hi) "inlier" range actually worth showing to
    scale (excludes outlier dots) -- None if nothing was drawn -- so the
    caller can clip the axis to it and mark the break (see
    _clip_axis_to_whiskers)."""
    vals = pd.Series(vals).dropna()
    if vals.empty:
        return None
    if plot_type == "bar":
        mean = vals.mean()
        ax.bar(pos, mean, width=width * 0.92, color=color, zorder=2)
        if errorbar is None:
            return (mean, mean)
        err = _spread(vals, "sem" if errorbar == "default" else errorbar)
        ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor="black",
                     elinewidth=1.0, capsize=2, zorder=3)
        return (mean - err, mean + err)
    elif len(vals) > 1:
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        whisker_lo = max(vals.min(), q1 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q3 + 1.5 * iqr)
        inliers = vals[(vals >= whisker_lo) & (vals <= whisker_hi)]
        outliers = vals[(vals < whisker_lo) | (vals > whisker_hi)]
        bounds = None
        if len(inliers) > 1:
            vp = ax.violinplot([inliers.values], positions=[pos], widths=width * 0.92,
                                showmeans=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.65)
            vp["cmeans"].set_edgecolor("black")
            vp["cmeans"].set_linewidth(1.0)
            if errorbar is not None:
                ax.plot([pos, pos], [inliers.min(), inliers.max()], color="black", linewidth=1.0, zorder=2.5)
            bounds = (inliers.min(), inliers.max())
        elif not inliers.empty:
            ax.scatter([pos], [inliers.iloc[0]], color=color, s=18, zorder=3)
            bounds = (inliers.iloc[0], inliers.iloc[0])
        if not outliers.empty:
            ax.scatter([pos] * len(outliers), outliers, color=color, s=6, alpha=0.4,
                       edgecolor="none", zorder=2)
        if errorbar in (None, "default"):
            return bounds
        mean, err = vals.mean(), _spread(vals, errorbar)
        ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3.5)
        lo, hi = mean - err, mean + err
        return (min(bounds[0], lo), max(bounds[1], hi)) if bounds else (lo, hi)
    else:
        v = vals.iloc[0]
        ax.scatter([pos], [v], color=color, s=18, zorder=3)
        return (v, v)


_AUTOLOG_MIN_DECADES = 2.0


def _resolve_logy(logy, dfs, metrics):
    """logy=True -> True. logy="auto" -> True when the pooled values of
    `metrics` (one name or an iterable, as in _apply_bounded_yaxis) across
    every config in `dfs` are all strictly positive and their 1st-99th
    percentile range spans >= _AUTOLOG_MIN_DECADES decades -- e.g. variances
    across window sizes. Anything else (incl. any value <= 0, which a log
    axis can't show) stays linear."""
    if logy != "auto":
        return bool(logy)
    names = (metrics,) if isinstance(metrics, str) else tuple(metrics)
    chunks = [d[m].to_numpy(dtype=float) for d in dfs.values() for m in names if m in d.columns]
    if not chunks:
        return False
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or vals.min() <= 0:
        return False
    lo, hi = np.percentile(vals, [1, 99])
    return math.log10(hi / lo) >= _AUTOLOG_MIN_DECADES


def _legend_below_axes(fig, handles, labels, top=0.95, pad_px=6, right=1.0, **kwargs):
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False, **kwargs)
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    legend_h = (legend.get_window_extent(renderer).height + 2 * pad_px) / fig_h_px
    fig.tight_layout(rect=[0, legend_h, right, top])
    fig.canvas.draw()
    axes_bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes if ax.get_visible())
    legend.set_bbox_to_anchor((0.5, (axes_bottom - pad_px) / fig_h_px), transform=fig.transFigure)
    return legend


def _apply_bounded_yaxis(ax, metrics):
    """Pins `ax`'s y-axis to [0, 1] with 0.2-step ticks when `metrics` (a
    single metric name, or an iterable of them sharing one combined subplot
    -- see plot()'s tuple-metric panels) are ALL in _BOUNDED_METRICS --
    returns True if it did, so the caller can skip _clip_axis_to_whiskers
    (which would otherwise override this fixed range based on this cell's
    own data)."""
    names = (metrics,) if isinstance(metrics, str) else tuple(metrics)
    if not names or not all(m in _BOUNDED_METRICS for m in names):
        return False
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(_BOUNDED_YTICKS)
    return True


def _clip_axis_to_whiskers(ax, bounds, pad=1.2):
    """After a cell's _draw_stat calls, clips `ax`'s y-axis to the configs'
    combined inlier range (`bounds`: list of (lo, hi) tuples, None entries
    ignored) with `pad` headroom, and draws a break marker at the top if that
    actually clips something off (an outlier scatter dot sitting above it) --
    matplotlib's violinplot has no built-in notion of a boxplot-style
    whisker, so without this an axis with any real outliers autoscales to fit
    them, squashing the rest of the distribution flat."""
    bounds = [b for b in bounds if b is not None]
    if not bounds:
        return
    hi = max(b[1] for b in bounds)
    lo = min(b[0] for b in bounds)
    visible_top = hi * pad if hi > 0 else (hi + 1.0)
    _, cur_top = ax.get_ylim()
    if cur_top > visible_top * 1.05:
        ax.set_ylim(lo / pad if ax.get_yscale() == "log" else min(0.0, lo), visible_top)
        _draw_break_marker(ax)


def _draw_break_marker(ax):
    """Small double-diagonal "//" marks at the top-left/top-right corners of
    `ax`, the standard convention for "this axis is truncated here"."""
    d = 0.018
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.1)
    for x0 in (0.0, 1.0):
        ax.plot([x0 - d, x0 + d], [1.0 - d, 1.0 + d], **kwargs)
        ax.plot([x0 - d, x0 + d], [1.0 - 2.5 * d, 1.0 - 0.5 * d], **kwargs)


def _param_value_sort_key(value):
    if value is None:
        return (0, "")
    if isinstance(value, (int, float)):
        return (1, value)
    return (2, str(value))


def _ordered_unique(values):
    """Distinct values from `values` in first-seen order (not sorted) --
    used for _plot_grouped's bar/hue/col/row axes, so their order matches
    however resolve_configs_cartesian generated them (already the caller's
    agg-value order) instead of being re-sorted."""
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


_LEADING_NUM_RE = re.compile(r"[-+]?\d+\.?\d*")
_ABBREV_TOKEN_RE = re.compile(r"^[A-Za-z]*(-?\d+\.?\d*[km]?)$", re.IGNORECASE)


def _token_sort_key(token):
    """Sorts a heatmap/subplot-grid axis's raw label tokens numerically when
    possible instead of lexicographically -- "80000" must sort after
    "40000", not before it. token is usually a string, but can be the Python
    None self._config_axes falls back to for an axis slot that doesn't exist
    for a given config (e.g. y when a heatmap's agg only has 1 real level) --
    float(None) raises TypeError, not ValueError, so that's caught too
    rather than propagating out of what looks like a harmless sort call;
    None sorts last (after both numeric and non-numeric strings).

    A -w/-s display token (_combo_label_part, e.g. "w50k", "w100") glues a
    bare flag letter onto a k/m-abbreviated value, which a whole-token
    float() parse can't see through -- and treating "w50k"/"w10k" as
    literal-string-order or via the bin-range fallback below (which reads
    only the leading digits, "50"/"10", ignoring the k/m multiplier)
    resorts to a lexicographic-ish order that would land e.g. "w10" after
    "w50k" instead of near "w100". So a token matching
    <letters><number><k/m?> (anchored, nothing else in the string) is
    parsed with int_or_abbrev (same parser CLI flags like -w/-s/-L use) once
    the leading letters are stripped, ahead of the bin-range fallback.

    A whole-token float() parse fails for a bin-range label like
    "[0.000,0.067]"/"(0.067,0.133]" (_compute_hd_bin_edges), so this falls
    back to that label's own leading number (its lower bound) rather than
    the raw string -- plain string comparison would otherwise sort the
    first bin's "[..." after every other bin's "(..." (since '(' < '['
    lexicographically), landing the lowest bin last instead of first. Only
    a token with no number at all (e.g. a plain category name) falls back
    to the raw string -- except a merged_category value (e.g. "recombination
    up", "admixture"), which sorts by _MERGED_CATEGORY_ORDER instead: several
    of these (e.g. "10X up"/"10X down") contain a leading number that would
    otherwise misroute them into the numeric branch below by their category
    name's incidental digits rather than their actual (non-numeric) meaning,
    so this is checked first, before any numeric parse is attempted."""
    if token is None:
        return (2, "")
    if isinstance(token, str) and token in _MERGED_CATEGORY_ORDER:
        return (-1, _MERGED_CATEGORY_ORDER.index(token))
    try:
        return (0, float(token))
    except (TypeError, ValueError):
        pass
    if isinstance(token, str):
        if m := _ABBREV_TOKEN_RE.match(token):
            try:
                return (0, float(int_or_abbrev(m.group(1))))
            except ValueError:
                pass
        if m := _LEADING_NUM_RE.search(token):
            return (0, float(m.group()))
    return (1, token)


# Total multiplicative spread hue/sub-metric bars or violins occupy around
# their shared x position under options={"logx": True} -- the log-space
# analog of bar_width/sub_width's fixed additive spread in the usual
# categorical-index layout. Shared by plot()'s single-level-agg path and
# _plot_grouped so the two look consistent.
_LOG_SPAN = 0.6


def _token_to_number(token):
    """Numeric value behind a bar/x-axis token (e.g. "w50k" -> 50000.0, "1k"
    -> 1000.0), reusing _token_sort_key's own token parsing so the two stay
    consistent -- None when the token isn't numeric at all (a plain category
    label, a merged_category name, or the None placeholder for a missing
    axis slot). Used by plot()'s options={"logx": True} to decide whether an
    agg axis can be repositioned onto a real log-numeric scale instead of
    its usual evenly-spaced categorical index."""
    kind, value = _token_sort_key(token)
    return value if kind == 0 else None


def _heatmap_axis(vals, scaled=False):
    nums = [_token_to_number(v) for v in vals]
    if scaled and vals and all(n is not None for n in nums):
        order = sorted(range(len(vals)), key=lambda i: nums[i])
        vals, nums = [vals[i] for i in order], [nums[i] for i in order]
        log = nums[0] > 0 and nums[-1] / nums[0] > 20
        return vals, _axis_edges(nums, log=log), nums, "log" if log else "linear"
    vals = sorted(vals, key=_token_sort_key)
    return vals, list(range(len(vals) + 1)), [i + 0.5 for i in range(len(vals))], "linear"


def _axis_edges(nums, log=False):
    """N+1 cell-boundary edges for N sorted numeric axis values, each
    boundary at the midpoint between its neighbors (half the neighbor gap
    padded past the first/last value) -- pcolormesh needs edges, not
    centers, to draw variable-sized cells proportional to the real gaps
    between values instead of one fixed size per category. log=True does the
    same midpoint math in log-space (geometric mean of neighbors) so edges
    stay visually centered on their tick once the axis itself is
    log-scaled -- a linear midpoint would land off-center under a log
    transform."""
    if log:
        return [math.exp(e) for e in _axis_edges([math.log(n) for n in nums])]
    if len(nums) == 1:
        pad = max(abs(nums[0]) * 0.5, 1.0)
        return [nums[0] - pad, nums[0] + pad]
    edges = [nums[0] - (nums[1] - nums[0]) / 2]
    edges += [(nums[i] + nums[i + 1]) / 2 for i in range(len(nums) - 1)]
    edges.append(nums[-1] + (nums[-1] - nums[-2]) / 2)
    return edges


def _format_param_value(value):
    """Any number whose magnitude is >= 1000 gets a k/m suffix, one decimal
    place, trailing ".0"/zeros trimmed -- e.g. 50000 -> "50k", 1600 -> "1.6k"
    (not just exact multiples of 1000/1e6, unlike an earlier version of this)."""
    if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
        return str(value)
    for threshold, suffix in ((1_000_000, "m"), (1000, "k")):
        if abs(value) >= threshold:
            text = f"{value / threshold:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(value)


def _param_summary_lines(dirs):
    """One line per mirrored caster/phlag param that isn't constant across
    `dirs`' own args.json (a constant param says nothing about what's actually
    pooled into this config, so it's skipped) -- a varying store_true flag
    renders as its bare name (e.g. "annealing"), anything else as
    "label: v1,v2,..." (e.g. "window size: 50k,250k")."""
    recorded = [_read_args_json(Path(d) / "args.json") for d in dirs]
    lines = []
    for dest, flag, is_store_true in _MIRRORED_SPECS:
        values = {r.get(dest) for r in recorded}
        if len(values) <= 1:
            continue
        if is_store_true:
            lines.append(_strip_flag(flag))
        else:
            label = _PARAM_LABELS.get(dest, dest.replace("_", " "))
            formatted = ",".join(_format_param_value(v) for v in sorted(values, key=_param_value_sort_key))
            lines.append(f"{label}: {formatted}")
    return lines


def _is_true(series):
    return series.astype(str).str.strip() == "True"


@functools.cache
def _windowless_runs_df(source):
    from phlag.utils import get_data_dir, get_simulation_categories

    if source == PAPER_SOURCE:
        return _windowless_runs_df("gtrees")
    root = Path(PHLAG_SOURCE_ROOTS[source])
    stats = BenchmarkStats()
    rows = []
    for leaf_dir in discover_leaf_dirs(get_data_dir() / "simulations"):
        rel_leaf = leaf_dir.relative_to(get_data_dir() / "simulations").as_posix()
        cats = get_simulation_categories(str(leaf_dir))
        column = CATEGORY_TO_COLUMN.get(tuple(cats[:2])) if cats else None
        if column is None:
            continue
        for report in sorted((root / rel_leaf).glob("*/report.tsv")):
            if report.parent.name in DEFAULT_CONCAT_PATTERNS:
                record = stats._build_record(report, leaf_dir.name, rel_leaf, cats[0], cats[1], column, report.parent.name)
                rows.append(stats.run_row(record))
    df = pd.DataFrame(rows, columns=BenchmarkStats.RUN_COLUMNS)
    df[[c for c in df.columns if _GT_STATS_COLUMN.match(c)]] = np.nan
    return df


_HIDE_ADMIXTURE = False


@contextlib.contextmanager
def _hiding_admixture(hide):
    global _HIDE_ADMIXTURE, FIGURE_COLUMNS
    saved = _HIDE_ADMIXTURE, FIGURE_COLUMNS, dict(_DATA_COLUMN_AXES)
    if hide:
        _HIDE_ADMIXTURE = True
        FIGURE_COLUMNS = [c for c in FIGURE_COLUMNS if c != COL_ADMIXTURE]
        _DATA_COLUMN_AXES.update(x_bin=list(BRANCH_LENGTH_BINS), column=FIGURE_COLUMNS)
    try:
        yield
    finally:
        _HIDE_ADMIXTURE, FIGURE_COLUMNS = saved[:2]
        _DATA_COLUMN_AXES.update(saved[2])


def _drop_hidden_admixture(df):
    if _HIDE_ADMIXTURE and "category" in df.columns:
        return df[df["category"] != "admixture"]
    return df


def _load_raw_config_df(dirs, get_args, bin_specs=None):
    """Concats `dirs`' runs.tsv into one frame, tagging every row with its
    source dir's window_size as "_window_size" (a config's dirs, see
    resolve_configs_cartesian, can span several window sizes when -w is a
    pooled, non-agg axis) and adding the same merged_category/bin_specs
    derived columns _load_configs_runs does -- but with NO row_filter
    applied. get_args: run_dir -> args.json dict (pass a CrossRunAnalysis
    instance's self._get_args to reuse its cache). Used directly by
    _metric_by_axis_value, which needs a config's full underlying data
    regardless of any row-filter (x_bin, fraction_bin, em_gt_hd_bin, ...)
    agg axis -- otherwise the legend score it computes would shift just
    because a display-only row-filter axis was added to agg, splitting the
    same underlying dirs into more, smaller row-filtered buckets that then
    get pooled as if they were separate configs."""
    frames = []
    for d in dirs:
        p = Path(d) / "runs.tsv"
        source = _windowless_source_of(d)
        if source is not None:
            frame = _windowless_runs_df(source).copy()
        elif p.exists():
            frame = pd.read_csv(p, sep="\t")
        else:
            continue
        frame["_window_size"] = get_args(d).get("window_size")
        # Per-row source dir, so a post-load row_filter/panel_col filter
        # (either of which can drop an entire run's rows) can still be
        # checked against what actually survives into one hue -- see
        # _assert_single_run. len(dirs) alone can't catch that, since it
        # only reflects what was UNIONED in before filtering.
        frame["_source_dir"] = str(d)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True).copy() if frames else pd.DataFrame()
    if not df.empty and BRANCH_LENGTH_SCALE != 1.0 and "branch_length_cu" in df.columns:
        scaled = (df["category"] != "admixture") & df["branch_length_cu"].notna()
        df["x_bin"] = df["x_bin"].astype(object)
        df.loc[scaled, "x_bin"] = (df.loc[scaled, "branch_length_cu"] * BRANCH_LENGTH_SCALE).apply(
            lambda v: assign_bin(v, BRANCH_LENGTH_BIN_EDGES))
        df.loc[scaled & df["x_bin"].isna(), "in_panel_a"] = False
    if not df.empty and "exclusion" in df.columns:
        failed = (df["exclusion"] == _NO_METRICS_EXCLUSION) & df["fraction_bin"].notna()
        df.loc[failed, _FAILED_RUN_ZERO_METRICS] = 0.0
        df.loc[failed, "in_panel_b"] = True
        df.loc[failed & df["x_bin"].notna(), "in_panel_a"] = True
    if not df.empty and "category" in df.columns and "subcategory" in df.columns:
        df["merged_category"] = _merged_category(df["category"], df["subcategory"])
    if not df.empty and "pattern" in df.columns:
        df["outlier_pct"] = _outlier_pct(df["pattern"])
    if not df.empty and bin_specs:
        for bin_col, (source_col, edges) in bin_specs.items():
            if source_col in df.columns:
                df[bin_col] = df[source_col].apply(lambda v: assign_bin(v, edges))
    return _drop_hidden_admixture(df)


def _load_configs_runs(configs, row_filters=None, bin_specs=None, get_args=None):
    """row_filters: {label: {col: value, ...}}, from resolve_configs_cartesian
    agg'ing by a _DATA_COLUMN_AXES name (e.g. "x_bin") -- restricts that
    config's rows to matching values after loading, since (unlike a flag) a
    row-level column can't narrow which dirs get read in the first place.
    bin_specs: {bin_col: (source_col, edges)}, from
    CrossRunAnalysis.compute_hd_bin_edges -- adds a bin_col column (e.g.
    "em_hd_bin") via assign_bin(source_col value, edges), same derived-column
    treatment as merged_category, before row_filters get applied (so a
    row_filter naming a bin_col, e.g. from agg=["em_hd_bin"], has something
    to filter on). get_args: run_dir -> args.json dict, defaults to an
    uncached _read_args_json call; pass a CrossRunAnalysis instance's
    self._get_args to reuse its args.json cache. See _load_raw_config_df for
    the (row_filter-less) per-dir loading this wraps."""
    get_args = get_args or (lambda d: _read_args_json(Path(d) / "args.json"))
    result = {}
    for label, dirs in configs:
        df = _load_raw_config_df(dirs, get_args, bin_specs)
        row_filter = (row_filters or {}).get(label)
        if row_filter and not df.empty:
            for col, value in row_filter.items():
                df = df[df[col] == value]
        result[label] = df
    return result


def _compute_hd_bin_edges(values):
    """HD_NUM_BINS equal-width bins spanning `values`' observed range, rounded
    to 3 decimals at the ends -- mirrors BenchmarkStats._compute_hd_bins in
    benchmark.py exactly (kept as a plain function here since that one's a
    method reading self.runs, not a value list). Returns (label, low, high)
    triples for assign_bin; [] if `values` is empty."""
    if not values:
        return []
    lo = round(min(values), 3)
    hi = round(max(values), 3)
    if hi <= lo:
        hi = lo + 0.001
    width = (hi - lo) / HD_NUM_BINS
    edges = []
    for i in range(HD_NUM_BINS):
        bin_lo = lo + i * width
        bin_hi = lo + (i + 1) * width if i < HD_NUM_BINS - 1 else hi
        label = f"[{bin_lo:.3f},{bin_hi:.3f}]" if i == 0 else f"({bin_lo:.3f},{bin_hi:.3f}]"
        assign_lo = bin_lo - 1e-9 if i == 0 else bin_lo
        edges.append((label, assign_lo, bin_hi))
    return edges


def x_bins_for_column(column):
    return ADMIXTURE_BINS if column == COL_ADMIXTURE else BRANCH_LENGTH_BINS


def x_axis_label_for_column(column):
    if column == COL_ADMIXTURE:
        return f"Divergence time (split at {ADMIXTURE_DIVERGENCE_THRESHOLD_MYR:g} Myr)"
    return "Branch length (CU)"


def _config_color(i):
    return _CONFIG_PALETTE[i % len(_CONFIG_PALETTE)]


def _shade_color(base_color, frac):
    """frac in [0,1]: 0 -> lightest tint of base_color, 1 -> darkest shade. Used to
    encode an ordered x_bin position via marker shade instead of a text label."""
    r, g, b = matplotlib.colors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(max(l * (1.5 - 1.1 * frac), 0.08), 0.92)
    return colorsys.hls_to_rgb(h, l, s)


def _gradient_color(base_color, frac):
    h, _, s = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(base_color))
    return colorsys.hls_to_rgb(h, 0.8 - 0.65 * frac, s)


def _tpr_fpr_axes(ax, full=False):
    hi = 1.0 if full else 0.5
    ax.set_xlim(-0.03 * hi, 1.03 * hi)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, hi, 6 if full else 5))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.plot([0, hi], [0, hi], color="#cfcfc9", linewidth=1.0, linestyle="--", zorder=1)


def _cross_run_grid(configs, title, cell_plot_fn, categorical_x=True):
    dfs = _load_configs_runs(configs)
    n_rows, n_cols = len(FRACTION_BINS), len(FIGURE_COLUMNS)
    width_ratios = [len(x_bins_for_column(c)) for c in FIGURE_COLUMNS]
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
        squeeze=False, gridspec_kw={"width_ratios": width_ratios},
    )
    for row_idx, fraction_bin in enumerate(FRACTION_BINS):
        for col_idx, column in enumerate(FIGURE_COLUMNS):
            ax = axes[row_idx][col_idx]
            x_bins = x_bins_for_column(column)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            cell_plot_fn(ax, dfs, column, fraction_bin, x_bins)
            if row_idx == 0:
                ax.set_title(column, fontsize=9.5, pad=8)
            if row_idx == n_rows - 1 and categorical_x:
                ax.set_xlabel(x_axis_label_for_column(column), fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"alt proportion {fraction_bin}", fontsize=8.5)
    _suptitle(fig, title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def _recorded_value(recorded, dest):
    value = recorded.get(dest)
    return _RECORDED_DEFAULTS.get(dest) if value is None else value


def _hstack_figs(figs):
    imgs = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        imgs.append(plt.imread(buf))
        plt.close(fig)
    dpi = figs[0].dpi
    total_w, max_h = sum(img.shape[1] for img in imgs), max(img.shape[0] for img in imgs)
    out = plt.figure(figsize=(total_w / dpi, max_h / dpi), dpi=dpi)
    left = 0
    for img in imgs:
        h, w = img.shape[:2]
        ax = out.add_axes([left / total_w, 1 - h / max_h, w / total_w, h / max_h])
        ax.imshow(img)
        ax.set_axis_off()
        left += w
    return out


def _suptitle(fig, text, fontsize=12, **kwargs):
    width = max(20, int(fig.get_figwidth() * 72 / (fontsize * 0.6)))
    lines = textwrap.wrap(text, width, break_on_hyphens=False) or [""]
    if len(lines) > 1:
        fig.set_figheight(fig.get_figheight() + (len(lines) - 1) * fontsize * 1.2 / 72)
    return fig.suptitle("\n".join(lines), fontsize=fontsize, **kwargs)


def _fit_fig_to_axes(fig, axes, ax_w, ax_h, rect, iters=3):
    n_rows, n_cols = axes.shape
    fig.set_size_inches(n_cols * ax_w + 40, n_rows * ax_h + 40)
    for _ in range(iters):
        fig.tight_layout(rect=rect)
        pos = axes[0][0].get_position()
        fig_w, fig_h = fig.get_size_inches()
        fig.set_size_inches(fig_w + n_cols * (ax_w - pos.width * fig_w), fig_h + n_rows * (ax_h - pos.height * fig_h))
    fig.tight_layout(rect=rect)


def _title_bottom(fig, default=0.95):
    if fig._suptitle is None or not fig._suptitle.get_text():
        return default
    fig.canvas.draw()
    return min(default, (fig._suptitle.get_window_extent(fig.canvas.get_renderer()).y0 - 4) / fig.bbox.height)


def _below_title_top(fig, top):
    """tight_layout rect top that lands the axes at `top` even though
    tight_layout also reserves the suptitle's own height plus a pad (its
    default pad=1.08 x font size) below rect's top -- for figures whose
    axes top is already measured below the suptitle."""
    if fig._suptitle is None or not fig._suptitle.get_text():
        return top
    fig.canvas.draw()
    rel_height = fig._suptitle.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    pad = 1.08 * matplotlib.rcParams["font.size"] / 72 / fig.get_figheight()
    return min(1.0, top + rel_height + pad)
