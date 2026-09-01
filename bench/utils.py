import colorsys
import itertools
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from bench.benchmark import (
    ADMIXTURE_BINS,
    BRANCH_LENGTH_BINS,
    CASTER_ARG_SPECS,
    COL_ADMIXTURE,
    FIGURE_COLUMNS,
    FRACTION_BINS,
    HD_NUM_BINS,
    PHLAG_ARG_SPECS,
    _build_parser,
    _read_args_json,
    assign_bin,
)
from phlag.utils import ADMIXTURE_DIVERGENCE_THRESHOLD_MYR

STORE_ROOT = "store/phlag"

SWEEP_METRICS = ["em_hd", "roc_auc", "f1"]

_METRIC_PANEL_COLUMN = {
    "f1": "in_panel_a",
    "em_hd": "in_panel_em_divergence",
    "em_gt_hd": "in_panel_em_divergence",
    "mean_relerr_agg": "in_panel_relerr",
    "covar_relerr_agg": "in_panel_relerr",
}

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
    ("(none)") or missing state ("no-<flag>") -- see _combo_label_part --
    since that's the closest thing to a real baseline/control. Falls back to
    col_vals[0] (the first/leftmost column, in whatever order agg produced
    it) when no such state is present, e.g. a tuple axis's realized combos
    (_discover_flag_tuple_values sorts a fully-missing/off combo first when
    one exists on disk, so this still lands on it there too) or a plain
    value list like -w's, which has no notion of "off" at all."""
    for cv in col_vals:
        if cv == "(none)" or (isinstance(cv, str) and cv.startswith("no-")):
            return cv
    return col_vals[0]

# Metrics that are proportions or bounded distances (never outside [0, 1]) --
# their bar/violin y-axis is pinned to [0, 1] with 0.2-step ticks (see
# _apply_bounded_yaxis) instead of autoscaling/whisker-clipping to whatever a
# given cell's data happens to span, so figures stay comparable to each other.
_BOUNDED_METRICS = {"f1", "em_hd", "em_gt_hd", "roc_auc", "transition_null_to_alt", "transition_alt_to_null"}
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
_DYNAMIC_ROW_AXES = {"category", "subcategory", "merged_category", "em_hd_bin", "em_gt_hd_bin", "pattern"}


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
# Every option string (both short and long form, e.g. "-n" AND "--normalize")
# that the real parser registers for a mirrored dest -- not just the one
# canonical token CASTER_ARG_SPECS/PHLAG_ARG_SPECS happens to list, so a
# config dict can use either alias interchangeably, same as the actual CLI.
_MIRRORED_DESTS = {dest for dest, _, _ in _MIRRORED_SPECS}
_FLAG_TO_DEST = {
    option_string: action.dest
    for option_string, action in _PARSER._option_string_actions.items()
    if action.dest in _MIRRORED_DESTS
}
_PARAM_LABELS = {"emission_lambda": "lambda"}

_CONFIG_PALETTE = list(matplotlib.colormaps["tab10"].colors)


class _Not:
    """Marks a flags-dict value as "not this" -- used internally to auto-derive an
    axis's off-state from its on-state; never constructed directly by callers."""
    __slots__ = ("value",)
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"_Not({self.value!r})"


def _match_cache_key(flags):
    """Canonical, order-independent, hashable key for a find_matching_leaf_dirs
    `flags` dict -- used to memoize its result (see find_matching_leaf_dirs).
    _Not has no __eq__/__hash__ of its own (two separately-constructed
    _Not(False) instances are otherwise distinct by identity), so it's
    unwrapped to a ("NOT", value) tuple here instead of hashed directly."""
    return tuple(sorted(
        (flag, ("NOT", req.value) if isinstance(req, _Not) else req)
        for flag, req in flags.items()
    ))


class CrossRunAnalysis:
    """
    Compares finished phlag benchmark runs against each other, reading their
    runs.tsv/args.json straight off disk (no benchmark re-run). A config is a
    set of CLI flags (resolve_configs*), resolved to matching leaf run dirs via
    each run's own args.json, then all rendered via plot: pass "tpr_fpr" in
    `metrics` for the per-config-color TPR/FPR scatter grid, any other metric
    name for a bar/violin subplot.
    """

    def __init__(self, root=STORE_ROOT):
        self.root = root
        self.metrics = SWEEP_METRICS
        self._leaf_dirs = None
        self._row_filters = {}
        self._config_axes = {}
        self._agg_axis_names = {}
        self._restrictions = {}
        self._last_configs = None
        self._bin_specs = {}
        self._args_json_cache = {}
        self._match_cache = {}

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
            values.add(recorded.get(dest))
        return sorted(values, key=_param_value_sort_key)

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
        combos = set()
        for run_dir in self._all_leaf_dirs():
            if any(kw in str(run_dir) for kw in exclude_keywords):
                continue
            recorded = self._get_args(run_dir)
            combos.add(tuple(recorded.get(dest) for dest in dests))
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
            df = pd.read_csv(p, sep="\t")
            if col == "merged_category":
                if "category" in df.columns and "subcategory" in df.columns:
                    values.update(_merged_category(df["category"], df["subcategory"]).unique())
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
        _combo_label(flags, "(none)") per config. exclude_keywords: forwarded
        to find_matching_leaf_dirs."""
        configs = []
        for i, flags in enumerate(flags_specs):
            label = labels[i] if labels else _combo_label(flags, "(none)")
            dirs = self.find_matching_leaf_dirs(flags, exclude_keywords)
            if dirs:
                configs.append((label, dirs))
        return configs

    def resolve_configs_cartesian(self, axes, exclude_keywords=(), agg=()):
        """axes: {flag: value_spec} dict, one entry per flag -- e.g.
        {"-d": "gaussian", "-s": "1k", "--double-variance-init": True, "--lam": [1.0, 1.5]}.
        A list value_spec is an explicit multi-value axis, iterated as-is (no
        other state). Any other value_spec is a binary on/off toggle
        (on=value_spec, off=its negation) -- there's no separate "just pin this"
        mode, so a flag meant to stay fixed for every resulting config (e.g.
        -d/-s above) is still written as an on/off entry; if its negation
        genuinely has no matching runs on disk, that side is silently skipped
        (see resolve_configs), which is how a "fixed" flag ends up behaving
        like one in practice. exclude_keywords: see resolve_configs.

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
        (no auto on/off toggle). Left out of `axes` entirely, a
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
        docstring."""
        agg_seq = [agg] if isinstance(agg, str) else (list(agg) if agg else list(axes))
        agg_set = set(agg_seq)
        axes = dict(axes)

        data_col_axes = {}
        for name in list(axes.keys()) + agg_seq:
            if name in _DATA_COLUMN_AXES and name not in data_col_axes:
                data_col_axes[name] = axes.pop(name, _DATA_COLUMN_AXES[name])
            elif name in _DYNAMIC_ROW_AXES and name not in data_col_axes:
                data_col_axes[name] = axes.pop(name, None) or self._discover_row_values(name, exclude_keywords)
        agg_set -= set(data_col_axes)

        for flag in agg_seq:
            if flag not in axes and flag not in data_col_axes and flag in _FLAG_TO_DEST:
                axes[flag] = self._discover_flag_values(flag, exclude_keywords)
        pool_axes = {flag: spec for flag, spec in axes.items() if flag not in agg_set}
        pool_axis_states = [_cartesian_axis_states(flag, spec) for flag, spec in pool_axes.items()]
        pool_combos = list(itertools.product(*pool_axis_states)) if pool_axis_states else [()]
        restriction = _pool_axes_label(pool_axes)

        # One "level" per (still-existing) name in agg_seq -- a level's states are
        # (contribution_dict, is_flag, display_value) triples. A fractional -s
        # value's display is its raw fraction (see _combo_label_part); the
        # matching absolute step_size is resolved per (window, step) combo
        # below (see _resolve_step_fraction), not here.
        levels = []
        consumed_flags = set()
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
                for combo in self._discover_flag_tuple_values(name, exclude_keywords):
                    fs = dict(zip(name, combo))
                    disp = " ".join(
                        part for k, v in fs.items()
                        if (part := _combo_label_part(k, v)) is not None
                    )
                    states.append((fs, True, disp))
                consumed_flags.update(name)
            elif name in data_col_axes:
                states = [({name: v}, False, str(v)) for v in data_col_axes[name]]
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
        level_names = [_strip_flag(name) if name in _FLAG_TO_DEST else name for name, _ in levels]
        level_key_names = [name for name, _ in levels]

        configs = []
        level_combos = list(itertools.product(*[states for _, states in levels])) if levels else [()]
        for level_combo in level_combos:
            if not _x_bin_category_compatible(level_key_names, level_combo):
                continue
            agg_flags, row_filter, axis_values, label_bits = {}, {}, [], []
            for contribution, is_flag, disp in level_combo:
                axis_values.append(disp if disp else "(none)")
                # A flag level always contributes something to the label, even
                # "(none)" for its off state, so two configs whose only
                # difference is a flag's on/off state never collide onto the
                # same label text (which self._row_filters/_config_axes key
                # on) -- a data-column level's disp is never empty, so this
                # only changes behavior for flag levels.
                if disp or is_flag:
                    label_bits.append(disp if disp else "(none)")
                if is_flag:
                    agg_flags.update(contribution)
                else:
                    row_filter.update(contribution)
            label = " ".join(label_bits) or "(none)"
            dirs = []
            for pool_combo in pool_combos:
                merged = dict(agg_flags)
                for part in pool_combo:
                    merged.update(part)
                merged = _resolve_step_fraction(merged)
                dirs.extend(self.find_matching_leaf_dirs(merged, exclude_keywords))
            if not dirs:
                continue
            configs.append((label, dirs))
            self._row_filters[label] = row_filter
            self._config_axes[label] = axis_values
            self._agg_axis_names[label] = level_names
            self._restrictions[label] = restriction
        return configs

    def print_configs(self, configs=None, metrics=None, paths=False):
        """Prints each config's label, run count, and mean of every metric in
        `metrics` (pooled across all its runs -- same aggregate plot
        subplots show), followed by the args.json params that actually vary
        across its matching dirs (see _param_summary_lines) -- e.g. a config
        pooled via `agg` mixes lambda values, so its dirs disagree on lambda.
        configs defaults to self._last_configs -- the configs resolved by the
        most recent plot() call -- so you don't have to keep a configs
        variable around just to print it; pass configs= explicitly to
        override (e.g. to inspect a resolve_configs_cartesian call you
        didn't plot). metrics defaults to self.metrics -- the scalar metrics
        from the most recent plot() call -- so you don't have to repeat the
        same list; pass metrics= explicitly to override. paths=True also
        prints each config's sorted run dirs."""
        configs = self._last_configs if configs is None else configs
        metrics = self.metrics if metrics is None else metrics
        metric_values = {metric: self._metric_by_label(configs, metric) for metric in metrics}
        for label, dirs in configs:
            stats = ", ".join(f"{metric}={metric_values[metric][label]:.4f}" for metric in metrics)
            print(f"{label} ({len(dirs)} runs, {stats}):")
            for line in _param_summary_lines(dirs):
                print(f"  {line}")
            if paths:
                for d in sorted(dirs):
                    print(f"  {d}")

    # ---- cross-run PNG-metric grids ----

    def _plot_tpr_fpr_grid(self, configs, title="Cross-run TPR/FPR comparison"):
        colors = {label: _config_color(i) for i, (label, _) in enumerate(configs)}
        auc_by_label = self._metric_by_label(configs, "roc_auc")

        def cell(ax, dfs, column, fraction_bin, x_bins):
            ax.set_xlim(-0.015, 0.515)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([0.0, 0.125, 0.25, 0.375, 0.5])
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlabel("FPR", fontsize=8)
            ax.plot([0, 0.5], [0, 0.5], color="#cfcfc9", linewidth=1.0, linestyle="--", zorder=1)
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
                shades = [_shade_color(colors[label], i / max(len(points) - 1, 1)) for i in range(len(points))]
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

    def _plot_tpr_fpr_grouped(self, configs, title):
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
        _plot_grouped's legend -- see _metric_by_axis_value."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        line_vals = _ordered_unique(axis_val(label, 0) for label, _ in configs)
        point_vals = _ordered_unique(axis_val(label, 1) for label, _ in configs)
        col_vals = _ordered_unique(axis_val(label, 2) for label, _ in configs)
        row_vals = _ordered_unique(axis_val(label, 3) for label, _ in configs)

        by_key = {}
        for label, dirs in configs:
            key = (axis_val(label, 3), axis_val(label, 2), axis_val(label, 1), axis_val(label, 0))
            by_key[key] = (label, dirs)

        colors = {lv: _config_color(i) for i, lv in enumerate(line_vals)}
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
        show_line_legend = len(line_vals) > 1 or line_vals != [None]
        show_shade_legend = len(point_vals) > 1
        legend_rows = (
            (len(line_vals) if show_line_legend else 0)
            + (len(point_vals) + 1 if show_shade_legend else 0)  # +1: shade legend's own title row
        )
        legend_extra_in = 0.32 * legend_rows

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8 + legend_extra_in), squeeze=False)
        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                ax = axes[ri][ci]
                ax.set_xlim(-0.015, 0.515)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks([0.0, 0.125, 0.25, 0.375, 0.5])
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                ax.set_axisbelow(True)
                ax.plot([0, 0.5], [0, 0.5], color="#cfcfc9", linewidth=1.0, linestyle="--", zorder=1)
                if ri == n_rows - 1:
                    ax.set_xlabel("FPR", fontsize=8)
                if ri == 0 and cv not in (None, "(none)"):
                    ax.set_title(str(cv), fontsize=9.5, pad=8)
                if ci == 0 and rv not in (None, "(none)"):
                    ax.set_ylabel(str(rv), fontsize=8.5)
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
                        points.append((pv, d["fpr"].mean(), d["tpr"].mean()))
                    if not points:
                        continue
                    xs = [p[1] for p in points]
                    ys = [p[2] for p in points]
                    if len(xs) > 1:
                        ax.plot(xs, ys, color=colors[lv], alpha=0.45, linewidth=1.0, zorder=2)
                    shades = [_shade_color(colors[lv], i / max(len(points) - 1, 1)) for i in range(len(points))]
                    ax.scatter(xs, ys, color=shades, s=34, edgecolor="white", linewidth=0.7, zorder=3)
        fig.suptitle(title, fontsize=12, y=0.995)

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
            line_auc_values = self._metric_by_axis_value(configs, 0, "roc_auc")
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                           markerfacecolor=colors[lv], markeredgecolor="white",
                           label=f"{lv} (ROC-AUC={line_auc_values[lv][0]:.4f}±{line_auc_values[lv][1]:.4f})")
                for lv in line_vals
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
            shade_legend = fig.legend(
                handles=shade_handles, loc="upper right", bbox_to_anchor=(0.995, cursor_y), fontsize=7,
                framealpha=0.85, title="shade = agg[1] (light→dark)", title_fontsize=7,
            )
            fig.canvas.draw()
            cursor_y = (shade_legend.get_window_extent(renderer).y0 - 4) / fig_h_px

        fig.tight_layout(rect=[0, 0.04, 1, cursor_y])
        return fig

    def plot(self, axes, metrics, agg=(), exclude_keywords=(), plot_type="bar", grid_by=None,
             title="Cross-run comparison"):
        """Resolves `axes`/`agg`/`exclude_keywords` via resolve_configs_cartesian
        (see its docstring, including its "" placeholder and tuple-axis
        handling) and renders the result -- one call replaces the old
        separate resolve_configs_cartesian(...) + plot_stat(configs, ...)
        pair. The resolved configs are cached on self._last_configs so a
        follow-up print_configs() (no args) can reuse them without
        re-resolving. Renders one bar/violin per config for each metric in
        `metrics` (any
        runs.tsv numeric column -- "f1", "em_hd", "em_gt_hd", "mean_relerr_agg",
        "covar_relerr_agg", "transition_null_to_alt", "roc_auc", ... -- or the
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
        Only supported in this single-level-agg, plot_type="bar"/"violin"
        path -- not in _plot_grouped, heatmap, or grid_by="hd_bin".

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
        per window size, each holding one bar per x_bin. metrics=["tpr_fpr"]
        with a multi-level agg gets the same hierarchy but as a scatter/line
        plot instead of bars: agg[0] -> one colored line per value (shared
        legend, different segments), agg[1] -> points along that same
        segment (shaded light->dark), agg[2]/agg[3] -> subplot columns/rows
        (see _plot_tpr_fpr_grouped). A single-level (or no) agg keeps the old
        fixed fraction_bin x FIGURE_COLUMNS x_bin grid (_plot_tpr_fpr_grid).

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
        _plot_heatmap. When
        every x and y value parses as a positive number (e.g. agg=["-w",
        "-s"]), both share one log-scaled numeric scale (see
        _axis_edges/_plot_heatmap) -- cell width/height is proportional to
        the actual log-space gap between consecutive values, not a fixed
        size per category, so a window value and a step value that happen to
        be equal land at the same physical position on their respective axes
        (e.g. a (window, step) cell with step==window sits on the geometric
        diagonal). Otherwise (e.g. em_hd_bin/em_gt_hd_bin/merged_category
        values, which are category labels, not numbers) falls back to a
        plain evenly-spaced categorical grid, same as _plot_heatmap_hd_bin's
        hd_bin x fraction_bin grid. All of a figure's subplots share one
        color scale. A boolean agg axis in its off/negated state contributes
        no display value at all (see _combo_label_part), so mixing a bool
        toggle into agg[0]/agg[1] collapses that axis to one tick; stick to
        real-valued axes (like -w/-s, or a *_bin/merged_category axis) for
        x/y.

        Returns a list of Figures: the tpr_fpr grid first (if requested),
        then one figure per remaining metric (grid_by or heatmap set) or one
        shared row figure for all of them (grid_by=None, plot_type!="heatmap").
        Also sets self.metrics to the scalar metrics used (i.e. `metrics`
        minus "tpr_fpr") -- print_configs defaults to that instead of
        requiring metrics= on every call."""
        assert plot_type in ("bar", "violin", "heatmap")
        assert grid_by in (None, "hd_bin") or plot_type == "heatmap"
        configs = self.resolve_configs_cartesian(axes, exclude_keywords=exclude_keywords, agg=agg)
        self._last_configs = configs
        suffix = self._title_suffix(configs)
        figs = []
        if "tpr_fpr" in metrics:
            levels = max((len(self._config_axes.get(label, [])) for label, _ in configs), default=0)
            if levels >= 2:
                figs.append(self._plot_tpr_fpr_grouped(configs, title=f"{title} (TPR/FPR){suffix}"))
            else:
                figs.append(self._plot_tpr_fpr_grid(configs, title=f"{title} (TPR/FPR){suffix}"))
        metrics = [m for m in metrics if m != "tpr_fpr"]
        self.metrics = [m for panel in metrics for m in (panel if isinstance(panel, tuple) else (panel,))]
        if not metrics:
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
                figs.extend(self._plot_heatmap(configs, metrics, title_suffix=suffix))
            return figs

        levels = max((len(self._config_axes.get(label, [])) for label, _ in configs), default=0)
        if levels >= 2 and grid_by is None:
            figs.extend(self._plot_grouped(configs, metrics, plot_type, title_suffix=suffix))
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
            for metric in metrics:
                panel_col = _METRIC_PANEL_COLUMN.get(metric)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 3.0 * n_rows + 0.6), squeeze=False)
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
                            cell_bounds.append(_draw_stat(ax, cfg_idx, dd[metric], colors[label], 1.0, plot_type))
                        if plot_type == "violin" and not bounded:
                            _clip_axis_to_whiskers(ax, cell_bounds)
                fig.suptitle(f"Cross-run {metric} by HD bin{suffix}", fontsize=12, y=0.995)
                metric_values = self._metric_by_label(configs, metric)
                agg_legend_labels = [f"{label} ({metric}={metric_values[label]:.4f})" for label in legend_labels]
                fig.legend(legend_handles, agg_legend_labels, loc="upper right",
                           bbox_to_anchor=(0.995, 0.97), fontsize=8, framealpha=0.9)
                fig.tight_layout(rect=[0, 0.02, 1, 0.93])
                figs.append(fig)
            return figs

        n = len(configs)
        fig, axes = plt.subplots(1, len(metrics), figsize=(2.8 * len(metrics), 4.0), squeeze=False)
        for ax, panel in zip(axes[0], metrics):
            ax.set_xlim(-0.7, n - 0.3)
            ax.set_xticks(range(n))
            ax.set_xticklabels(legend_labels, rotation=30, ha="right", fontsize=7.5)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if isinstance(panel, tuple):
                ax.set_title(" / ".join(panel), fontsize=9.5)
                bounded = _apply_bounded_yaxis(ax, panel)
                panel_colors = {m: _config_color(i) for i, m in enumerate(panel)}
                sub_width = 0.82 / len(panel)
                cell_bounds = []
                for cfg_idx, (label, _) in enumerate(configs):
                    d = dfs[label]
                    if d.empty:
                        continue
                    for mi, m in enumerate(panel):
                        panel_col = _METRIC_PANEL_COLUMN.get(m)
                        dd = d[_is_true(d[panel_col])] if panel_col is not None else d
                        offset = (mi - (len(panel) - 1) / 2) * sub_width
                        cell_bounds.append(_draw_stat(ax, cfg_idx + offset, dd[m], panel_colors[m], sub_width, plot_type))
                if plot_type == "violin" and not bounded:
                    _clip_axis_to_whiskers(ax, cell_bounds)
                handles = [plt.Rectangle((0, 0), 1, 1, facecolor=panel_colors[m]) for m in panel]
                ax.legend(handles, panel, loc="best", fontsize=6.5, framealpha=0.85)
            else:
                metric = panel
                panel_col = _METRIC_PANEL_COLUMN.get(metric)
                ax.set_title(metric, fontsize=9.5)
                bounded = _apply_bounded_yaxis(ax, metric)
                cell_bounds = []
                for cfg_idx, (label, _) in enumerate(configs):
                    d = dfs[label]
                    if d.empty:
                        continue
                    if panel_col is not None:
                        d = d[_is_true(d[panel_col])]
                    cell_bounds.append(_draw_stat(ax, cfg_idx, d[metric], colors[label], 1.0, plot_type))
                if plot_type == "violin" and not bounded:
                    _clip_axis_to_whiskers(ax, cell_bounds)
        fig.suptitle(f"{title}{suffix}", fontsize=12, y=0.99)
        flat_metrics = [m for panel in metrics for m in (panel if isinstance(panel, tuple) else (panel,))]
        metric_values = {metric: self._metric_by_label(configs, metric) for metric in flat_metrics}
        agg_legend_labels = [
            f"{label} (" + ", ".join(f"{m}={metric_values[m][label]:.4f}" for m in flat_metrics) + ")"
            for label in legend_labels
        ]
        fig.legend(legend_handles, agg_legend_labels, loc="upper right",
                   bbox_to_anchor=(0.995, 0.90), fontsize=8, framealpha=0.9)
        fig.tight_layout(rect=[0, 0, 1, 0.8])
        figs.append(fig)
        return figs

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
        return f" — {restriction}" if restriction else ""

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
        norm = _strip_flag(axis_name) if axis_name in _FLAG_TO_DEST else axis_name
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
        IQR, 0.0 for a single window size) of the per-window-size means, to
        match _draw_stat's own bar errorbar convention -- this is std across
        window sizes' pooled means, not across raw rows or configs. Returns
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
                if d.empty:
                    continue
                if panel_col is not None:
                    d = d[_is_true(d[panel_col])]
                for window_size, sub in d.groupby("_window_size", dropna=False):
                    m = sub[metric].dropna().mean()
                    if not pd.isna(m):
                        by_window.setdefault(window_size, []).append(m)
            window_means = pd.Series(
                [pd.Series(ms, dtype=float).mean() for ms in by_window.values()], dtype=float,
            )
            if window_means.empty:
                values[target] = (float("nan"), float("nan"))
            else:
                values[target] = (window_means.mean(), window_means.std() if len(window_means) > 1 else 0.0)
        return values

    def _plot_grouped(self, configs, metrics, plot_type, title_suffix=""):
        """One figure per metric, laid out by `configs`' per-config axis
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
        actually being averaged -- see _metric_by_axis_value."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        hue_vals = _ordered_unique(axis_val(label, 0) for label, _ in configs)
        bar_vals = _ordered_unique(axis_val(label, 1) for label, _ in configs)
        col_vals = _ordered_unique(axis_val(label, 2) for label, _ in configs)
        row_vals = _ordered_unique(axis_val(label, 3) for label, _ in configs)

        hue_idx, bar_idx = 0, 1
        pooled_legend = hue_vals == ["(none)"] and len(bar_vals) > 1

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

        colors = {hv: _config_color(i) for i, hv in enumerate(hue_vals)}
        n_rows, n_cols = len(row_vals), len(col_vals)
        bar_width = 0.82 / max(len(hue_vals), 1)

        figs = []
        for metric in metrics:
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows), squeeze=False)
            for ri, rv in enumerate(row_vals):
                for ci, cv in enumerate(col_vals):
                    ax = axes[ri][ci]
                    cell_bar_vals = local_bar_vals[(rv, cv)]
                    ax.set_xlim(-0.7, len(cell_bar_vals) - 0.3)
                    ax.set_xticks(range(len(cell_bar_vals)))
                    ax.set_xticklabels(cell_bar_vals, rotation=30, ha="right", fontsize=7.5)
                    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                    ax.set_axisbelow(True)
                    bounded = _apply_bounded_yaxis(ax, metric)
                    if ri == 0 and cv not in (None, "(none)"):
                        ax.set_title(str(cv), fontsize=9)
                    if ci == 0 and rv not in (None, "(none)"):
                        ax.set_ylabel(str(rv), fontsize=9)
                    cell_bounds = []
                    for bi, bv in enumerate(cell_bar_vals):
                        for hi, hv in enumerate(hue_vals):
                            entry = by_key.get((rv, cv, bv, hv))
                            if entry is None:
                                continue
                            label, _ = entry
                            d = dfs[label]
                            if d.empty:
                                continue
                            if panel_col is not None:
                                d = d[_is_true(d[panel_col])]
                            offset = (hi - (len(hue_vals) - 1) / 2) * bar_width
                            cell_bounds.append(_draw_stat(ax, bi + offset, d[metric], colors[hv], bar_width, plot_type))
                    if plot_type == "violin" and not bounded:
                        _clip_axis_to_whiskers(ax, cell_bounds)
            fig.suptitle(f"Cross-run {metric} comparison{title_suffix}", fontsize=12, y=0.99)
            if pooled_legend or len(hue_vals) > 1 or hue_vals not in ([None], ["(none)"]):
                hue_metric_values = self._metric_by_axis_value(configs, hue_idx, metric, panel_col)
                handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[hv]) for hv in hue_vals]
                if pooled_legend:
                    hue_legend_labels = [
                        f"all ({metric}={hue_metric_values[hv][0]:.4f}±{hue_metric_values[hv][1]:.4f})"
                        for hv in hue_vals
                    ]
                else:
                    hue_legend_labels = [
                        f"{hv} ({metric}={hue_metric_values[hv][0]:.4f}±{hue_metric_values[hv][1]:.4f})"
                        for hv in hue_vals
                    ]
                fig.legend(handles, hue_legend_labels, loc="upper right",
                           bbox_to_anchor=(0.995, 0.90), fontsize=8, framealpha=0.9)
            fig.tight_layout(rect=[0, 0, 1, 0.85])
            figs.append(fig)
        return figs

    def _plot_heatmap(self, configs, metrics, title_suffix=""):
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

        x/y share one log-scaled numeric scale (see _axis_edges) -- cell
        width/height proportional to the actual log-space gap between
        consecutive values, not a fixed per-category size, e.g. a (window,
        step) cell with step==window sits on the geometric diagonal -- ONLY
        when every x and y value parses as a positive float (the -w/-s use
        case). Otherwise (e.g. em_hd_bin/merged_category values, which are
        category labels, not numbers) falls back to a plain evenly-spaced
        categorical grid, one cell per (x, y) pair, same as
        _plot_heatmap_hd_bin's hd_bin x fraction_bin grid.

        A metric name ending in "_signed" (e.g. "f1_signed" for the real
        column "f1" -- see _signed_base_metric) renders that real metric as
        usual, then subtracts its "null" column's grid from every OTHER
        column's grid, cell by cell, within the same row -- i.e. agg[2]
        (the 3rd agg axis, which this method puts on subplot columns) gets
        treated as "baseline vs. variants": the null/baseline column is
        whichever col_vals entry is a flag's off/negated ("(none)") or
        missing ("no-<flag>") display state if one is present, else
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

        numeric = True
        try:
            x_nums = [float(v) for v in x_vals]
            y_nums = [float(v) for v in y_vals]
            if min(x_nums, default=1.0) <= 0 or min(y_nums, default=1.0) <= 0:
                numeric = False
        except (TypeError, ValueError):
            numeric = False
        if numeric:
            x_vals = [v for _, v in sorted(zip(x_nums, x_vals))]
            y_vals = [v for _, v in sorted(zip(y_nums, y_vals))]
            x_nums = sorted(x_nums)
            y_nums = sorted(y_nums)
            x_edges = _axis_edges(x_nums, log=True)
            y_edges = _axis_edges(y_nums, log=True)
            tick_x, tick_y = x_nums, y_nums
        else:
            x_vals = sorted(x_vals, key=_token_sort_key)
            y_vals = sorted(y_vals, key=_token_sort_key)
            tick_x = [i + 0.5 for i in range(len(x_vals))]
            tick_y = [i + 0.5 for i in range(len(y_vals))]

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
                            grid[yi][xi] = d[real_metric].dropna().mean() if not d.empty else float("nan")
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
                    if numeric:
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
                    else:
                        mesh = ax.pcolormesh(grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
                    if is_diff.get((rv, cv)):
                        diff_mesh = mesh
                    else:
                        raw_mesh = mesh
                    ax.set_xticks(tick_x)
                    ax.set_xticklabels(x_vals, rotation=30, ha="right", fontsize=7.5)
                    ax.set_yticks(tick_y)
                    # Only the leftmost column in each row needs y-tick text --
                    # every subplot got its own labels before, which rendered
                    # on top of the previous (left-neighbor) subplot's data
                    # cells once panels were packed close together.
                    if ci == 0:
                        ax.set_yticklabels(y_vals, fontsize=7.5)
                    else:
                        ax.set_yticklabels([])
                    for yi, yn in enumerate(tick_y):
                        for xi, xn in enumerate(tick_x):
                            v = grid[yi][xi]
                            if not pd.isna(v):
                                color = "white" if (v - vmin) / (vmax - vmin or 1) > 0.6 else "black"
                                ax.text(xn, yn, f"{v:.3f}", ha="center", va="center", color=color, fontsize=7)
                    if ri == 0 and cv not in (None, "(none)"):
                        ax.set_title(str(cv), fontsize=9.5)
                    if ci == 0 and rv not in (None, "(none)"):
                        ax.set_ylabel(str(rv), fontsize=9, labelpad=28)
            title_metric = "/".join(metrics) if metric_role else outer_metric
            base_label = "value" if metric_role else outer_metric
            fig.suptitle(f"Cross-run {title_metric} heatmap{title_suffix}", fontsize=12)
            all_axes = list(axes.flat)
            if diff_mesh is None:
                # Ordinary (unsigned) heatmap, or a signed one whose baseline
                # column is the only thing being shown -- one colorbar, right
                # side, exactly as before.
                if raw_mesh is not None:
                    fig.colorbar(raw_mesh, ax=all_axes, label=base_label, fraction=0.05, pad=0.02, location="right")
                fig.subplots_adjust(left=0.12, bottom=0.2, right=0.88, top=0.88, wspace=0.3, hspace=0.35)
            else:
                # Split, rather than stack, when both are present: the diff
                # colorbar keeps the usual right side (unaffected below), the
                # raw/baseline one goes on the left -- stacking both on the same
                # side left no room for the row-label text and y-tick bin
                # labels between them and the subplot grid.
                fig.colorbar(diff_mesh, ax=all_axes, label=f"Δ {base_label}", fraction=0.05, pad=0.02, location="right")
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
                fig.subplots_adjust(left=0.32, bottom=0.2, right=0.85, top=0.88, wspace=0.3, hspace=0.35)
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
                ax.set_xticklabels(hd_bins, rotation=45, ha="right", fontsize=6.5)
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
            fig.suptitle(f"Cross-run {metric} by HD bin x alt proportion{title_suffix}", fontsize=11, y=1.0)
            fig.text(0.5, 0.01, "HD bin (em_gt_hd)", ha="center", fontsize=8.5)
            fig.text(0.005, 0.5, "alt proportion", va="center", rotation="vertical", fontsize=8.5)
            fig.colorbar(mesh, ax=axes[0].tolist(), label=metric, fraction=0.05, pad=0.02)
            fig.subplots_adjust(left=0.09, bottom=0.28, right=0.9, top=0.85, wspace=0.15)
            figs.append(fig)
        return figs


# ---- module-private helpers: pure functions, no instance state needed ----

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
    simply absent; a _Not(value) requires the recorded value NOT equal that value --
    see _cartesian_axis_states, which builds these automatically. A missing dest
    (recorded.get(dest) is None) also satisfies an explicit requirement=False (and,
    for _Not, counts as the negated value when negated is False): a store_true flag
    that was never recorded (e.g. an older args.json written before that flag existed
    in CASTER_ARG_SPECS/PHLAG_ARG_SPECS -- see --site, added in b0c28fa) defaulted off
    exactly like an explicit False, so treating them as distinct would wrongly exclude
    those older runs from a pool/agg axis pinned to that flag's off state. recorded:
    one leaf's args.json, dest-keyed."""
    for flag, requirement in flags.items():
        dest = _FLAG_TO_DEST.get(flag)
        if dest is None:
            raise KeyError(f"{flag!r} is not one of benchmark.py's mirrored caster/phlag flags")
        value = recorded.get(dest)
        if isinstance(requirement, _Not):
            negated = requirement.value
            if negated is None:
                is_negated = value is None
            else:
                typed_negated = _typed_value(flag, negated)
                is_negated = value == typed_negated or (value is None and typed_negated is False)
            if is_negated:
                return False
        elif requirement is None:
            if value is not None:
                return False
        else:
            typed_requirement = _typed_value(flag, requirement)
            if value != typed_requirement and not (value is None and typed_requirement is False):
                return False
    return True


def _strip_flag(flag):
    return flag.lstrip("-")


def _pool_axes_label(pool_axes):
    """Display text for resolve_configs_cartesian's pooled (non-agg) axes --
    e.g. {"-w": ["50k"], "-s": ["40k"]} -> "w 50k s 40k" -- used as
    plot's title suffix so a figure states what its configs were
    restricted to instead of how many there are."""
    parts = []
    for flag, spec in pool_axes.items():
        name = _strip_flag(flag) if flag in _FLAG_TO_DEST else flag
        val_str = ",".join(str(v) for v in spec) if isinstance(spec, list) else str(spec)
        parts.append(f"{name} {val_str}")
    return " ".join(parts)


def _combo_label_part(flag, requirement):
    """Display text for one axis's state in a suffix combo label: an on-state renders as
    its bare value (value flags, e.g. "repulsion" for {"--ap": "repulsion"}) or bare flag
    name stripped of leading dashes (bool flags, e.g. "annealing"); requirement=None (the
    flag is missing from args.json) renders as "no-<flag>"; an off/negated state
    is dropped entirely (returns None) rather than shown with a "!" prefix. -w/-s
    specifically render as "w"/"s" glued directly onto their value, shortened to a
    k/m suffix like the rest of the codebase (e.g. {"-w": "50000"} -> "w50k"). A
    fractional -s value (e.g. 0.8, meaning 80% of whatever window it ends up
    paired with -- see _resolve_step_fraction) renders as its bare fraction
    ("s0.8") instead, since it has no single absolute step size to format
    until matched against a specific window."""
    if requirement is True:
        return _strip_flag(flag)
    if requirement is False or isinstance(requirement, _Not):
        return None
    if requirement is None:
        return f"no-{_strip_flag(flag)}"
    if flag == "-s" and (frac := _as_step_fraction(requirement)) is not None:
        return f"s{frac:g}"
    if flag in ("-w", "-s"):
        return f"{_strip_flag(flag)}{_format_param_value(_typed_value(flag, requirement))}"
    return str(requirement)


def _combo_label(flags, empty_label):
    parts = [part for flag, req in flags.items() if (part := _combo_label_part(flag, req)) is not None]
    return " ".join(parts) if parts else empty_label


def _cartesian_axis_states(flag, value_spec):
    """A scalar value_spec is a binary on/off axis (on=value_spec, off=its
    negation) -- e.g. {"--annealing": True}. A list value_spec is an explicit
    multi-value axis, used as-is with no automatic off/negation state -- e.g.
    {"-w": ["50k", "100k", "250k"]}. None is a valid list element (or scalar):
    it means the flag is missing from args.json entirely -- e.g. {"--lam": [None,
    1.0, 1.5]} includes runs with no recorded lambda alongside 1.0 and 1.5."""
    if isinstance(value_spec, list):
        return [{flag: v} for v in value_spec]
    return [{flag: value_spec}, {flag: _Not(value_spec)}]


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
    fraction, -w is missing, or -w is itself negated/unresolved)."""
    if "-s" not in flags or "-w" not in flags:
        return flags
    frac = _as_step_fraction(flags["-s"])
    if frac is None:
        return flags
    w_val = flags["-w"]
    if isinstance(w_val, _Not):
        return flags
    w_int = _typed_value("-w", w_val)
    flags = dict(flags)
    flags["-s"] = max(1, round(frac * w_int))
    return flags


def _draw_stat(ax, pos, vals, color, width, plot_type):
    """Draws one config's distribution at x=pos: plot_type="bar" is mean+std
    errorbar (matching production's errorbar="sd"); "violin" is the
    distribution as a standard box-and-whisker would show it -- the body's
    KDE is built only from values inside the Tukey whisker
    [Q1-1.5*IQR, Q3+1.5*IQR] (matplotlib's own showextrema draws the data's
    raw min/max instead, which is what made the tails so long with any real
    outliers), with the whisker itself drawn as a line and anything beyond it
    scattered as faint outlier dots so it isn't silently dropped. Falls back
    to a single dot when there's only one value (violinplot's KDE needs at
    least 2). Returns the (lo, hi) "inlier" range actually worth showing to
    scale (excludes outlier dots) -- None if nothing was drawn -- so the
    caller can clip the axis to it and mark the break (see
    _clip_axis_to_whiskers)."""
    vals = pd.Series(vals).dropna()
    if vals.empty:
        return None
    if plot_type == "bar":
        mean = vals.mean()
        err = vals.std() if len(vals) > 1 else 0.0
        ax.bar(pos, mean, width=width * 0.92, color=color, zorder=2)
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
            ax.plot([pos, pos], [inliers.min(), inliers.max()], color="black", linewidth=1.0, zorder=2.5)
            bounds = (inliers.min(), inliers.max())
        elif not inliers.empty:
            ax.scatter([pos], [inliers.iloc[0]], color=color, s=18, zorder=3)
            bounds = (inliers.iloc[0], inliers.iloc[0])
        if not outliers.empty:
            ax.scatter([pos] * len(outliers), outliers, color=color, s=6, alpha=0.4,
                       edgecolor="none", zorder=2)
        return bounds
    else:
        v = vals.iloc[0]
        ax.scatter([pos], [v], color=color, s=18, zorder=3)
        return (v, v)


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
        ax.set_ylim(min(0.0, lo), visible_top)
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


def _token_sort_key(token):
    """Sorts a heatmap axis's raw label tokens numerically when possible
    instead of lexicographically -- "80000" must sort after "40000", not
    before it. token is usually a string, but can be the Python None
    self._config_axes falls back to for an axis slot that doesn't exist for
    a given config (e.g. y when a heatmap's agg only has 1 real level) --
    float(None) raises TypeError, not ValueError, so that's caught too
    rather than propagating out of what looks like a harmless sort call;
    None sorts last (after both numeric and non-numeric strings).

    A whole-token float() parse fails for a bin-range label like
    "[0.000,0.067]"/"(0.067,0.133]" (_compute_hd_bin_edges), so this falls
    back to that label's own leading number (its lower bound) rather than
    the raw string -- plain string comparison would otherwise sort the
    first bin's "[..." after every other bin's "(..." (since '(' < '['
    lexicographically), landing the lowest bin last instead of first. Only
    a token with no number at all (e.g. a plain category name) falls back
    to the raw string."""
    if token is None:
        return (2, "")
    try:
        return (0, float(token))
    except (TypeError, ValueError):
        pass
    if isinstance(token, str) and (m := _LEADING_NUM_RE.search(token)):
        return (0, float(m.group()))
    return (1, token)


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
        if not p.exists():
            continue
        frame = pd.read_csv(p, sep="\t")
        frame["_window_size"] = get_args(d).get("window_size")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty and "category" in df.columns and "subcategory" in df.columns:
        df["merged_category"] = _merged_category(df["category"], df["subcategory"])
    if not df.empty and bin_specs:
        for bin_col, (source_col, edges) in bin_specs.items():
            if source_col in df.columns:
                df[bin_col] = df[source_col].apply(lambda v: assign_bin(v, edges))
    return df


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
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig
