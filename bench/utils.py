import colorsys
import itertools
import math
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
# _discover_row_values instead of a static constant.
_DYNAMIC_ROW_AXES = {"category", "subcategory", "merged_category"}


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
_FLAG_TO_DEST = {flag: dest for dest, flag, _ in _MIRRORED_SPECS}
_PARSER = _build_parser()
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
    each run's own args.json, then all rendered via plot_stat: pass "tpr_fpr" in
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
        tree from scratch -- args.json presence is all we actually check."""
        if self._leaf_dirs is None:
            self._leaf_dirs = sorted(
                p.parent for p in Path(self.root).rglob("args.json")
                if (p.parent / "runs.tsv").exists()
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

    def _discover_row_values(self, col, exclude_keywords=()):
        """Every distinct value `col` takes across runs.tsv rows on disk --
        used by resolve_configs_cartesian to auto-populate a _DYNAMIC_ROW_AXES
        agg axis (e.g. "category") that wasn't given an explicit list in
        `axes`. merged_category isn't itself a runs.tsv column, so it's
        derived here the same way _load_configs_runs derives it."""
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
        return sorted(values, key=_token_sort_key)

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
            else:
                print(f"[skip] no matching run dirs for config {label!r}")
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
        that appears in the label/legend. Every OTHER axis in `axes` (not in
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
        get loaded, e.g. by plot_stat/print_configs -- see
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
        fraction_bin x column x x_bin breakdown grids (plot_stat's "tpr_fpr"
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
        "step = 80% of window" and expands to (50000, 40000)/(100000, 80000),
        not two configs that both try to match step_size==0.8. Requires "-w"
        as a list in `axes` (explicit, or auto-discovered per above -- e.g.
        agg=["-w", "-s"] alone is enough); a literal "-s" value (>1, or a
        string like "1k") is left alone and still pairs with every window
        size independently, same as ever.

        Each resulting config also records its own per-agg-axis value, in
        `agg`'s order, on self._config_axes[label] -- plot_stat uses this to
        arrange bars/subplots hierarchically instead of one flat row: agg[0]
        -> x-position (bar groups), agg[1] -> bars within a group (hue/color,
        with a shared legend), agg[2] -> subplot columns, agg[3] -> subplot
        rows. agg=["-w", "x_bin"] (2 levels) therefore draws one bar-group
        per window size, each holding one bar per x_bin; adding a 3rd axis
        makes one column of subplots per its value, a 4th makes one row per
        its value. -s folded into -w (see above) still counts as -w's single
        level, not two."""
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
        axes = _expand_step_fractions(axes)
        if "-s" in agg_set:
            agg_set.add("-w")
        pool_axes = {flag: spec for flag, spec in axes.items() if flag not in agg_set}
        pool_axis_states = [_cartesian_axis_states(flag, spec) for flag, spec in pool_axes.items()]
        pool_combos = list(itertools.product(*pool_axis_states)) if pool_axis_states else [()]

        # One "level" per (still-existing) name in agg_seq -- a level's states are
        # (contribution_dict, is_flag, display_value) triples; -w's contribution can
        # carry -s along with it (see _expand_step_fractions), whose own display
        # value gets folded into -w's -- consumed_flags below skips re-processing it
        # as its own separate level.
        levels = []
        consumed_flags = set()
        for name in agg_seq:
            if name in consumed_flags:
                continue
            if name in data_col_axes:
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
                dirs.extend(self.find_matching_leaf_dirs(merged, exclude_keywords))
            if not dirs:
                print(f"[skip] no matching run dirs for config {label!r}")
                continue
            configs.append((label, dirs))
            self._row_filters[label] = row_filter
            self._config_axes[label] = axis_values
            self._agg_axis_names[label] = level_names
        return configs

    def print_configs(self, configs, metrics=None, paths=False):
        """Prints each config's label, run count, and mean of every metric in
        `metrics` (pooled across all its runs -- same aggregate plot_stat
        subplots show), followed by the args.json params that actually vary
        across its matching dirs (see _param_summary_lines) -- e.g. a config
        pooled via `agg` mixes lambda values, so its dirs disagree on lambda.
        metrics defaults to self.metrics -- the scalar metrics from the most
        recent plot_stat call -- so you don't have to repeat the same list;
        pass metrics= explicitly to override. paths=True also prints each
        config's sorted run dirs."""
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
        _plot_grouped) version of _plot_tpr_fpr_grid: agg[0] -> one point per
        value on each line (shaded light->dark, replacing the old fixed
        x_bin-driven points), agg[1] -> one line per value (colored, shared
        legend), agg[2] -> subplot columns, agg[3] -> subplot rows. Each
        point is the (mean FPR, mean TPR) of that (row, col, point, line)
        combo's already-row-filtered rows (in_panel_b only). The legend's
        ROC-AUC per line value is pooled across every (point, col, row) combo
        sharing that line value (agg[1]), not just one cell -- see
        _metric_by_axis_value."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        point_vals = _ordered_unique(axis_val(label, 0) for label, _ in configs)
        line_vals = _ordered_unique(axis_val(label, 1) for label, _ in configs)
        col_vals = _ordered_unique(axis_val(label, 2) for label, _ in configs)
        row_vals = _ordered_unique(axis_val(label, 3) for label, _ in configs)

        by_key = {}
        for label, dirs in configs:
            key = (axis_val(label, 3), axis_val(label, 2), axis_val(label, 0), axis_val(label, 1))
            by_key[key] = (label, dirs)

        colors = {lv: _config_color(i) for i, lv in enumerate(line_vals)}
        dfs = _load_configs_runs(configs, self._row_filters)
        n_rows, n_cols = len(row_vals), len(col_vals)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8), squeeze=False)
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
                if ri == 0 and cv is not None:
                    ax.set_title(str(cv), fontsize=9.5, pad=8)
                if ci == 0 and rv is not None:
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

        if len(line_vals) > 1 or line_vals != [None]:
            line_auc_values = self._metric_by_axis_value(configs, dfs, 1, "roc_auc")
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                           markerfacecolor=colors[lv], markeredgecolor="white",
                           label=f"{lv} (ROC-AUC={line_auc_values[lv]:.4f})")
                for lv in line_vals
            ]
            fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.97), fontsize=8, framealpha=0.9)

        # Shared shading legend showing agg[0]'s values (point axis), light->dark,
        # matching the shading drawn on every line's points -- same role as
        # _plot_tpr_fpr_grid's fixed x_bin shading legend, generalized to whatever
        # axis agg[0] actually is.
        if len(point_vals) > 1:
            shade_handles = [
                plt.Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                           markerfacecolor=_shade_color("#808080", i / max(len(point_vals) - 1, 1)),
                           markeredgecolor="white", markeredgewidth=0.6, label=str(pv))
                for i, pv in enumerate(point_vals)
            ]
            fig.legend(
                handles=shade_handles, loc="upper left", bbox_to_anchor=(0.005, 0.97), fontsize=7,
                framealpha=0.85, title="shade = agg[0] (light→dark)", title_fontsize=7,
            )
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        return fig

    def plot_stat(self, configs, metrics, plot_type="bar", grid_by=None, title="Cross-run comparison"):
        """Renders one bar/violin per config for each metric in `metrics` (any
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
        repeating config names as x-ticks on every subplot.

        grid_by=None, `configs` built with a multi-level agg (2-4 axes):
        each metric instead gets its own figure, laid out hierarchically by
        agg's axes instead of one flat bar per config -- agg[0] positions
        bar-groups along x, agg[1] draws one bar per group (colored, with a
        shared hue legend), a 3rd agg axis splits into subplot columns, a
        4th into subplot rows. E.g. agg=["-w", "x_bin"] draws one bar-group
        per window size, each holding one bar per x_bin. metrics=["tpr_fpr"]
        with a multi-level agg gets the same hierarchy but as a scatter/line
        plot instead of bars: agg[0] -> points along one line (shaded
        light->dark, with a legend), agg[1] -> one colored line per value
        (shared legend), agg[2]/agg[3] -> subplot columns/rows (see
        _plot_tpr_fpr_grouped). A single-level (or no) agg keeps the old
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

        plot_type="heatmap" without grid_by="hd_bin": one figure per metric, cells colored by that
        metric's mean value on a 2D grid built from each config's *label*,
        splitting it on the first space into (x, y) -- i.e. the first two
        agg axes from the resolve_configs_cartesian call that built `configs`
        (matching agg's given order, e.g. agg=["-w", "-s"] labels look like
        "50000 1000" -> x="50000", y="1000"), both parsed as numbers and
        required to be > 0. Both axes share one log-scaled numeric scale (see
        _axis_edges/_plot_heatmap) -- cell width/height is proportional to
        the actual log-space gap between consecutive values, not a fixed
        size per category, so a window value and a step value that happen to
        be equal land at the same physical position on their respective axes
        (e.g. a (window, step) cell with step==window sits on the geometric
        diagonal). Only meaningful when every config's
        label actually has 2 numeric tokens -- a boolean agg axis in its
        off/negated state contributes no token at all (see
        _combo_label_part), so mixing a bool toggle into a heatmap's agg
        list will misalign columns; stick to value axes (like -w/-s) for
        heatmap configs.

        Returns a list of Figures: the tpr_fpr grid first (if requested),
        then one figure per remaining metric (grid_by or heatmap set) or one
        shared row figure for all of them (grid_by=None, plot_type!="heatmap").
        Also sets self.metrics to the scalar metrics used (i.e. `metrics`
        minus "tpr_fpr") -- print_configs defaults to that instead of
        requiring metrics= on every call."""
        assert plot_type in ("bar", "violin", "heatmap")
        assert grid_by in (None, "hd_bin") or plot_type == "heatmap"
        suffix = self._title_suffix(configs)
        figs = []
        if "tpr_fpr" in metrics:
            levels = max((len(self._config_axes.get(label, [])) for label, _ in configs), default=0)
            if levels >= 2:
                figs.append(self._plot_tpr_fpr_grouped(configs, title=f"{title} (TPR/FPR){suffix}"))
            else:
                figs.append(self._plot_tpr_fpr_grid(configs, title=f"{title} (TPR/FPR){suffix}"))
        metrics = [m for m in metrics if m != "tpr_fpr"]
        self.metrics = metrics
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
        dfs = _load_configs_runs(configs, self._row_filters)
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
                        if plot_type == "violin":
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
        for ax, metric in zip(axes[0], metrics):
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            ax.set_title(metric, fontsize=9.5)
            ax.set_xlim(-0.7, n - 0.3)
            ax.set_xticks(range(n))
            ax.set_xticklabels(legend_labels, rotation=30, ha="right", fontsize=7.5)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            cell_bounds = []
            for cfg_idx, (label, _) in enumerate(configs):
                d = dfs[label]
                if d.empty:
                    continue
                if panel_col is not None:
                    d = d[_is_true(d[panel_col])]
                cell_bounds.append(_draw_stat(ax, cfg_idx, d[metric], colors[label], 1.0, plot_type))
            if plot_type == "violin":
                _clip_axis_to_whiskers(ax, cell_bounds)
        fig.suptitle(f"{title}{suffix}", fontsize=12, y=0.99)
        metric_values = {metric: self._metric_by_label(configs, metric) for metric in metrics}
        agg_legend_labels = [
            f"{label} (" + ", ".join(f"{m}={metric_values[m][label]:.4f}" for m in metrics) + ")"
            for label in legend_labels
        ]
        fig.legend(legend_handles, agg_legend_labels, loc="upper right",
                   bbox_to_anchor=(0.995, 0.90), fontsize=8, framealpha=0.9)
        fig.tight_layout(rect=[0, 0, 1, 0.8])
        figs.append(fig)
        return figs

    def _title_suffix(self, configs):
        """" -- by <agg axis names> (<config labels>)" appended to every
        plot_stat figure title, so a figure is self-descriptive (what it was
        aggregated by, and which configs it holds) without needing a
        separate print_configs call alongside it. Agg axis names come from
        self._agg_axis_names[label] (set by resolve_configs_cartesian,
        same for every label produced by one call -- read off the first
        config); empty for configs built via resolve_configs (no agg
        concept there)."""
        labels = [label for label, _ in configs]
        if not labels:
            return ""
        agg_names = self._agg_axis_names.get(labels[0], [])
        bits = []
        if agg_names:
            bits.append("by " + ", ".join(agg_names))
        bits.append(", ".join(labels) if len(labels) <= 6 else f"{len(labels)} configs")
        return " — " + " — ".join(bits)

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
        used by plot_stat's heatmap grid_by generalization to render one
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
        dfs = _load_configs_runs(configs, self._row_filters)
        return {
            label: dfs[label][metric].dropna().mean() if not dfs[label].empty else float("nan")
            for label, _ in configs
        }

    def _metric_by_axis_value(self, configs, dfs, axis_idx, metric, panel_col=None):
        """Mean `metric` pooled across every config sharing the same
        self._config_axes value at `axis_idx` (e.g. axis_idx=1 pools by the
        hue/line axis a grouped plot's legend colors by -- resolve_configs_
        cartesian's agg[1], see its docstring) -- used to attach an
        aggregate score to each legend entry in _plot_grouped/
        _plot_tpr_fpr_grouped instead of leaving the legend as bare axis
        values with no number attached. `dfs`: configs' already-loaded
        runs.tsv frames (label-keyed, from _load_configs_runs) -- passed in
        rather than reloaded, since callers already have it."""
        def axis_val(label):
            vals = self._config_axes.get(label, [])
            return vals[axis_idx] if axis_idx < len(vals) else None

        values = {}
        for target in _ordered_unique(axis_val(label) for label, _ in configs):
            frames = [dfs[label] for label, _ in configs if axis_val(label) == target and not dfs[label].empty]
            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if panel_col is not None and not combined.empty:
                combined = combined[_is_true(combined[panel_col])]
            values[target] = combined[metric].dropna().mean() if not combined.empty else float("nan")
        return values

    def _plot_grouped(self, configs, metrics, plot_type, title_suffix=""):
        """One figure per metric, laid out by `configs`' per-config axis
        values (self._config_axes, set by resolve_configs_cartesian's agg --
        see its docstring): axis 0 -> x-position (bar groups), axis 1 -> one
        bar per group (hue, shared legend), axis 2 -> subplot columns, axis 3
        -> subplot rows. A config with fewer than 4 levels just gets a
        constant (None) value for the missing ones, collapsing that
        dimension to a single group/column/row. The legend's score per hue
        value is pooled across every (bar, col, row) combo sharing that hue
        value (agg[1]), not just one cell -- see _metric_by_axis_value."""
        def axis_val(label, idx):
            vals = self._config_axes.get(label, [])
            return vals[idx] if idx < len(vals) else None

        bar_vals = _ordered_unique(axis_val(label, 0) for label, _ in configs)
        hue_vals = _ordered_unique(axis_val(label, 1) for label, _ in configs)
        col_vals = _ordered_unique(axis_val(label, 2) for label, _ in configs)
        row_vals = _ordered_unique(axis_val(label, 3) for label, _ in configs)

        by_key = {}
        for label, dirs in configs:
            key = (axis_val(label, 3), axis_val(label, 2), axis_val(label, 0), axis_val(label, 1))
            by_key[key] = (label, dirs)

        colors = {hv: _config_color(i) for i, hv in enumerate(hue_vals)}
        dfs = _load_configs_runs(configs, self._row_filters)
        n_rows, n_cols = len(row_vals), len(col_vals)
        bar_width = 0.82 / max(len(hue_vals), 1)

        figs = []
        for metric in metrics:
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows), squeeze=False)
            for ri, rv in enumerate(row_vals):
                for ci, cv in enumerate(col_vals):
                    ax = axes[ri][ci]
                    ax.set_xlim(-0.7, len(bar_vals) - 0.3)
                    ax.set_xticks(range(len(bar_vals)))
                    ax.set_xticklabels(bar_vals, rotation=30, ha="right", fontsize=7.5)
                    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                    ax.set_axisbelow(True)
                    if ri == 0 and cv is not None:
                        ax.set_title(str(cv), fontsize=9)
                    if ci == 0 and rv is not None:
                        ax.set_ylabel(str(rv), fontsize=9)
                    cell_bounds = []
                    for bi, bv in enumerate(bar_vals):
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
                    if plot_type == "violin":
                        _clip_axis_to_whiskers(ax, cell_bounds)
            fig.suptitle(f"Cross-run {metric} comparison{title_suffix}", fontsize=12, y=0.99)
            if len(hue_vals) > 1 or hue_vals != [None]:
                hue_metric_values = self._metric_by_axis_value(configs, dfs, 1, metric, panel_col)
                handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[hv]) for hv in hue_vals]
                hue_legend_labels = [f"{hv} ({metric}={hue_metric_values[hv]:.4f})" for hv in hue_vals]
                fig.legend(handles, hue_legend_labels, loc="upper right",
                           bbox_to_anchor=(0.995, 0.90), fontsize=8, framealpha=0.9)
            fig.tight_layout(rect=[0, 0, 1, 0.85])
            figs.append(fig)
        return figs

    def _plot_heatmap(self, configs, metrics, title_suffix=""):
        """One shared log-scaled numeric scale for both axes -- cell
        width/height is proportional to the actual (log-space) gap between
        consecutive x/y values (via pcolormesh's irregular cell edges, see
        _axis_edges), not a fixed per-category size. A window value and a
        step value that happen to be equal land at the same physical
        position on their respective axes, so e.g. the (window, step) cell
        where step==window sits on the geometric diagonal instead of
        wherever its categorical index happened to fall. Log rather than
        linear because window/step sizes here span a ~500x range, which on a
        linear scale squashes every small value into a sliver against the
        axis; requires every x/y value to be > 0 (true for window/step
        sizes, the intended use)."""
        dfs = _load_configs_runs(configs, self._row_filters)
        x_of, y_of = {}, {}
        for label, _ in configs:
            tokens = label.split(" ", 1)
            x_of[label] = tokens[0]
            y_of[label] = tokens[1] if len(tokens) > 1 else ""
        x_vals = sorted({x_of[label] for label, _ in configs}, key=_token_sort_key)
        y_vals = sorted({y_of[label] for label, _ in configs}, key=_token_sort_key)
        x_nums = [float(v) for v in x_vals]
        y_nums = [float(v) for v in y_vals]
        x_edges = _axis_edges(x_nums, log=True)
        y_edges = _axis_edges(y_nums, log=True)

        figs = []
        for metric in metrics:
            panel_col = _METRIC_PANEL_COLUMN.get(metric)
            grid = [[float("nan")] * len(x_vals) for _ in y_vals]
            for label, _ in configs:
                d = dfs[label]
                if panel_col is not None and not d.empty:
                    d = d[_is_true(d[panel_col])]
                val = d[metric].dropna().mean() if not d.empty else float("nan")
                grid[y_vals.index(y_of[label])][x_vals.index(x_of[label])] = val

            fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(x_vals) + 3.0),
                                             max(5.0, 0.45 * len(y_vals) + 3.0)))
            ax.set_xscale("log")
            ax.set_yscale("log")
            mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap="viridis", shading="flat")
            ax.set_xticks(x_nums)
            ax.set_xticklabels(x_vals, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(y_nums)
            ax.set_yticklabels(y_vals, fontsize=8)
            for yi, yn in enumerate(y_nums):
                for xi, xn in enumerate(x_nums):
                    v = grid[yi][xi]
                    if not pd.isna(v):
                        ax.text(xn, yn, f"{v:.3f}", ha="center", va="center", color="white", fontsize=7)
            fig.colorbar(mesh, ax=ax, label=metric)
            ax.set_title(f"Cross-run {metric} heatmap{title_suffix}", fontsize=10.5)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def _plot_heatmap_hd_bin(self, configs, metrics, title_suffix=""):
        """One figure per metric, one subplot per config: cells colored by
        that metric's mean over a hd_bin (columns) x fraction_bin (rows)
        grid -- hd_bin recomputed fresh across all configs' pooled em_gt_hd
        values, same as plot_stat's grid_by="hd_bin" bar/violin path (see
        _compute_hd_bin_edges). All of a figure's subplots share one color
        scale (min/max across every config's grid) so e.g. a dvi-on vs
        dvi-off pair of subplots is visually comparable."""
        dfs = _load_configs_runs(configs, self._row_filters)
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
                mesh = ax.pcolormesh(grids[idx], cmap="viridis", vmin=vmin, vmax=vmax, shading="flat")
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
                            ax.text(ci + 0.5, ri + 0.5, f"{v:.3f}", ha="center", va="center",
                                     color="white", fontsize=6.5)
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
    simply absent (recorded.get(dest) is None either way -- there's no way to tell
    "missing key" from "explicit null" apart, and there's no need to); a _Not(value)
    requires the recorded value NOT equal that value (NOT be missing/null, if
    value is None) -- see _cartesian_axis_states, which builds these
    automatically. recorded: one leaf's args.json, dest-keyed."""
    for flag, requirement in flags.items():
        dest = _FLAG_TO_DEST.get(flag)
        if dest is None:
            raise KeyError(f"{flag!r} is not one of benchmark.py's mirrored caster/phlag flags")
        value = recorded.get(dest)
        if isinstance(requirement, _Not):
            negated = requirement.value
            is_negated = value is None if negated is None else value == _typed_value(flag, negated)
            if is_negated:
                return False
        elif requirement is None:
            if value is not None:
                return False
        elif value != _typed_value(flag, requirement):
            return False
    return True


def _strip_flag(flag):
    return flag.lstrip("-")


def _combo_label_part(flag, requirement):
    """Display text for one axis's state in a suffix combo label: an on-state renders as
    its bare value (value flags, e.g. "repulsion" for {"--ap": "repulsion"}) or bare flag
    name stripped of leading dashes (bool flags, e.g. "annealing"); requirement=None (the
    flag is missing from args.json) renders as "no-<flag>"; an off/negated state
    is dropped entirely (returns None) rather than shown with a "!" prefix."""
    if requirement is True:
        return _strip_flag(flag)
    if requirement is False or isinstance(requirement, _Not):
        return None
    if requirement is None:
        return f"no-{_strip_flag(flag)}"
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
    1.0, 1.5]} includes runs with no recorded lambda alongside 1.0 and 1.5.
    A list of dicts (see _expand_step_fractions) is already-resolved
    multi-flag states -- e.g. [{"-w": 50000, "-s": 40000}, ...] -- and passed
    through as-is instead of being re-wrapped in another {flag: v} layer."""
    if isinstance(value_spec, list) and value_spec and isinstance(value_spec[0], dict):
        return value_spec
    if isinstance(value_spec, list):
        return [{flag: v} for v in value_spec]
    return [{flag: value_spec}, {flag: _Not(value_spec)}]


def _as_step_fraction(value):
    """value (int/float/numeric string) as a float if it's in (0, 1] -- a
    step-size fraction, else None (including for bools and anything
    unparseable). caster.py's own step_size_or_fraction only treats (0, 1)
    (exclusive) as a fraction -- 1.0 falls through to int_or_abbrev there,
    which can't parse a bare "1.0" (no k/m suffix) and raises -- but this
    analysis tool's -w/-s coupling (_expand_step_fractions) has no such
    parsing conflict, so 1.0 is included here as a convenience meaning
    "step == window"."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        frac = float(value)
    except (TypeError, ValueError):
        return None
    return frac if 0 < frac <= 1 else None


def _expand_step_fractions(axes):
    """If "-s"'s value_spec in axes is a list containing any fraction in
    (0, 1) and "-w" is also present as a list, replaces both entries with
    "-w" alone holding one coupled state per (window, resolved step) pair --
    see resolve_configs_cartesian's docstring for why. Leaves axes untouched
    otherwise (including when -w is missing/not a list, or -s has no
    fractional entries)."""
    w_spec, s_spec = axes.get("-w"), axes.get("-s")
    if not isinstance(w_spec, list) or not isinstance(s_spec, list):
        return axes
    fractions = [(v, _as_step_fraction(v)) for v in s_spec if _as_step_fraction(v) is not None]
    if not fractions:
        return axes
    fraction_raws = {v for v, _ in fractions}
    literals = [v for v in s_spec if v not in fraction_raws]
    axes = dict(axes)
    del axes["-s"]
    states = []
    for w in w_spec:
        w_int = _typed_value("-w", w)
        for _, frac in fractions:
            states.append({"-w": w, "-s": max(1, round(frac * w_int))})
        for s in literals:
            states.append({"-w": w, "-s": s})
    axes["-w"] = states
    return axes


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


def _token_sort_key(token):
    """Sorts a heatmap axis's raw label tokens (always strings, split out of
    a config label) numerically when possible instead of lexicographically
    -- "80000" must sort after "40000", not before it."""
    try:
        return (0, float(token))
    except ValueError:
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
    if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) and value != 0 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if isinstance(value, int) and value != 0 and value % 1000 == 0:
        return f"{value // 1000}k"
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


def _load_configs_runs(configs, row_filters=None):
    """row_filters: {label: {col: value, ...}}, from resolve_configs_cartesian
    agg'ing by a _DATA_COLUMN_AXES name (e.g. "x_bin") -- restricts that
    config's rows to matching values after loading, since (unlike a flag) a
    row-level column can't narrow which dirs get read in the first place."""
    result = {}
    for label, dirs in configs:
        frames = [
            pd.read_csv(p, sep="\t")
            for d in dirs if (p := Path(d) / "runs.tsv").exists()
        ]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty and "category" in df.columns and "subcategory" in df.columns:
            df["merged_category"] = _merged_category(df["category"], df["subcategory"])
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
