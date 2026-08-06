"""
Benchmark driver, statistics layer, and Figure-3 rendering.

Three responsibilities live here, kept as separate classes/sections within
this one module (the same pattern ``caster.py`` uses for ``CasterPlotter``
and ``phlag.py`` uses for ``PhlagPlotter``):

  1. ``run_all()`` -- walk the source ``simulations/`` tree and run caster + phlag
     over every complete simulation that has no output yet.
  2. ``BenchmarkStats`` -- discover the finished runs, parse their ``report.tsv``
     files, bin each run into its Figure-3 cell, and aggregate per-cell summaries
     into a ``Figure3Data`` record. Pure computation -- no plotting.
  3. ``BenchmarkFigurePlotter`` -- consumes an already-built ``Figure3Data`` and
     draws it. Pure rendering -- no discovery, parsing, or aggregation.
"""

import re
import sys
import math
import pathlib
import argparse
import subprocess

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RED = "\033[91m"
RESET = "\033[0m"

DEFAULT_DIST_TYPE = "gaussian"
DEFAULT_WINDOW_SIZE = 50000
DEFAULT_STEP_SIZE = 1000
DEFAULT_CONCAT_FILENAME = "40-65.fa"


# ---------------------------------------------------------------------------
# Figure 3 layout
# ---------------------------------------------------------------------------

COL_RECOMBINATION_INCREASE = "recombination increase"
COL_RECOMBINATION_SUPPRESSION = "recombination suppression"
COL_POPSIZE_INCREASE = "population size increase"
COL_POPSIZE_DECREASE = "population size decrease"
COL_ADMIXTURE = "admixture"

#: Maps a (category, subcategory) pair from ``utils.get_simulation_categories``
#: onto its Figure-3 column. This mapping is an ASSUMPTION about how the
#: simulator's directory names line up with the paper's panel columns; it is
#: deliberately the single place that assumption is encoded, so correcting it
#: is a one-line edit here and nowhere else.
CATEGORY_TO_COLUMN = {
    ("recombination", "up"): COL_RECOMBINATION_INCREASE,
    ("recombination", "down"): COL_RECOMBINATION_SUPPRESSION,
    ("10X", "up"): COL_POPSIZE_INCREASE,
    ("10X", "down"): COL_POPSIZE_DECREASE,
    ("admixture", "low"): COL_ADMIXTURE,
    ("admixture", "high"): COL_ADMIXTURE,
}

FIGURE_COLUMNS = [
    COL_RECOMBINATION_INCREASE,
    COL_RECOMBINATION_SUPPRESSION,
    COL_POPSIZE_INCREASE,
    COL_POPSIZE_DECREASE,
    COL_ADMIXTURE,
]

#: The x-axis bin used for runs whose branch length could not be resolved. This
#: is the common case, not an edge case: /drive2/iang/63K.tre carries no internal
#: node labels matching the simulated clade/node names, so every non-admixture
#: run currently reports "Branch length (CU): N/A". The bin is therefore always
#: present, never opt-in -- otherwise panel A would be entirely empty.
UNKNOWN_BIN = "Unknown"

#: Branch-length (coalescent units) bins, as (label, low_exclusive, high_inclusive).
BRANCH_LENGTH_BIN_EDGES = [
    ("(0,0.1]", 0.0, 0.1),
    ("(0.1,1]", 0.1, 1.0),
    ("(1,5]", 1.0, 5.0),
]
BRANCH_LENGTH_BINS = [label for label, _, _ in BRANCH_LENGTH_BIN_EDGES] + [UNKNOWN_BIN]

#: Admixture reuses the low/high divergence-time split that
#: ``utils.get_simulation_categories`` already computes (threshold
#: ``utils.ADMIXTURE_DIVERGENCE_THRESHOLD_MYR``); no new numeric edges are
#: invented for this column.
ADMIXTURE_BINS = ["low", "high"]

#: Row facets: total portion of the gene trees affected by the anomaly.
FRACTION_BIN_EDGES = [
    ("(0,0.1]", 0.0, 0.1),
    ("(0.1,0.25]", 0.1, 0.25),
]
FRACTION_BINS = [label for label, _, _ in FRACTION_BIN_EDGES]

#: Slack allowed on a bin's upper edge before a value is pushed out of it.
#: The anomaly fraction is measured on the discrete window grid, so a nominal
#: 25% design lands at e.g. 1488/5951 = 0.250042 -- a hair over the (0.1,0.25]
#: edge. One window is ~1/6000 = 0.00017 of the genome, so 0.002 absorbs the
#: discretization with ~10x headroom while staying far away from any fraction
#: that would genuinely belong in a different bin.
BIN_EDGE_TOL = 0.002

ERRORBAR_KINDS = ("sd", "sem", "ci95")


# ---------------------------------------------------------------------------
# report.tsv parsing
# ---------------------------------------------------------------------------
#
# Reports are written by Phlag.compute_output(). Layout:
#   line 1              the command line that produced the report
#   header lines        plain "Key: value" lines (NOT comment-prefixed)
#   rel-error table     tab-separated rows (legacy reports use a "Topology X: ..." form)
#   "Path N final joint log-likelihood: ..."
#   state paths         one or more comma-separated 0/1 rows
#
# The header block is everything before the first state-path row. Legacy reports
# predate the Clade / Branch length / Anomaly fraction / Confusion / Label
# polarity lines, so every field below is optional and absence is never fatal.

_RE_STATE_PATH = re.compile(r'^[01](?:,[01])+$')
_RE_CLADE = re.compile(r'^Clade:\s*(.+?)\s*$')
_RE_CLADE_NUMBER = re.compile(r'^Clade number:\s*(.+?)\s*$')
_RE_BRANCH_LENGTH = re.compile(r'^Branch length \(CU[^)]*\):\s*(.+?)\s*$')
_RE_ANOMALY_FRACTION = re.compile(
    r'^Anomaly fraction:\s*([0-9.eE+-]+)(?:\s*\((\d+)\s*/\s*(\d+)\s+windows\))?\s*$'
)
_RE_CONFUSION = re.compile(
    r'^Confusion:\s*TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+TN=(\d+)\s*$'
)
_RE_FLIPPED = re.compile(
    r'^Label polarity flipped for evaluation:\s*(True|False)\s*$', re.IGNORECASE
)
_RE_METRICS = re.compile(
    r'^TPR:\s*([-\d.eE+]+),\s*FPR:\s*([-\d.eE+]+),\s*'
    r'F1:\s*([-\d.eE+]+),\s*Accuracy:\s*([-\d.eE+]+)\s*$'
)


def _parse_branch_length_value(value_str):
    """
    Turns the right-hand side of a 'Branch length (CU...):' header into a float.
    Both emitted forms are handled:
      '0.123456'                                     -> 0.123456
      'N/A (node not found directly in clade_cu.csv)' -> None
    """
    token = (value_str or "").strip().split()
    if not token:
        return None
    head = token[0]
    if head.upper().startswith("N/A"):
        return None
    try:
        return float(head)
    except ValueError:
        return None


def parse_report(report_path):
    """
    Parses a phlag report.tsv header block into a plain dict.

    Every key is always present; values are None when the corresponding header
    line is missing, which is the normal state of affairs for reports written
    before the Clade / Anomaly fraction / Confusion / Label polarity lines
    existed. Raises only on unreadable files.
    """
    parsed = {
        "clade_name": None,
        "clade_number": None,
        "branch_length": None,
        "branch_length_present": False,
        "anomaly_fraction": None,
        "n_anomaly_windows": None,
        "n_windows": None,
        "tp": None, "fp": None, "fn": None, "tn": None,
        "label_flipped": None,
        "tpr": None, "fpr": None, "f1": None, "accuracy": None,
        "n_path_entries": None,
    }

    with open(report_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            # The header block ends at the first Viterbi state path.
            if _RE_STATE_PATH.match(line):
                parsed["n_path_entries"] = line.count(",") + 1
                break

            m = _RE_METRICS.match(line)
            if m:
                parsed["tpr"] = float(m.group(1))
                parsed["fpr"] = float(m.group(2))
                parsed["f1"] = float(m.group(3))
                parsed["accuracy"] = float(m.group(4))
                continue

            m = _RE_CLADE.match(line)
            if m:
                value = m.group(1)
                parsed["clade_name"] = None if value.upper().startswith("N/A") else value
                continue

            m = _RE_CLADE_NUMBER.match(line)
            if m:
                parsed["clade_number"] = m.group(1)
                continue

            m = _RE_BRANCH_LENGTH.match(line)
            if m:
                parsed["branch_length_present"] = True
                parsed["branch_length"] = _parse_branch_length_value(m.group(1))
                continue

            m = _RE_ANOMALY_FRACTION.match(line)
            if m:
                parsed["anomaly_fraction"] = float(m.group(1))
                if m.group(2) is not None:
                    parsed["n_anomaly_windows"] = int(m.group(2))
                    parsed["n_windows"] = int(m.group(3))
                continue

            m = _RE_CONFUSION.match(line)
            if m:
                parsed["tp"] = int(m.group(1))
                parsed["fp"] = int(m.group(2))
                parsed["fn"] = int(m.group(3))
                parsed["tn"] = int(m.group(4))
                continue

            m = _RE_FLIPPED.match(line)
            if m:
                parsed["label_flipped"] = (m.group(1).lower() == "true")
                continue

    if parsed["n_windows"] is None and parsed["n_path_entries"] is not None:
        parsed["n_windows"] = parsed["n_path_entries"]

    return parsed


def nominal_anomaly_fraction(pattern_str):
    """
    Fallback anomaly fraction for reports written before the 'Anomaly fraction:'
    header existed: the NOMINAL portion of the genome the locus pattern marks as
    anomalous, e.g. '40-65' -> 0.25, '40-45' -> 0.05, '70-80,85-100' -> 0.25.

    This is a design-level number read off the pattern alone; the measured
    fraction in the report header is the authoritative one because it also
    accounts for where the window grid actually falls. Runs that fall back to
    this are tagged 'nominal' in benchmark_runs.tsv.
    """
    from .utils import parse_pattern_string

    if not pattern_str:
        return None
    try:
        _blocks, anomaly_intervals, total_span = parse_pattern_string(str(pattern_str))
    except Exception:
        return None
    if not anomaly_intervals or not total_span or total_span <= 0:
        return None
    covered = sum(max(0.0, float(end) - float(start)) for start, end in anomaly_intervals)
    fraction = covered / float(total_span)
    return fraction if 0.0 < fraction <= 1.0 else None


def assign_bin(value, bin_edges, tol=BIN_EDGE_TOL):
    """
    Assigns a value to the first (label, low, high) bin with low < value <= high + tol.
    Returns None when the value falls outside every bin (or is None).
    """
    if value is None:
        return None
    for label, low, high in bin_edges:
        if value > low and value <= high + tol:
            return label
    return None


# ---------------------------------------------------------------------------
# Data contract between the stats layer and the plotting layer
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One discovered phlag run: everything needed to trace a bar back to its source."""
    run_id: str
    source_leaf: str
    report_path: str
    category: str
    subcategory: str
    column: str
    pattern: str
    clade_name: Optional[str] = None
    clade_number: Optional[str] = None
    branch_length: Optional[float] = None
    x_variable: str = ""
    x_value: Optional[float] = None
    x_bin: Optional[str] = None
    anomaly_fraction: Optional[float] = None
    anomaly_fraction_source: str = ""
    fraction_bin: Optional[str] = None
    tpr: Optional[float] = None
    fpr: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tn: Optional[int] = None
    n_windows: Optional[int] = None
    label_flipped: Optional[bool] = None
    in_panel_a: bool = False
    in_panel_b: bool = False
    exclusion: str = ""


@dataclass
class PanelACell:
    """One grouped-bar cell of panel A: mean F1 over the runs in this cell."""
    column: str
    fraction_bin: str
    x_bin: str
    n_runs: int
    mean_f1: float
    sd_f1: float
    err_minus: float
    err_plus: float
    run_ids: List[str] = field(default_factory=list)


@dataclass
class PanelBCell:
    """
    One point of panel B: the macro-averaged TPR/FPR over the runs in this cell.

    Macro (mean of each run's own rate), not micro (a rate recomputed from pooled
    confusion counts), so that panel B estimates the same quantity panel A does --
    performance on a randomly drawn replicate run rather than on a randomly drawn
    genomic window.
    """
    column: str
    fraction_bin: str
    n_runs: int
    mean_tpr: float
    mean_fpr: float
    sd_tpr: float
    sd_fpr: float
    tpr_err_minus: float
    tpr_err_plus: float
    fpr_err_minus: float
    fpr_err_plus: float
    run_ids: List[str] = field(default_factory=list)


@dataclass
class Figure3Data:
    """
    The complete, already-aggregated contents of the two Figure-3 panels.

    This is the ONLY thing that crosses from the statistics layer
    (``BenchmarkStats``) to the rendering layer (``BenchmarkFigurePlotter``,
    below). Everything the plotter draws is a field read off this record; the
    plotter opens no files and computes no aggregates.
    """
    columns: List[str]
    fraction_bins: List[str]
    x_bins_by_column: Dict[str, List[str]]
    x_axis_label_by_column: Dict[str, str]
    panel_a: Dict[Tuple[str, str, str], PanelACell]
    panel_b: Dict[Tuple[str, str], PanelBCell]
    errorbar: str
    dist_type: str
    window_size: int
    step_size: int
    n_reports_found: int = 0
    n_runs_panel_a: int = 0
    n_runs_panel_b: int = 0
    #: Pre-totalled here rather than in the plotter, so the plotter never has to
    #: add anything up -- it only ever reads fields off this record.
    n_runs_excluded: int = 0
    exclusions: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _t_critical_95(n):
    """Two-sided 95% critical value for n observations; normal approx if scipy is absent."""
    if n < 2:
        return 0.0
    try:
        from scipy import stats as scipy_stats
        return float(scipy_stats.t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def summarize_values(values, errorbar="sd", low=0.0, high=1.0):
    """
    Reduces a list of per-run values to (mean, sd, err_minus, err_plus).

    The whisker halves are returned pre-clipped to [low, high] so a bar's error
    bar can never run below 0 or above 1 -- both F1 and TPR/FPR are bounded
    rates, and an unclipped +/- SD on a mean near 1.0 would draw outside the axis.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = math.sqrt(max(0.0, variance))
    else:
        sd = 0.0

    if errorbar == "sd":
        err = sd
    elif errorbar == "sem":
        err = sd / math.sqrt(n)
    elif errorbar == "ci95":
        err = (sd / math.sqrt(n)) * _t_critical_95(n)
    else:
        raise ValueError(f"Unknown errorbar kind '{errorbar}' (expected one of {ERRORBAR_KINDS})")

    lower = max(low, mean - err)
    upper = min(high, mean + err)
    return mean, sd, mean - lower, upper - mean


class BenchmarkStats:
    """
    Discovers finished phlag runs, parses their reports, bins them, and aggregates.

    Discovery walks the SOURCE ``simulations/`` tree rather than the phlag output
    tree: the output tree carries stale directories left over from earlier runs
    (e.g. a ``recombination/down/N115`` output whose source actually lives under
    ``recombination/up/N115``) and interleaves several dist-type/window/step
    configurations. Walking the source tree and computing each leaf's expected
    output directory keeps the run set pinned to one configuration and free of
    orphans.

    Statistics only -- this class never imports matplotlib and never draws.
    """

    def __init__(self, sim_root=None, dist_type=DEFAULT_DIST_TYPE,
                 window_size=DEFAULT_WINDOW_SIZE, step_size=DEFAULT_STEP_SIZE,
                 errorbar="sd"):
        from .utils import get_data_dir

        if errorbar not in ERRORBAR_KINDS:
            raise ValueError(f"Unknown errorbar kind '{errorbar}' (expected one of {ERRORBAR_KINDS})")

        self.sim_root = pathlib.Path(sim_root) if sim_root is not None else get_data_dir() / "simulations"
        self.dist_type = dist_type
        self.window_size = window_size
        self.step_size = step_size
        self.errorbar = errorbar

        self.runs = []                      # List[RunRecord], including excluded ones
        self.notes = []                     # Free-text diagnostics for the operator
        self.n_leaf_dirs = 0
        self.n_leaf_dirs_uncategorized = 0
        self.n_leaf_dirs_without_output = 0

    # -- paths ------------------------------------------------------------

    def default_stats_dir(self):
        """<phlag_base>/<dist_type>/w<W>_s<S>/benchmark/"""
        from .utils import get_data_dir, get_phlag_output_base
        from .caster import format_val

        window_str = format_val(self.window_size)
        step_str = format_val(self.step_size)
        return (
            get_phlag_output_base(get_data_dir())
            / self.dist_type
            / f"w{window_str}_s{step_str}"
            / "benchmark"
        )

    # -- discovery --------------------------------------------------------

    def collect(self):
        """
        Populates ``self.runs`` with one RunRecord per report.tsv found under the
        expected output directory of a categorized source simulation.
        """
        from .utils import get_simulation_categories

        self.runs = []
        self.notes = []

        if not self.sim_root.exists():
            self.notes.append(f"simulations root not found at '{self.sim_root}' -- nothing to summarize")
            return self.runs

        leaf_dirs = discover_leaf_dirs(self.sim_root)
        self.n_leaf_dirs = len(leaf_dirs)
        self.n_leaf_dirs_uncategorized = 0
        self.n_leaf_dirs_without_output = 0

        for leaf_dir in leaf_dirs:
            rel_leaf = leaf_dir.relative_to(self.sim_root).as_posix()

            # Always categorize from the FULL path: bare node names such as
            # 'N228' or 'N717' occur under several category directories at once,
            # and the bare-stub fallbacks inside get_simulation_categories would
            # silently pick just one of them.
            cats = get_simulation_categories(str(leaf_dir))
            if not cats:
                self.n_leaf_dirs_uncategorized += 1
                self.notes.append(f"[skip] {rel_leaf}: not a recognized simulation category")
                continue

            category, subcategory = cats[0], cats[1]
            column = CATEGORY_TO_COLUMN.get((category, subcategory))
            if column is None:
                self.n_leaf_dirs_uncategorized += 1
                self.notes.append(
                    f"[skip] {rel_leaf}: category ('{category}', '{subcategory}') "
                    f"has no Figure-3 column in CATEGORY_TO_COLUMN"
                )
                continue

            sim_output_dir = get_expected_sim_output_dir(
                leaf_dir, leaf_dir,
                dist_type=self.dist_type,
                window_size=self.window_size,
                step_size=self.step_size,
            )
            report_paths = sorted(sim_output_dir.glob("*/phlag/report.tsv"))
            if not report_paths:
                self.n_leaf_dirs_without_output += 1
                self.notes.append(f"[skip] {rel_leaf}: no report.tsv under {sim_output_dir}")
                continue

            for report_path in report_paths:
                self.runs.append(
                    self._build_record(report_path, leaf_dir, rel_leaf, category, subcategory, column)
                )

        return self.runs

    def _build_record(self, report_path, leaf_dir, rel_leaf, category, subcategory, column):
        """Parses one report.tsv and bins it. Never raises on a malformed report."""
        # .../<sim>/<pattern>/phlag/report.tsv  ->  <pattern>
        pattern = report_path.parent.parent.name
        record = RunRecord(
            run_id=f"{rel_leaf}::{pattern}",
            source_leaf=rel_leaf,
            report_path=str(report_path),
            category=category,
            subcategory=subcategory,
            column=column,
            pattern=pattern,
        )

        try:
            parsed = parse_report(report_path)
        except Exception as exc:
            record.exclusion = f"unreadable report ({exc})"
            return record

        record.clade_name = parsed["clade_name"]
        record.clade_number = parsed["clade_number"]
        record.branch_length = parsed["branch_length"]
        record.tpr = parsed["tpr"]
        record.fpr = parsed["fpr"]
        record.f1 = parsed["f1"]
        record.accuracy = parsed["accuracy"]
        record.tp, record.fp = parsed["tp"], parsed["fp"]
        record.fn, record.tn = parsed["fn"], parsed["tn"]
        record.n_windows = parsed["n_windows"]
        record.label_flipped = parsed["label_flipped"]

        # -- affected gene-tree fraction (row facet) --
        if parsed["anomaly_fraction"] is not None:
            record.anomaly_fraction = parsed["anomaly_fraction"]
            record.anomaly_fraction_source = "report"
        else:
            record.anomaly_fraction = nominal_anomaly_fraction(pattern)
            record.anomaly_fraction_source = "nominal" if record.anomaly_fraction is not None else "missing"
        record.fraction_bin = assign_bin(record.anomaly_fraction, FRACTION_BIN_EDGES)

        # -- x-axis position --
        if column == COL_ADMIXTURE:
            from .utils import get_admixture_divergence_time, ADMIXTURE_DIVERGENCE_THRESHOLD_MYR
            record.x_variable = "divergence time (Myr)"
            record.x_value = get_admixture_divergence_time(leaf_dir.name)
            # Reuse the exact low/high split get_simulation_categories already made,
            # rather than re-deriving a threshold here.
            record.x_bin = subcategory if subcategory in ADMIXTURE_BINS else None
            if record.x_bin is None and record.x_value is not None:
                record.x_bin = (
                    "low" if record.x_value < ADMIXTURE_DIVERGENCE_THRESHOLD_MYR else "high"
                )
        else:
            record.x_variable = "branch length (CU)"
            record.x_value = record.branch_length
            if record.branch_length is None:
                record.x_bin = UNKNOWN_BIN
            else:
                record.x_bin = assign_bin(record.branch_length, BRANCH_LENGTH_BIN_EDGES)

        # -- eligibility --
        has_rates = record.tpr is not None and record.fpr is not None
        if not has_rates:
            record.exclusion = "no TPR/FPR metrics line in report"
            return record
        if record.fraction_bin is None:
            record.exclusion = (
                f"anomaly fraction {record.anomaly_fraction} outside every row bin"
                if record.anomaly_fraction is not None
                else "anomaly fraction could not be determined"
            )
            return record

        # A run with valid rates and a row bin ALWAYS contributes to panel B --
        # its branch-length availability is a panel-A concern only.
        record.in_panel_b = True

        if record.f1 is None:
            record.exclusion = "no F1 in report (panel B only)"
        elif record.x_bin is None:
            record.exclusion = f"x value {record.x_value} outside every x bin (panel B only)"
        else:
            record.in_panel_a = True

        return record

    # -- aggregation ------------------------------------------------------

    def aggregate(self):
        """Reduces ``self.runs`` to a Figure3Data. Call ``collect()`` first."""
        panel_a_groups = defaultdict(list)
        panel_b_groups = defaultdict(list)
        exclusions = defaultdict(int)

        for record in self.runs:
            if record.exclusion:
                exclusions[record.exclusion] += 1
            if record.in_panel_a:
                panel_a_groups[(record.column, record.fraction_bin, record.x_bin)].append(record)
            if record.in_panel_b:
                panel_b_groups[(record.column, record.fraction_bin)].append(record)

        panel_a = {}
        for key, records in panel_a_groups.items():
            column, fraction_bin, x_bin = key
            mean, sd, err_minus, err_plus = summarize_values(
                [r.f1 for r in records], self.errorbar
            )
            panel_a[key] = PanelACell(
                column=column, fraction_bin=fraction_bin, x_bin=x_bin,
                n_runs=len(records), mean_f1=mean, sd_f1=sd,
                err_minus=err_minus, err_plus=err_plus,
                run_ids=sorted(r.run_id for r in records),
            )

        panel_b = {}
        for key, records in panel_b_groups.items():
            column, fraction_bin = key
            tpr_mean, tpr_sd, tpr_minus, tpr_plus = summarize_values(
                [r.tpr for r in records], self.errorbar
            )
            fpr_mean, fpr_sd, fpr_minus, fpr_plus = summarize_values(
                [r.fpr for r in records], self.errorbar
            )
            panel_b[key] = PanelBCell(
                column=column, fraction_bin=fraction_bin, n_runs=len(records),
                mean_tpr=tpr_mean, mean_fpr=fpr_mean,
                sd_tpr=tpr_sd, sd_fpr=fpr_sd,
                tpr_err_minus=tpr_minus, tpr_err_plus=tpr_plus,
                fpr_err_minus=fpr_minus, fpr_err_plus=fpr_plus,
                run_ids=sorted(r.run_id for r in records),
            )

        x_bins_by_column = {}
        x_axis_label_by_column = {}
        for column in FIGURE_COLUMNS:
            if column == COL_ADMIXTURE:
                from .utils import ADMIXTURE_DIVERGENCE_THRESHOLD_MYR
                x_bins_by_column[column] = list(ADMIXTURE_BINS)
                x_axis_label_by_column[column] = (
                    f"Divergence time (split at {ADMIXTURE_DIVERGENCE_THRESHOLD_MYR:g} Myr)"
                )
            else:
                x_bins_by_column[column] = list(BRANCH_LENGTH_BINS)
                x_axis_label_by_column[column] = "Branch length (CU)"

        return Figure3Data(
            columns=list(FIGURE_COLUMNS),
            fraction_bins=list(FRACTION_BINS),
            x_bins_by_column=x_bins_by_column,
            x_axis_label_by_column=x_axis_label_by_column,
            panel_a=panel_a,
            panel_b=panel_b,
            errorbar=self.errorbar,
            dist_type=self.dist_type,
            window_size=self.window_size,
            step_size=self.step_size,
            n_reports_found=len(self.runs),
            n_runs_panel_a=sum(1 for r in self.runs if r.in_panel_a),
            n_runs_panel_b=sum(1 for r in self.runs if r.in_panel_b),
            n_runs_excluded=sum(exclusions.values()),
            exclusions=dict(exclusions),
        )

    # -- outputs ----------------------------------------------------------

    RUN_COLUMNS = [
        "run_id", "source_leaf", "category", "subcategory", "column", "pattern",
        "anomaly_fraction", "anomaly_fraction_source", "fraction_bin",
        "x_variable", "x_value", "x_bin", "branch_length_cu",
        "clade_name", "clade_number",
        "tpr", "fpr", "f1", "accuracy", "tp", "fp", "fn", "tn",
        "n_windows", "label_flipped",
        "in_panel_a", "in_panel_b", "exclusion", "report_path",
    ]

    BIN_COLUMNS = [
        "panel", "column", "fraction_bin", "x_bin", "metric",
        "n_runs", "mean", "sd", "err_minus", "err_plus", "errorbar", "run_ids",
    ]

    @staticmethod
    def _fmt(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def write_tables(self, out_dir, figure_data=None):
        """
        Writes benchmark_runs.tsv (one row per discovered report, excluded ones
        included with their reason) and benchmark_bins.tsv (one row per
        panel/cell/metric, carrying the source run ids) next to the PNGs.
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if figure_data is None:
            figure_data = self.aggregate()

        runs_path = out_dir / "benchmark_runs.tsv"
        with open(runs_path, "w") as handle:
            handle.write("\t".join(self.RUN_COLUMNS) + "\n")
            for record in sorted(self.runs, key=lambda r: r.run_id):
                row = {
                    "run_id": record.run_id,
                    "source_leaf": record.source_leaf,
                    "category": record.category,
                    "subcategory": record.subcategory,
                    "column": record.column,
                    "pattern": record.pattern,
                    "anomaly_fraction": record.anomaly_fraction,
                    "anomaly_fraction_source": record.anomaly_fraction_source,
                    "fraction_bin": record.fraction_bin,
                    "x_variable": record.x_variable,
                    "x_value": record.x_value,
                    "x_bin": record.x_bin,
                    "branch_length_cu": record.branch_length,
                    "clade_name": record.clade_name,
                    "clade_number": record.clade_number,
                    "tpr": record.tpr, "fpr": record.fpr,
                    "f1": record.f1, "accuracy": record.accuracy,
                    "tp": record.tp, "fp": record.fp,
                    "fn": record.fn, "tn": record.tn,
                    "n_windows": record.n_windows,
                    "label_flipped": record.label_flipped,
                    "in_panel_a": record.in_panel_a,
                    "in_panel_b": record.in_panel_b,
                    "exclusion": record.exclusion,
                    "report_path": record.report_path,
                }
                handle.write("\t".join(self._fmt(row[c]) for c in self.RUN_COLUMNS) + "\n")

        bins_path = out_dir / "benchmark_bins.tsv"
        with open(bins_path, "w") as handle:
            handle.write("\t".join(self.BIN_COLUMNS) + "\n")

            def emit(panel, column, fraction_bin, x_bin, metric, n_runs,
                     mean, sd, err_minus, err_plus, run_ids):
                row = {
                    "panel": panel, "column": column, "fraction_bin": fraction_bin,
                    "x_bin": x_bin, "metric": metric, "n_runs": n_runs,
                    "mean": mean, "sd": sd,
                    "err_minus": err_minus, "err_plus": err_plus,
                    "errorbar": self.errorbar, "run_ids": ";".join(run_ids),
                }
                handle.write("\t".join(self._fmt(row[c]) for c in self.BIN_COLUMNS) + "\n")

            for key in sorted(figure_data.panel_a.keys()):
                cell = figure_data.panel_a[key]
                emit("A", cell.column, cell.fraction_bin, cell.x_bin, "f1", cell.n_runs,
                     cell.mean_f1, cell.sd_f1, cell.err_minus, cell.err_plus, cell.run_ids)

            for key in sorted(figure_data.panel_b.keys()):
                cell = figure_data.panel_b[key]
                emit("B", cell.column, cell.fraction_bin, "", "tpr", cell.n_runs,
                     cell.mean_tpr, cell.sd_tpr, cell.tpr_err_minus, cell.tpr_err_plus,
                     cell.run_ids)
                emit("B", cell.column, cell.fraction_bin, "", "fpr", cell.n_runs,
                     cell.mean_fpr, cell.sd_fpr, cell.fpr_err_minus, cell.fpr_err_plus,
                     cell.run_ids)

        return runs_path, bins_path

    def print_diagnostics(self, figure_data=None):
        """Prints why every discovered run did or did not end up in a bar/point."""
        if figure_data is None:
            figure_data = self.aggregate()

        print("\n=== Benchmark summary ===")
        print(f"Configuration: dist-type={self.dist_type}, window={self.window_size}, "
              f"step={self.step_size}, errorbar={self.errorbar}")
        print(f"Source tree: {self.sim_root}")
        print(f"Leaf simulation dirs walked: {self.n_leaf_dirs} "
              f"({self.n_leaf_dirs_uncategorized} uncategorized, "
              f"{self.n_leaf_dirs_without_output} with no output for this config)")
        print(f"Reports found: {figure_data.n_reports_found}  |  "
              f"panel A runs: {figure_data.n_runs_panel_a}  |  "
              f"panel B runs: {figure_data.n_runs_panel_b}")

        if figure_data.exclusions:
            print("\nExclusions:")
            for reason, count in sorted(figure_data.exclusions.items()):
                print(f"  {count:4d}  {reason}")

        print("\nPanel A cells (mean F1):")
        for fraction_bin in figure_data.fraction_bins:
            row_cells = [
                (column, x_bin, figure_data.panel_a[(column, fraction_bin, x_bin)])
                for column in figure_data.columns
                for x_bin in figure_data.x_bins_by_column[column]
                if (column, fraction_bin, x_bin) in figure_data.panel_a
            ]
            if not row_cells:
                print(f"  fraction {fraction_bin}: EMPTY (no runs fall in this row)")
                continue
            print(f"  fraction {fraction_bin}:")
            for column, x_bin, cell in row_cells:
                print(f"    {column:28s} x={x_bin:10s} n={cell.n_runs:3d}  "
                      f"F1={cell.mean_f1:.4f} +/-{cell.sd_f1:.4f}")

        print("\nPanel B cells (macro-averaged TPR/FPR):")
        for fraction_bin in figure_data.fraction_bins:
            row_cells = [
                (column, figure_data.panel_b[(column, fraction_bin)])
                for column in figure_data.columns
                if (column, fraction_bin) in figure_data.panel_b
            ]
            if not row_cells:
                print(f"  fraction {fraction_bin}: EMPTY (no runs fall in this row)")
                continue
            print(f"  fraction {fraction_bin}:")
            for column, cell in row_cells:
                print(f"    {column:28s} n={cell.n_runs:3d}  "
                      f"TPR={cell.mean_tpr:.4f}  FPR={cell.mean_fpr:.4f}")

        if self.notes:
            print("\nDiscovery notes:")
            for note in self.notes:
                print(f"  {note}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
#
# BenchmarkFigurePlotter consumes an already-aggregated Figure3Data and draws
# it. It never opens a data file, never walks a directory, and never computes
# a mean, a standard deviation, or any other aggregate -- every number it
# draws was computed by BenchmarkStats above. That split is the same one
# Phlag / PhlagPlotter use in phlag/phlag.py: the plotter consumes a finished
# object's attributes and recomputes nothing.

# --- Chart chrome & ink ------------------------------------------------------
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"

# --- Series color --------------------------------------------------------------
# The x axis of panel A is an ORDERED variable (branch length in coalescent
# units; divergence time for admixture), so its bars take an ordinal single-hue
# blue ramp rather than categorical hues. These are the documented ramp steps
# 250/400/550, validated for ordinal use on the light surface (the lightest,
# step 250, clears the 2:1 floor against #fcfcfb).
_ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#1c5cab"]

# "Unknown" is the absence of a magnitude, not another step of one, so it is
# drawn in the neutral muted gray and never given a hue.
_UNKNOWN_COLOR = _INK_MUTED

# Panel B carries a single series (Phlag); categorical slot 1.
_SERIES_PHLAG = "#2a78d6"

_BAR_WIDTH = 0.62
_MARKER_SIZE = 8.0


def _ordinal_colors(n):
    """A light-to-dark single-hue ramp of n ordered steps."""
    if n <= 0:
        return []
    if n == 1:
        return [_ORDINAL_BLUE[1]]
    if n == 2:
        return [_ORDINAL_BLUE[0], _ORDINAL_BLUE[2]]
    if n == 3:
        return list(_ORDINAL_BLUE)
    cmap = matplotlib.colormaps["Blues"]
    return [cmap(v) for v in np.linspace(0.40, 0.92, n)]


class BenchmarkFigurePlotter:
    """
    Draws the two Figure-3 panels from a ``Figure3Data``.

    Panel A: a fraction-bin x category grid of grouped bar charts, F1 on y and
    the branch-length (or admixture divergence-time) bins on x.
    Panel B: the same grid with a single TPR-vs-FPR point per cell. This repo
    implements only Phlag -- there is no baseline method and no rho sweep -- so a
    cell holds one point, not a curve.

    Rendering only; see the section comment above.
    """

    def __init__(self, figure_data: "Figure3Data", out_dir):
        self.data = figure_data
        self.out_dir = pathlib.Path(out_dir)

    # -- shared chrome ----------------------------------------------------

    def _apply_theme(self):
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "figure.facecolor": _SURFACE,
            "axes.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
            "text.color": _INK_PRIMARY,
            "axes.labelcolor": _INK_SECONDARY,
            "xtick.color": _INK_MUTED,
            "ytick.color": _INK_MUTED,
        })

    def _style_axes(self, ax, grid_axis="y"):
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis, color=_GRIDLINE, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_BASELINE)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(labelsize=7.5, length=3, color=_BASELINE)

    def _annotate_empty(self, ax):
        ax.text(
            0.5, 0.5, "no runs", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, style="italic", color=_INK_MUTED,
        )

    def _footer(self, fig):
        data = self.data
        fig.text(
            0.006, 0.004,
            f"Phlag {data.dist_type}, window {data.window_size} / step {data.step_size}  |  "
            f"error bars: {data.errorbar} (clipped to [0,1])  |  "
            f"{data.n_reports_found} reports found, "
            f"{data.n_runs_panel_a} in panel A, {data.n_runs_panel_b} in panel B, "
            f"{data.n_runs_excluded} excluded\n"
            f"Every bar/point is traceable: per-run detail in benchmark_runs.tsv, "
            f"per-cell detail (with source run ids) in benchmark_bins.tsv",
            fontsize=6.5, color=_INK_MUTED, ha="left", va="bottom", linespacing=1.5,
        )

    # -- panel A ----------------------------------------------------------

    def render_panel_a(self, filename="figure3_panel_a.png"):
        data = self.data
        n_rows, n_cols = len(data.fraction_bins), len(data.columns)

        self._apply_theme()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
            squeeze=False,
        )

        for row_idx, fraction_bin in enumerate(data.fraction_bins):
            for col_idx, column in enumerate(data.columns):
                ax = axes[row_idx][col_idx]
                x_bins = data.x_bins_by_column[column]
                ordered_bins = [b for b in x_bins if b != UNKNOWN_BIN]
                ramp = _ordinal_colors(len(ordered_bins))
                color_for_bin = {b: ramp[i] for i, b in enumerate(ordered_bins)}
                color_for_bin[UNKNOWN_BIN] = _UNKNOWN_COLOR

                self._style_axes(ax)
                ax.set_xlim(-0.7, len(x_bins) - 0.3)
                ax.set_ylim(0.0, 1.18)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticks(list(range(len(x_bins))))
                ax.set_xticklabels(x_bins, rotation=30, ha="right", fontsize=7.5)

                drew_any = False
                for pos, x_bin in enumerate(x_bins):
                    cell = data.panel_a.get((column, fraction_bin, x_bin))
                    if cell is None or cell.n_runs == 0:
                        continue
                    drew_any = True
                    # A 1.2pt surface-colored edge keeps adjacent fills separated.
                    ax.bar(
                        pos, cell.mean_f1, width=_BAR_WIDTH,
                        color=color_for_bin[x_bin],
                        edgecolor=_SURFACE, linewidth=1.2, zorder=2,
                    )
                    ax.errorbar(
                        pos, cell.mean_f1,
                        yerr=[[cell.err_minus], [cell.err_plus]],
                        fmt="none", ecolor=_INK_SECONDARY,
                        elinewidth=1.2, capsize=3, capthick=1.2, zorder=3,
                    )
                    ax.text(
                        pos, min(1.0, cell.mean_f1 + cell.err_plus) + 0.035,
                        f"n={cell.n_runs}",
                        ha="center", va="bottom", fontsize=6.5, color=_INK_MUTED,
                    )

                if not drew_any:
                    self._annotate_empty(ax)

                if row_idx == 0:
                    ax.set_title(column, fontsize=9.5, color=_INK_PRIMARY, pad=8)
                # Hide interior tick labels via tick_params, never set_*ticklabels([]):
                # the ticks are on a FixedLocator here, and a length mismatch against
                # an empty label list is a ValueError in matplotlib >= 3.5.
                if row_idx == n_rows - 1:
                    ax.set_xlabel(data.x_axis_label_by_column[column], fontsize=8)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    # One line, not two: a two-line rotated ylabel renders as two
                    # side-by-side vertical text columns and reads as two labels.
                    ax.set_ylabel(f"F1   |   affected fraction {fraction_bin}", fontsize=8.5)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fig.suptitle(
            "Phlag anomaly detection: F1 by branch length and affected gene-tree fraction",
            fontsize=12, color=_INK_PRIMARY, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        self._footer(fig)

        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    # -- panel B ----------------------------------------------------------

    def render_panel_b(self, filename="figure3_panel_b.png"):
        data = self.data
        n_rows, n_cols = len(data.fraction_bins), len(data.columns)

        self._apply_theme()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
            squeeze=False,
        )

        for row_idx, fraction_bin in enumerate(data.fraction_bins):
            for col_idx, column in enumerate(data.columns):
                ax = axes[row_idx][col_idx]
                self._style_axes(ax, grid_axis="both")
                ax.set_xlim(-0.03, 1.03)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                # Chance line: a point on it detects no better than random.
                ax.plot([0, 1], [0, 1], color=_GRIDLINE, linewidth=1.0,
                        linestyle="--", zorder=1)

                cell = data.panel_b.get((column, fraction_bin))
                if cell is None or cell.n_runs == 0:
                    self._annotate_empty(ax)
                else:
                    ax.errorbar(
                        cell.mean_fpr, cell.mean_tpr,
                        xerr=[[cell.fpr_err_minus], [cell.fpr_err_plus]],
                        yerr=[[cell.tpr_err_minus], [cell.tpr_err_plus]],
                        fmt="o", markersize=_MARKER_SIZE, color=_SERIES_PHLAG,
                        ecolor=_INK_SECONDARY, elinewidth=1.2,
                        capsize=3, capthick=1.2,
                        markeredgecolor=_SURFACE, markeredgewidth=1.5, zorder=3,
                    )
                    ax.annotate(
                        f"n={cell.n_runs}",
                        xy=(cell.mean_fpr, cell.mean_tpr),
                        xytext=(12, -17), textcoords="offset points",
                        fontsize=6.5, color=_INK_MUTED,
                    )

                if row_idx == 0:
                    ax.set_title(column, fontsize=9.5, color=_INK_PRIMARY, pad=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("FPR", fontsize=8.5)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    ax.set_ylabel(f"TPR   |   affected fraction {fraction_bin}", fontsize=8.5)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fig.suptitle(
            "Phlag anomaly detection: TPR vs FPR by affected gene-tree fraction",
            fontsize=12, color=_INK_PRIMARY, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        self._footer(fig)

        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    # -- entry point ------------------------------------------------------

    def render(self):
        """Renders both panels; returns the list of written PNG paths."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return [self.render_panel_a(), self.render_panel_b()]


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark: run caster and phlag over every dataset in simulations/ "
                    "(skipping stages whose output already exists), then summarize every "
                    "finished run into the Figure-3 panels."
    )
    parser.add_argument(
        "--rerun",
        dest="rerun",
        action="store_true",
        help="Force caster and phlag to re-run even where output already exists (default: skip what's already there)",
    )
    parser.add_argument(
        "-d",
        "--dist-type",
        dest="dist_type",
        default=DEFAULT_DIST_TYPE,
        choices=["gaussian", "gmm"],
        help=f"Distribution type, threaded through to caster and phlag and used to locate "
             f"outputs when summarizing (default: {DEFAULT_DIST_TYPE})",
    )
    parser.add_argument(
        "--stats-out",
        dest="stats_out",
        type=pathlib.Path,
        default=None,
        help="Directory for the summary tables and figures "
             "(default: <phlag_base>/<dist-type>/w<W>_s<S>/benchmark/)",
    )
    parser.add_argument(
        "--errorbar",
        dest="errorbar",
        default="sd",
        choices=list(ERRORBAR_KINDS),
        help="Error bar shown on each aggregated cell (default: sd)",
    )
    return parser.parse_args(argv)


def discover_leaf_dirs(sim_root):
    """Every simulation leaf directory (one holding a 'concat' subdirectory), sorted.
    The 'null' category is excluded -- it has no Figure-3 column mapping and is not
    part of the benchmark."""
    leaf_dirs = sorted(
        {p.parent for p in sim_root.rglob("concat") if p.is_dir()},
        key=lambda p: p.relative_to(sim_root).as_posix(),
    )
    return [d for d in leaf_dirs if d.relative_to(sim_root).parts[0] != "null"]


def get_expected_sim_output_dir(sim_path, leaf_dir, dist_type=DEFAULT_DIST_TYPE,
                                window_size=DEFAULT_WINDOW_SIZE, step_size=DEFAULT_STEP_SIZE):
    """
    Where caster/phlag output for ``leaf_dir`` lands under one pinned
    (dist_type, window, step) configuration.

    ``sim_path`` only feeds ``get_simulation_categories`` -- it may be the source
    FASTA or the leaf directory itself, and need not exist; it must, however, be
    a FULL path rather than a bare node name, because node names such as 'N228'
    live under four different category directories at once.
    """
    from .utils import get_data_dir, get_phlag_output_base, get_simulation_categories, get_short_sim_name
    from .caster import format_val

    phlag_base = get_phlag_output_base(get_data_dir())
    window_str = format_val(window_size)
    step_str = format_val(step_size)
    cats = get_simulation_categories(str(sim_path))
    short_sim = get_short_sim_name(leaf_dir.name)

    base = phlag_base / dist_type / f"w{window_str}_s{step_str}"
    if cats:
        return base / cats[0] / cats[1] / short_sim
    return base / short_sim


def get_expected_scores_path(fasta_path, leaf_dir, dist_type=DEFAULT_DIST_TYPE,
                             window_size=DEFAULT_WINDOW_SIZE, step_size=DEFAULT_STEP_SIZE):
    from .utils import clean_locus_name

    sim_output_dir = get_expected_sim_output_dir(
        fasta_path, leaf_dir, dist_type=dist_type,
        window_size=window_size, step_size=step_size,
    )
    pattern_stem = clean_locus_name(fasta_path.stem)
    return sim_output_dir / pattern_stem / "caster" / "scores.tsv"


def get_expected_report_path(fasta_path, leaf_dir, dist_type=DEFAULT_DIST_TYPE,
                             window_size=DEFAULT_WINDOW_SIZE, step_size=DEFAULT_STEP_SIZE):
    from .utils import clean_locus_name

    sim_output_dir = get_expected_sim_output_dir(
        fasta_path, leaf_dir, dist_type=dist_type,
        window_size=window_size, step_size=step_size,
    )
    pattern_stem = clean_locus_name(fasta_path.stem)
    return sim_output_dir / pattern_stem / "phlag" / "report.tsv"


def run_step(cmd, label):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    return True, result.stdout


def run_all(args, sim_root):
    """
    Runs caster + phlag over every complete simulation leaf.

    Caster and phlag are checked and skipped independently: if
    <...>/caster/scores.tsv already exists, caster is not re-run for that leaf
    even if phlag still needs to run (and vice versa) -- so a leaf that's half
    done never redoes the half that's already there. --rerun forces both
    stages regardless of what's already on disk.
    """
    from .utils import check_simulation_complete

    leaf_dirs = discover_leaf_dirs(sim_root)

    processed = 0
    skipped_incomplete = 0
    skipped_present = 0
    failures = []

    for leaf_dir in leaf_dirs:
        rel_path = leaf_dir.relative_to(sim_root).as_posix()

        is_complete, reason, fasta_path, mapping_path = check_simulation_complete(
            leaf_dir, concat_filename=DEFAULT_CONCAT_FILENAME
        )
        if not is_complete:
            print(f"[skip] {rel_path}: incomplete ({reason})")
            skipped_incomplete += 1
            continue

        scores_path = get_expected_scores_path(fasta_path, leaf_dir, dist_type=args.dist_type)
        report_path = get_expected_report_path(fasta_path, leaf_dir, dist_type=args.dist_type)
        caster_done = scores_path.exists() and not args.rerun
        phlag_done = report_path.exists() and not args.rerun

        if caster_done and phlag_done:
            print(f"[skip] {rel_path}: already present at {report_path.parent}")
            skipped_present += 1
            continue

        print(f"[run] {rel_path}")

        if caster_done:
            print(f"[skip caster] {rel_path}: scores already at {scores_path}")
        else:
            caster_ok, caster_out = run_step(
                [sys.executable, "-m", "phlag.caster", str(fasta_path), "-d", args.dist_type],
                "caster",
            )
            if not caster_ok:
                msg = f"{rel_path}: caster failed\n{caster_out}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue
            if not scores_path.exists():
                msg = f"{rel_path}: caster reported success but scores file not found at {scores_path}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue

        if phlag_done:
            print(f"[skip phlag] {rel_path}: report already at {report_path}")
        else:
            phlag_ok, phlag_out = run_step(
                [sys.executable, "-m", "phlag.phlag", str(scores_path), "-d", args.dist_type],
                "phlag",
            )
            if not phlag_ok:
                msg = f"{rel_path}: phlag failed\n{phlag_out}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue

        print(f"[done] {rel_path}")
        processed += 1

    print(
        f"\nBenchmark complete: {processed} processed, "
        f"{skipped_incomplete} skipped (incomplete), "
        f"{skipped_present} skipped (already present), "
        f"{len(failures)} failed."
    )

    if failures:
        print(f"{RED}\nFailures:{RESET}")
        for failure in failures:
            print(f"{RED}- {failure}{RESET}")

    return processed, skipped_incomplete, skipped_present, failures


def summarize(args, sim_root):
    """Aggregates every finished run into the Figure-3 panels and writes them out."""
    stats = BenchmarkStats(
        sim_root=sim_root,
        dist_type=args.dist_type,
        errorbar=args.errorbar,
    )
    stats.collect()
    figure_data = stats.aggregate()
    stats.print_diagnostics(figure_data)

    out_dir = pathlib.Path(args.stats_out) if args.stats_out else stats.default_stats_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_path, bins_path = stats.write_tables(out_dir, figure_data)
    print(f"\nWrote per-run table:  {runs_path}")
    print(f"Wrote per-cell table: {bins_path}")

    if figure_data.n_runs_panel_a == 0 and figure_data.n_runs_panel_b == 0:
        print(f"{RED}No runs qualified for either panel -- skipping figures "
              f"(see the exclusions above).{RESET}")
        return figure_data

    plotter = BenchmarkFigurePlotter(figure_data, out_dir)
    for figure_path in plotter.render():
        print(f"Wrote figure:         {figure_path}")

    return figure_data


def main(argv=None):
    args = parse_arguments(argv)

    from .utils import get_data_dir

    sim_root = get_data_dir() / "simulations"
    if not sim_root.exists():
        sys.exit(f"Error: simulations directory not found at '{sim_root}'")

    run_all(args, sim_root)
    summarize(args, sim_root)


if __name__ == "__main__":
    main()
