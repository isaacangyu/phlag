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

import ast
import re
import sys
import time
import math
import shutil
import pathlib
import argparse
import subprocess

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

RED = "\033[91m"
RESET = "\033[0m"

DEFAULT_DIST_TYPE = "gaussian"
DEFAULT_WINDOW_SIZE = 50000
DEFAULT_STEP_SIZE = 1000

TOPOLOGY_NAMES = ["ABBA", "BABA", "AABB"]
RELERR_STATES = ["Null", "Alt"]
RELERR_STATISTICS = ["mean", "std"]

DEFAULT_CONCAT_PATTERNS = ["37-62", "40-60", "42-57", "45-55", "47-52", "49-51"]



COL_RECOMBINATION_INCREASE = "recombination increase"
COL_RECOMBINATION_SUPPRESSION = "recombination suppression"
COL_POPSIZE_INCREASE = "population size increase"
COL_POPSIZE_DECREASE = "population size decrease"
COL_ADMIXTURE = "admixture"

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

BRANCH_LENGTH_BIN_EDGES = [
    ("(0,0.1]", 0.0, 0.1),
    ("(0.1,1]", 0.1, 1.0),
    ("(1,5]", 1.0, 5.0),
]
BRANCH_LENGTH_BINS = [label for label, _, _ in BRANCH_LENGTH_BIN_EDGES]

ADMIXTURE_BINS = ["low", "high"]

FRACTION_BIN_EDGES = [
    ("(0,0.1]", 0.0, 0.1),
    ("(0.1,0.25]", 0.1, 0.25),
]
FRACTION_BINS = [label for label, _, _ in FRACTION_BIN_EDGES]

BIN_EDGE_TOL = 0.002

ERRORBAR_KINDS = ("sd", "sem", "ci95")



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
_RE_RELERR_ROW = re.compile(
    r'^(\w+)\t(Null|Alt)\t(mean|std)\t([-\d.eE+]+|nan)\t([-\d.eE+]+|nan)\t([-\d.eE+]+|nan)\s*$',
    re.IGNORECASE,
)
_RE_EM_DIVERGENCE = re.compile(
    r'^EM divergence \(Bhattacharyya\):\s*([-\d.eE+]+)\s*$'
)
_RE_GT_EM_DIVERGENCE = re.compile(
    r'^Ground truth EM divergence \(Bhattacharyya\):\s*([-\d.eE+]+)\s*$'
)
_RE_TRANSITION_MATRIX = re.compile(
    r'^Final transition matrix \(after EM\):\s*'
    r'\[\[([-\d.eE+]+),\s*([-\d.eE+]+)\],\s*\[([-\d.eE+]+),\s*([-\d.eE+]+)\]\]\s*$'
)
_RE_CASTER_MTIME = re.compile(r'^Caster source mtime:\s*([\d.eE+]+)\s*$')
_RE_PHLAG_MTIME = re.compile(r'^Phlag source mtime:\s*([\d.eE+]+)\s*$')
_RE_BIC = re.compile(
    r'^BIC:\s*([-\d.eE+]+)\s*\(log-likelihood=([-\d.eE+]+),\s*'
    r'k=(\d+)\s*trainable params,\s*n=(\d+)\s*windows\)\s*$'
)
# Older reports' "--- Bookkeeping ---"/"--- Fitted parameters ..." section --
# no longer written by phlag.py, but archived reports still carry it, and
# it's the only way to recompute em_bd exactly for those (see parse_report).
_RE_FITTED_MEAN = re.compile(r'^(Null|Alt) fitted mean:\s*(\[.+\])\s*$')
_RE_FITTED_COV = re.compile(r'^(Null|Alt) fitted covariance:\s*(\[\[.+\]\])\s*$')


def gaussian_hellinger_distance_1d(mu1, var1, mu2, var2, eps=1e-6):
    """
    1D Gaussian Hellinger distance -- var1/var2 are variances (std**2),
    bounded [0, 1] (0 = identical distributions, 1 = fully separated).
    Algebraically hmm.gaussian_hellinger_distance reduced to one dimension,
    kept as a plain-math reimplementation here so this module doesn't need a
    jax dependency just for this.
    """
    sigma_avg = 0.5 * (var1 + var2) + eps
    log_coef = 0.25 * math.log(var1 + eps) + 0.25 * math.log(var2 + eps) - 0.5 * math.log(sigma_avg)
    quad = (mu1 - mu2) ** 2 / sigma_avg
    bc = math.exp(log_coef - 0.125 * quad)
    return math.sqrt(max(1.0 - bc, 0.0))


def gaussian_hellinger_distance_nd(mu1, Sigma1, mu2, Sigma2, eps=1e-6):
    """
    General-N Gaussian Hellinger distance (numpy), matching
    hmm.gaussian_hellinger_distance -- used to recompute em_bd for archived
    reports whose "EM divergence" header line predates the Hellinger-distance
    convention, straight from the joint fitted mean/covariance bookkeeping
    block those older reports still carry.
    """
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    d = mu1.shape[0]
    Sigma_avg = 0.5 * (Sigma1 + Sigma2) + eps * np.eye(d)
    diff = mu1 - mu2
    _, logdet1 = np.linalg.slogdet(Sigma1 + eps * np.eye(d))
    _, logdet2 = np.linalg.slogdet(Sigma2 + eps * np.eye(d))
    _, logdet_avg = np.linalg.slogdet(Sigma_avg)
    log_coef = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg
    quad = diff @ np.linalg.solve(Sigma_avg, diff)
    bc = np.exp(log_coef - 0.125 * quad)
    return float(np.sqrt(max(1.0 - bc, 0.0)))


def hellinger_distance_from_legacy_log_bc(distance_value):
    """
    Converts an "EM divergence (Bhattacharyya):" header value written under
    the original -log(BC) convention (before this repo switched to a
    Bhattacharyya coefficient, then to a Hellinger distance) into a Hellinger
    distance: BC = exp(-distance_value), Hellinger = sqrt(1 - BC).

    NOT safe to apply automatically/generally -- a value in [0, 1] is
    genuinely ambiguous between this legacy convention and a real Hellinger
    distance already written by current code (they overlap in range), so
    this is only correct when the report's provenance (e.g. source mtimes,
    presence/absence of other header lines added or removed at known points)
    independently confirms it predates the coefficient/Hellinger switch.
    Only call this with that external confirmation in hand.
    """
    bc = math.exp(-distance_value)
    return math.sqrt(max(1.0 - bc, 0.0))


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
        "rel_err": {},
        "ground_truth_stats": {},
        "fitted_null_mean": None, "fitted_null_cov": None,
        "fitted_alt_mean": None, "fitted_alt_cov": None,
        "em_bd": None,
        "em_gt_bd": None,
        "transition_null_to_null": None, "transition_null_to_alt": None,
        "transition_alt_to_null": None, "transition_alt_to_alt": None,
        "caster_source_mtime": None, "phlag_source_mtime": None,
        "bic": None, "log_likelihood": None, "n_trainable_params": None,
    }

    with open(report_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

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

            m = _RE_RELERR_ROW.match(line)
            if m:
                topology, state, statistic, _fitted_str, gt_str, relerr_str = m.groups()
                try:
                    value = float(relerr_str)
                except ValueError:
                    value = None
                if value is not None and not math.isnan(value):
                    parsed["rel_err"][(topology, state.capitalize(), statistic.lower())] = value
                try:
                    gt_value = float(gt_str)
                except ValueError:
                    gt_value = None
                if gt_value is not None and not math.isnan(gt_value):
                    parsed["ground_truth_stats"][(topology, state.capitalize(), statistic.lower())] = gt_value
                continue

            m = _RE_EM_DIVERGENCE.match(line)
            if m:
                parsed["em_bd"] = float(m.group(1))
                continue

            m = _RE_GT_EM_DIVERGENCE.match(line)
            if m:
                parsed["em_gt_bd"] = float(m.group(1))
                continue

            m = _RE_FITTED_MEAN.match(line)
            if m:
                state, vec_str = m.groups()
                try:
                    vec = ast.literal_eval(vec_str)
                except (ValueError, SyntaxError):
                    vec = None
                parsed[f"fitted_{state.lower()}_mean"] = vec
                continue

            m = _RE_FITTED_COV.match(line)
            if m:
                state, mat_str = m.groups()
                try:
                    mat = ast.literal_eval(mat_str)
                except (ValueError, SyntaxError):
                    mat = None
                parsed[f"fitted_{state.lower()}_cov"] = mat
                continue

            m = _RE_TRANSITION_MATRIX.match(line)
            if m:
                parsed["transition_null_to_null"] = float(m.group(1))
                parsed["transition_null_to_alt"] = float(m.group(2))
                parsed["transition_alt_to_null"] = float(m.group(3))
                parsed["transition_alt_to_alt"] = float(m.group(4))
                continue

            m = _RE_CASTER_MTIME.match(line)
            if m:
                parsed["caster_source_mtime"] = float(m.group(1))
                continue

            m = _RE_PHLAG_MTIME.match(line)
            if m:
                parsed["phlag_source_mtime"] = float(m.group(1))
                continue

            m = _RE_BIC.match(line)
            if m:
                parsed["bic"] = float(m.group(1))
                parsed["log_likelihood"] = float(m.group(2))
                parsed["n_trainable_params"] = int(m.group(3))
                continue

    if parsed["n_windows"] is None and parsed["n_path_entries"] is not None:
        parsed["n_windows"] = parsed["n_path_entries"]

    # em_gt_bd: always recompute from the per-topology GroundTruth mean/std
    # this table has always carried, rather than trusting a pre-existing
    # "Ground truth EM divergence" header line -- that line's number could be
    # from a report written under an earlier convention (Bhattacharyya
    # coefficient, or -log(BC) distance, before the current Hellinger
    # distance), and there's no reliable way to tell which just by looking at
    # the number. Recomputing from the raw per-topology stats sidesteps that
    # ambiguity entirely and works identically for every report era.
    if parsed["ground_truth_stats"]:
        topologies = sorted({topo for (topo, _state, _stat) in parsed["ground_truth_stats"]})
        distances = []
        for topo in topologies:
            try:
                mu_null = parsed["ground_truth_stats"][(topo, "Null", "mean")]
                std_null = parsed["ground_truth_stats"][(topo, "Null", "std")]
                mu_alt = parsed["ground_truth_stats"][(topo, "Alt", "mean")]
                std_alt = parsed["ground_truth_stats"][(topo, "Alt", "std")]
            except KeyError:
                continue
            distances.append(gaussian_hellinger_distance_1d(mu_null, std_null ** 2, mu_alt, std_alt ** 2))
        if distances:
            parsed["em_gt_bd"] = sum(distances) / len(distances)

    # em_bd: same reasoning -- if this report still carries the "Null/Alt
    # fitted mean/covariance" bookkeeping block, recompute straight from it
    # instead of trusting the "EM divergence" header line's possibly-stale-
    # convention number. Reports without that block (not just "written after
    # it was removed" -- some older reports, predating when it was ever
    # added, also lack it) fall through to whatever _RE_EM_DIVERGENCE parsed
    # above.
    if (
        parsed["fitted_null_mean"] is not None and parsed["fitted_null_cov"] is not None
        and parsed["fitted_alt_mean"] is not None and parsed["fitted_alt_cov"] is not None
    ):
        parsed["em_bd"] = gaussian_hellinger_distance_nd(
            parsed["fitted_null_mean"], parsed["fitted_null_cov"],
            parsed["fitted_alt_mean"], parsed["fitted_alt_cov"],
        )

    # A Hellinger distance is mathematically bounded to [0, 1] -- a header
    # line outside that range (with slack for float error) can only be a
    # relic of an earlier "EM divergence" convention (this report predates
    # the Hellinger-distance one AND lacks the bookkeeping block needed to
    # recompute it), not a valid value. Drop it rather than propagate a
    # number that would blow up downstream error-bar/axis math.
    if parsed["em_bd"] is not None and not (-1e-6 <= parsed["em_bd"] <= 1 + 1e-6):
        parsed["em_bd"] = None

    return parsed


def nominal_anomaly_fraction(pattern_str):
    """
    Fallback anomaly fraction for reports written before the 'Anomaly fraction:'
    header existed: the NOMINAL portion of the genome the locus pattern marks as
    anomalous, e.g. '37-62' -> 0.25, '47-52' -> 0.05, '49-51' -> 0.02.

    This is a design-level number read off the pattern alone; the measured
    fraction in the report header is the authoritative one because it also
    accounts for where the window grid actually falls. Runs that fall back to
    this are tagged 'nominal' in runs.tsv.
    """
    from phlag.utils import parse_pattern_string

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
    in_panel_relerr: bool = False
    in_panel_em_divergence: bool = False
    exclusion: str = ""
    rel_err: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    mean_relerr_agg: Optional[float] = None
    covar_relerr_agg: Optional[float] = None
    em_bd: Optional[float] = None
    em_gt_bd: Optional[float] = None
    transition_null_to_null: Optional[float] = None
    transition_null_to_alt: Optional[float] = None
    transition_alt_to_null: Optional[float] = None
    transition_alt_to_alt: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None
    n_trainable_params: Optional[int] = None


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
    One point of panel B: the macro-averaged TPR/FPR over the runs in this
    (column, fraction_bin, x_bin) cell -- the same x_bin (branch-length bin,
    or admixture divergence-time bin) panel A groups by, so panel B draws one
    connected line of points per (column, fraction_bin) subplot instead of a
    single aggregate point.

    Macro (mean of each run's own rate), not micro (a rate recomputed from pooled
    confusion counts), so that panel B estimates the same quantity panel A does --
    performance on a randomly drawn replicate run rather than on a randomly drawn
    genomic window.
    """
    column: str
    fraction_bin: str
    x_bin: str
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
class RelErrCell:
    """
    One grouped-bar cell of the relerr figure: two bars, the mean |relative
    error| (%) of the EM-fitted mean and of the EM-fitted covariance (std),
    each vs. the ground-truth-split empirical fit, over the runs in this cell.

    Same (column, fraction_bin, x_bin) grouping panel A uses, so relerr.png
    and f1.png are directly comparable bucket for bucket.
    """
    column: str
    fraction_bin: str
    x_bin: str
    n_runs: int
    mean_relerr: float
    mean_relerr_sd: float
    mean_relerr_err_minus: float
    mean_relerr_err_plus: float
    covar_relerr: float
    covar_relerr_sd: float
    covar_relerr_err_minus: float
    covar_relerr_err_plus: float
    run_ids: List[str] = field(default_factory=list)


@dataclass
class EmDivergenceCell:
    """
    One grouped-bar cell of the em_divergence figure: mean Hellinger distance
    between the fitted Null/Alt states' Gaussians, over the runs in this
    cell.

    Same (column, fraction_bin, x_bin) grouping panel A/relerr use, so
    em_divergence.png lines up bucket for bucket with f1.png and relerr.png.
    """
    column: str
    fraction_bin: str
    x_bin: str
    n_runs: int
    mean_distance: float
    sd_distance: float
    err_minus: float
    err_plus: float
    run_ids: List[str] = field(default_factory=list)
    n_gt_runs: int = 0
    gt_mean_distance: float = 0.0
    gt_sd_distance: float = 0.0
    gt_err_minus: float = 0.0
    gt_err_plus: float = 0.0


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
    panel_b: Dict[Tuple[str, str, str], PanelBCell]
    panel_relerr: Dict[Tuple[str, str, str], RelErrCell]
    panel_em_divergence: Dict[Tuple[str, str, str], EmDivergenceCell]
    errorbar: str
    dist_type: str
    window_size: int
    step_size: int
    n_reports_found: int = 0
    n_runs_panel_a: int = 0
    n_runs_panel_b: int = 0
    n_runs_relerr: int = 0
    n_runs_em_divergence: int = 0
    n_runs_excluded: int = 0
    exclusions: Dict[str, int] = field(default_factory=dict)



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
        from phlag.utils import get_data_dir

        if errorbar not in ERRORBAR_KINDS:
            raise ValueError(f"Unknown errorbar kind '{errorbar}' (expected one of {ERRORBAR_KINDS})")

        self.sim_root = pathlib.Path(sim_root) if sim_root is not None else get_data_dir() / "simulations"
        self.dist_type = dist_type
        self.window_size = window_size
        self.step_size = step_size
        self.errorbar = errorbar

        self.runs = []
        self.notes = []
        self.n_leaf_dirs = 0
        self.n_leaf_dirs_uncategorized = 0
        self.n_leaf_dirs_without_output = 0


    def default_stats_dir(self):
        """<phlag_base>/<dist_type>/w<W>_s<S>/benchmark/ -- the container for numbered per-run subfolders."""
        from phlag.utils import get_data_dir, get_phlag_output_base
        from phlag.caster import format_val

        window_str = format_val(self.window_size)
        step_str = format_val(self.step_size)
        return (
            get_phlag_output_base(get_data_dir())
            / self.dist_type
            / f"w{window_str}_s{step_str}"
            / "benchmark"
        )

    def next_run_dir(self):
        """
        <default_stats_dir()>/<N>/, where N is one past the highest existing
        numbered run subfolder (1 if none exist yet). Each summarize() call
        gets its own folder instead of overwriting the previous run's tables
        and figures, so old runs stay around for comparison.
        """
        root = self.default_stats_dir()
        existing_ids = [
            int(child.name) for child in (root.iterdir() if root.exists() else [])
            if child.is_dir() and child.name.isdigit()
        ]
        next_id = max(existing_ids, default=0) + 1
        return root / str(next_id)

    def latest_run_dir(self):
        """
        <default_stats_dir()>/<N>/, where N is the highest existing numbered
        run subfolder, or None if none exist yet. Used to reuse a prior run's
        output dir (modify tables/figures in place, append to report.txt)
        when a `benchmark` invocation processed nothing new.
        """
        root = self.default_stats_dir()
        existing_ids = [
            int(child.name) for child in (root.iterdir() if root.exists() else [])
            if child.is_dir() and child.name.isdigit()
        ]
        if not existing_ids:
            return None
        return root / str(max(existing_ids))


    def collect(self):
        """
        Populates ``self.runs`` with one RunRecord per report.tsv found under the
        expected output directory of a categorized source simulation, restricted to
        the current DEFAULT_CONCAT_PATTERNS sweep. This always re-walks the output
        tree on disk, so it picks up every finished default-pattern run that exists
        there -- not only ones produced by a `run_all()` call earlier in this process.
        """
        from phlag.utils import get_simulation_categories

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
            report_paths = sorted(
                p for p in sim_output_dir.glob("*/phlag/report.tsv")
                if p.parent.parent.name in DEFAULT_CONCAT_PATTERNS
            )
            if not report_paths:
                self.n_leaf_dirs_without_output += 1
                self.notes.append(
                    f"[skip] {rel_leaf}: no default-pattern report.tsv under {sim_output_dir} "
                    f"(patterns: {', '.join(DEFAULT_CONCAT_PATTERNS)})"
                )
                continue

            for report_path in report_paths:
                self.runs.append(
                    self._build_record(report_path, leaf_dir, rel_leaf, category, subcategory, column)
                )

        return self.runs

    def _build_record(self, report_path, leaf_dir, rel_leaf, category, subcategory, column):
        """Parses one report.tsv and bins it. Never raises on a malformed report."""
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

        record.rel_err = parsed["rel_err"]
        if record.rel_err:
            mean_vals = [abs(v) for (_, _, stat), v in record.rel_err.items() if stat == "mean"]
            std_vals = [abs(v) for (_, _, stat), v in record.rel_err.items() if stat == "std"]
            record.mean_relerr_agg = sum(mean_vals) / len(mean_vals) if mean_vals else None
            record.covar_relerr_agg = sum(std_vals) / len(std_vals) if std_vals else None

        record.em_bd = parsed["em_bd"]
        record.em_gt_bd = parsed["em_gt_bd"]
        record.bic = parsed["bic"]
        record.log_likelihood = parsed["log_likelihood"]
        record.n_trainable_params = parsed["n_trainable_params"]

        record.transition_null_to_null = parsed["transition_null_to_null"]
        record.transition_null_to_alt = parsed["transition_null_to_alt"]
        record.transition_alt_to_null = parsed["transition_alt_to_null"]
        record.transition_alt_to_alt = parsed["transition_alt_to_alt"]

        if parsed["anomaly_fraction"] is not None:
            record.anomaly_fraction = parsed["anomaly_fraction"]
            record.anomaly_fraction_source = "report"
        else:
            record.anomaly_fraction = nominal_anomaly_fraction(pattern)
            record.anomaly_fraction_source = "nominal" if record.anomaly_fraction is not None else "missing"
        record.fraction_bin = assign_bin(record.anomaly_fraction, FRACTION_BIN_EDGES)

        if column == COL_ADMIXTURE:
            from phlag.utils import get_admixture_divergence_time, ADMIXTURE_DIVERGENCE_THRESHOLD_MYR
            record.x_variable = "divergence time (Myr)"
            record.x_value = get_admixture_divergence_time(leaf_dir.name)
            record.x_bin = subcategory if subcategory in ADMIXTURE_BINS else None
            if record.x_bin is None and record.x_value is not None:
                record.x_bin = (
                    "low" if record.x_value < ADMIXTURE_DIVERGENCE_THRESHOLD_MYR else "high"
                )
        else:
            record.x_variable = "branch length (CU)"
            record.x_value = record.branch_length
            record.x_bin = assign_bin(record.branch_length, BRANCH_LENGTH_BIN_EDGES)

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

        record.in_panel_b = True

        if record.f1 is None:
            record.exclusion = "no F1 in report (panel B only)"
        elif record.x_bin is None:
            record.exclusion = f"x value {record.x_value} outside every x bin (panel B only)"
        else:
            record.in_panel_a = True

        if record.x_bin is not None and record.mean_relerr_agg is not None and record.covar_relerr_agg is not None:
            record.in_panel_relerr = True
        elif not record.exclusion:
            record.exclusion = (
                "no rel-err table in report (relerr panel only)"
                if record.x_bin is not None
                else f"x value {record.x_value} outside every x bin (relerr panel only)"
            )

        if record.x_bin is not None and record.em_bd is not None:
            record.in_panel_em_divergence = True
        elif not record.exclusion:
            record.exclusion = (
                "no EM divergence line in report (em_divergence panel only)"
                if record.x_bin is not None
                else f"x value {record.x_value} outside every x bin (em_divergence panel only)"
            )

        return record


    def aggregate(self):
        """Reduces ``self.runs`` to a Figure3Data. Call ``collect()`` first."""
        panel_a_groups = defaultdict(list)
        panel_b_groups = defaultdict(list)
        panel_relerr_groups = defaultdict(list)
        panel_em_divergence_groups = defaultdict(list)
        exclusions = defaultdict(int)

        for record in self.runs:
            if record.exclusion:
                exclusions[record.exclusion] += 1
            if record.in_panel_a:
                panel_a_groups[(record.column, record.fraction_bin, record.x_bin)].append(record)
            if record.in_panel_b and record.x_bin is not None:
                panel_b_groups[(record.column, record.fraction_bin, record.x_bin)].append(record)
            if record.in_panel_relerr:
                panel_relerr_groups[(record.column, record.fraction_bin, record.x_bin)].append(record)
            if record.in_panel_em_divergence:
                panel_em_divergence_groups[(record.column, record.fraction_bin, record.x_bin)].append(record)

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
            column, fraction_bin, x_bin = key
            tpr_mean, tpr_sd, tpr_minus, tpr_plus = summarize_values(
                [r.tpr for r in records], self.errorbar
            )
            fpr_mean, fpr_sd, fpr_minus, fpr_plus = summarize_values(
                [r.fpr for r in records], self.errorbar
            )
            panel_b[key] = PanelBCell(
                column=column, fraction_bin=fraction_bin, x_bin=x_bin, n_runs=len(records),
                mean_tpr=tpr_mean, mean_fpr=fpr_mean,
                sd_tpr=tpr_sd, sd_fpr=fpr_sd,
                tpr_err_minus=tpr_minus, tpr_err_plus=tpr_plus,
                fpr_err_minus=fpr_minus, fpr_err_plus=fpr_plus,
                run_ids=sorted(r.run_id for r in records),
            )

        panel_relerr = {}
        for key, records in panel_relerr_groups.items():
            column, fraction_bin, x_bin = key
            mean_relerr, mean_relerr_sd, mean_minus, mean_plus = summarize_values(
                [r.mean_relerr_agg for r in records], self.errorbar, low=0.0, high=math.inf
            )
            covar_relerr, covar_relerr_sd, covar_minus, covar_plus = summarize_values(
                [r.covar_relerr_agg for r in records], self.errorbar, low=0.0, high=math.inf
            )
            panel_relerr[key] = RelErrCell(
                column=column, fraction_bin=fraction_bin, x_bin=x_bin, n_runs=len(records),
                mean_relerr=mean_relerr, mean_relerr_sd=mean_relerr_sd,
                mean_relerr_err_minus=mean_minus, mean_relerr_err_plus=mean_plus,
                covar_relerr=covar_relerr, covar_relerr_sd=covar_relerr_sd,
                covar_relerr_err_minus=covar_minus, covar_relerr_err_plus=covar_plus,
                run_ids=sorted(r.run_id for r in records),
            )

        panel_em_divergence = {}
        for key, records in panel_em_divergence_groups.items():
            column, fraction_bin, x_bin = key
            mean_dist, sd_dist, dist_minus, dist_plus = summarize_values(
                [r.em_bd for r in records], self.errorbar, low=0.0, high=1.0
            )
            gt_values = [r.em_gt_bd for r in records if r.em_gt_bd is not None]
            gt_mean_dist, gt_sd_dist, gt_dist_minus, gt_dist_plus = summarize_values(
                gt_values, self.errorbar, low=0.0, high=1.0
            )
            panel_em_divergence[key] = EmDivergenceCell(
                column=column, fraction_bin=fraction_bin, x_bin=x_bin, n_runs=len(records),
                mean_distance=mean_dist, sd_distance=sd_dist,
                err_minus=dist_minus, err_plus=dist_plus,
                run_ids=sorted(r.run_id for r in records),
                n_gt_runs=len(gt_values), gt_mean_distance=gt_mean_dist, gt_sd_distance=gt_sd_dist,
                gt_err_minus=gt_dist_minus, gt_err_plus=gt_dist_plus,
            )

        x_bins_by_column = {}
        x_axis_label_by_column = {}
        for column in FIGURE_COLUMNS:
            if column == COL_ADMIXTURE:
                from phlag.utils import ADMIXTURE_DIVERGENCE_THRESHOLD_MYR
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
            panel_relerr=panel_relerr,
            panel_em_divergence=panel_em_divergence,
            errorbar=self.errorbar,
            dist_type=self.dist_type,
            window_size=self.window_size,
            step_size=self.step_size,
            n_reports_found=len(self.runs),
            n_runs_panel_a=sum(1 for r in self.runs if r.in_panel_a),
            n_runs_panel_b=sum(1 for r in self.runs if r.in_panel_b),
            n_runs_relerr=sum(1 for r in self.runs if r.in_panel_relerr),
            n_runs_em_divergence=sum(1 for r in self.runs if r.in_panel_em_divergence),
            n_runs_excluded=sum(exclusions.values()),
            exclusions=dict(exclusions),
        )


    RELERR_RUN_COLUMNS = [
        f"relerr_{topo}_{state}_{stat}"
        for topo in TOPOLOGY_NAMES
        for state in RELERR_STATES
        for stat in RELERR_STATISTICS
    ]

    RUN_COLUMNS = [
        "run_id", "source_leaf", "category", "subcategory", "column", "pattern",
        "anomaly_fraction", "anomaly_fraction_source", "fraction_bin",
        "x_variable", "x_value", "x_bin", "branch_length_cu",
        "clade_name", "clade_number",
        "tpr", "fpr", "f1", "accuracy", "tp", "fp", "fn", "tn",
        "n_windows", "label_flipped",
        "in_panel_a", "in_panel_b", "in_panel_relerr", "in_panel_em_divergence",
        "mean_relerr_agg", "covar_relerr_agg",
        "em_bd", "em_gt_bd",
        "transition_null_to_null", "transition_null_to_alt",
        "transition_alt_to_null", "transition_alt_to_alt",
        "bic", "log_likelihood", "n_trainable_params",
    ] + RELERR_RUN_COLUMNS + [
        "exclusion", "report_path",
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
        Writes runs.tsv (one row per discovered report, excluded ones
        included with their reason) and bins.tsv (one row per
        panel/cell/metric, carrying the source run ids) next to the PNGs.
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if figure_data is None:
            figure_data = self.aggregate()

        runs_path = out_dir / "runs.tsv"
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
                    "in_panel_relerr": record.in_panel_relerr,
                    "in_panel_em_divergence": record.in_panel_em_divergence,
                    "mean_relerr_agg": record.mean_relerr_agg,
                    "covar_relerr_agg": record.covar_relerr_agg,
                    "em_bd": record.em_bd,
                    "em_gt_bd": record.em_gt_bd,
                    "transition_null_to_null": record.transition_null_to_null,
                    "transition_null_to_alt": record.transition_null_to_alt,
                    "transition_alt_to_null": record.transition_alt_to_null,
                    "transition_alt_to_alt": record.transition_alt_to_alt,
                    "bic": record.bic,
                    "log_likelihood": record.log_likelihood,
                    "n_trainable_params": record.n_trainable_params,
                    "exclusion": record.exclusion,
                    "report_path": record.report_path,
                }
                for topo in TOPOLOGY_NAMES:
                    for state in RELERR_STATES:
                        for stat in RELERR_STATISTICS:
                            row[f"relerr_{topo}_{state}_{stat}"] = record.rel_err.get((topo, state, stat))
                handle.write("\t".join(self._fmt(row[c]) for c in self.RUN_COLUMNS) + "\n")

        bins_path = out_dir / "bins.tsv"
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
                emit("B", cell.column, cell.fraction_bin, cell.x_bin, "tpr", cell.n_runs,
                     cell.mean_tpr, cell.sd_tpr, cell.tpr_err_minus, cell.tpr_err_plus,
                     cell.run_ids)
                emit("B", cell.column, cell.fraction_bin, cell.x_bin, "fpr", cell.n_runs,
                     cell.mean_fpr, cell.sd_fpr, cell.fpr_err_minus, cell.fpr_err_plus,
                     cell.run_ids)

            for key in sorted(figure_data.panel_relerr.keys()):
                cell = figure_data.panel_relerr[key]
                emit("RELERR", cell.column, cell.fraction_bin, cell.x_bin, "mean_relerr", cell.n_runs,
                     cell.mean_relerr, cell.mean_relerr_sd, cell.mean_relerr_err_minus, cell.mean_relerr_err_plus,
                     cell.run_ids)
                emit("RELERR", cell.column, cell.fraction_bin, cell.x_bin, "covar_relerr", cell.n_runs,
                     cell.covar_relerr, cell.covar_relerr_sd, cell.covar_relerr_err_minus, cell.covar_relerr_err_plus,
                     cell.run_ids)

            for key in sorted(figure_data.panel_em_divergence.keys()):
                cell = figure_data.panel_em_divergence[key]
                emit("EM_DIVERGENCE", cell.column, cell.fraction_bin, cell.x_bin, "bhattacharyya_distance",
                     cell.n_runs, cell.mean_distance, cell.sd_distance,
                     cell.err_minus, cell.err_plus, cell.run_ids)

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
              f"panel B runs: {figure_data.n_runs_panel_b}  |  "
              f"relerr runs: {figure_data.n_runs_relerr}  |  "
              f"em_divergence runs: {figure_data.n_runs_em_divergence}")

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
                (column, x_bin, figure_data.panel_b[(column, fraction_bin, x_bin)])
                for column in figure_data.columns
                for x_bin in figure_data.x_bins_by_column[column]
                if (column, fraction_bin, x_bin) in figure_data.panel_b
            ]
            if not row_cells:
                print(f"  fraction {fraction_bin}: EMPTY (no runs fall in this row)")
                continue
            print(f"  fraction {fraction_bin}:")
            for column, x_bin, cell in row_cells:
                print(f"    {column:28s} x={x_bin:10s} n={cell.n_runs:3d}  "
                      f"TPR={cell.mean_tpr:.4f}  FPR={cell.mean_fpr:.4f}")

        print("\nRelerr cells (mean |rel.err(%)| of EM mean/covariance vs. ground truth):")
        for fraction_bin in figure_data.fraction_bins:
            row_cells = [
                (column, x_bin, figure_data.panel_relerr[(column, fraction_bin, x_bin)])
                for column in figure_data.columns
                for x_bin in figure_data.x_bins_by_column[column]
                if (column, fraction_bin, x_bin) in figure_data.panel_relerr
            ]
            if not row_cells:
                print(f"  fraction {fraction_bin}: EMPTY (no runs fall in this row)")
                continue
            print(f"  fraction {fraction_bin}:")
            for column, x_bin, cell in row_cells:
                print(f"    {column:28s} x={x_bin:10s} n={cell.n_runs:3d}  "
                      f"mean={cell.mean_relerr:.4f}  covar={cell.covar_relerr:.4f}")

        print("\nEm divergence cells (mean Bhattacharyya distance between fitted Null/Alt states):")
        for fraction_bin in figure_data.fraction_bins:
            row_cells = [
                (column, x_bin, figure_data.panel_em_divergence[(column, fraction_bin, x_bin)])
                for column in figure_data.columns
                for x_bin in figure_data.x_bins_by_column[column]
                if (column, fraction_bin, x_bin) in figure_data.panel_em_divergence
            ]
            if not row_cells:
                print(f"  fraction {fraction_bin}: EMPTY (no runs fall in this row)")
                continue
            print(f"  fraction {fraction_bin}:")
            for column, x_bin, cell in row_cells:
                print(f"    {column:28s} x={x_bin:10s} n={cell.n_runs:3d}  "
                      f"distance={cell.mean_distance:.4f} +/-{cell.sd_distance:.4f}")

        if self.notes:
            print("\nDiscovery notes:")
            for note in self.notes:
                print(f"  {note}")



_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"

_ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#1c5cab"]

_SERIES_PHLAG = "#2a78d6"

_SERIES_RELERR_MEAN = "#2B4C7E"
_SERIES_RELERR_COVAR = "#E05A47"
_SERIES_GT_DIVERGENCE = "#1B998B"

_BAR_WIDTH = 0.62
_MARKER_SIZE = 8.0
_N_GRIDLINES = 6


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

    def _bin_colors_for_column(self, column):
        """
        Assigns each of a column's x_bins (e.g. branch-length bins, or the
        admixture low/high split) a color from the ordinal ramp. Shared by
        panel A's bars and panel B's per-bin points so the same bin reads as
        the same color in both.
        """
        x_bins = self.data.x_bins_by_column[column]
        ramp = _ordinal_colors(len(x_bins))
        return {b: ramp[i] for i, b in enumerate(x_bins)}

    def _width_ratios(self):
        """
        Column widths proportional to each column's number of x_bins, so a
        column with fewer categories (e.g. admixture's low/high split vs. the
        4 branch-length bins elsewhere) renders narrower instead of matching
        width with empty space.
        """
        return [len(self.data.x_bins_by_column[c]) for c in self.data.columns]

    def _int_yaxis(self, ax):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:.0f}"))


    def render_panel_a(self, filename="f1.png"):
        data = self.data
        n_rows, n_cols = len(data.fraction_bins), len(data.columns)

        self._apply_theme()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
            squeeze=False,
            gridspec_kw={"width_ratios": self._width_ratios()},
        )

        for row_idx, fraction_bin in enumerate(data.fraction_bins):
            for col_idx, column in enumerate(data.columns):
                ax = axes[row_idx][col_idx]
                x_bins = data.x_bins_by_column[column]
                color_for_bin = self._bin_colors_for_column(column)

                self._style_axes(ax)
                ax.set_xlim(-0.7, len(x_bins) - 0.3)
                ax.set_ylim(0.0, 1.18)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xticks(list(range(len(x_bins))))
                ax.set_xticklabels(x_bins, rotation=30, ha="right", fontsize=7.5)

                drew_any = False
                for pos, x_bin in enumerate(x_bins):
                    cell = data.panel_a.get((column, fraction_bin, x_bin))
                    if cell is None or cell.n_runs == 0:
                        continue
                    drew_any = True
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
                if row_idx == n_rows - 1:
                    ax.set_xlabel(data.x_axis_label_by_column[column], fontsize=8)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    ax.set_ylabel(f"alt proportion {fraction_bin}", fontsize=8.5)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fig.suptitle(
            "Phlag anomaly detection: F1 by branch length and alternative sequence fraction",
            fontsize=12, color=_INK_PRIMARY, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path


    def render_panel_b(self, filename="tpr_fpr.png"):
        from matplotlib.lines import Line2D

        data = self.data
        n_rows, n_cols = len(data.fraction_bins), len(data.columns)

        self._apply_theme()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
            squeeze=False,
            gridspec_kw={"width_ratios": self._width_ratios()},
        )

        for row_idx, fraction_bin in enumerate(data.fraction_bins):
            for col_idx, column in enumerate(data.columns):
                ax = axes[row_idx][col_idx]
                self._style_axes(ax, grid_axis="both")
                ax.set_xlim(-0.015, 0.515)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xticks([0.0, 0.125, 0.25, 0.375, 0.5])
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.plot([0, 0.5], [0, 0.5], color=_GRIDLINE, linewidth=1.0,
                        linestyle="--", zorder=1)

                x_bins = data.x_bins_by_column[column]
                color_for_bin = self._bin_colors_for_column(column)

                points = []
                for x_bin in x_bins:
                    cell = data.panel_b.get((column, fraction_bin, x_bin))
                    if cell is None or cell.n_runs == 0:
                        continue
                    points.append((x_bin, cell))

                if not points:
                    self._annotate_empty(ax)
                else:
                    if len(points) > 1:
                        ax.plot(
                            [c.mean_fpr for _, c in points],
                            [c.mean_tpr for _, c in points],
                            color=_INK_MUTED, linewidth=1.0, zorder=2,
                        )
                    for x_bin, cell in points:
                        color = color_for_bin[x_bin]
                        ax.errorbar(
                            cell.mean_fpr, cell.mean_tpr,
                            xerr=[[cell.fpr_err_minus], [cell.fpr_err_plus]],
                            yerr=[[cell.tpr_err_minus], [cell.tpr_err_plus]],
                            fmt="o", markersize=_MARKER_SIZE, color=color,
                            ecolor=_INK_SECONDARY, elinewidth=1.2,
                            capsize=3, capthick=1.2,
                            markeredgecolor=_SURFACE, markeredgewidth=1.5, zorder=3,
                        )
                        ax.annotate(
                            f"n={cell.n_runs}",
                            xy=(cell.mean_fpr, cell.mean_tpr),
                            xytext=(6, -9), textcoords="offset points",
                            fontsize=5.5, color=_INK_MUTED,
                        )

                    legend_handles = [
                        Line2D([0], [0], marker="o", linestyle="none",
                               markersize=5.5, markerfacecolor=color_for_bin[x_bin],
                               markeredgecolor=_SURFACE, markeredgewidth=0.8, label=x_bin)
                        for x_bin in x_bins
                    ]
                    ax.legend(
                        handles=legend_handles, loc="upper right",
                        fontsize=5.5, framealpha=0.85, borderpad=0.4,
                        handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4,
                    )

                if row_idx == 0:
                    ax.set_title(column, fontsize=9.5, color=_INK_PRIMARY, pad=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("FPR", fontsize=8.5)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    ax.set_ylabel(f"TPR   |   alt proportion {fraction_bin}", fontsize=8.5)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fig.suptitle(
            "Phlag anomaly detection: TPR vs FPR by branch length and alternative sequence fraction",
            fontsize=12, color=_INK_PRIMARY, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])

        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path


    def render_relerr(self, filename="relerr.png"):
        """
        Same grid as f1.png (fraction bin x category x branch-length/divergence
        bin), but each cell holds two bars instead of one: mean |relative
        error(%)| of the EM-fitted mean, and of the EM-fitted covariance (std),
        vs. the ground-truth-split empirical fit. Lower is better -- this is EM
        fit quality, not detection performance.
        """
        data = self.data
        n_rows, n_cols = len(data.fraction_bins), len(data.columns)

        pair_offset = _BAR_WIDTH * 0.27
        pair_width = _BAR_WIDTH * 0.42

        # Fixed axis -- relerr is a percentage, so a consistent 0-150 scale
        # with every-25 gridlines makes cells comparable at a glance instead
        # of rescaling per run. Bars/whiskers past 150 are simply clipped at
        # the top edge.
        y_top = 150.0
        yticks = np.arange(0, 151, 25)

        self._apply_theme()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
            squeeze=False,
            gridspec_kw={"width_ratios": self._width_ratios()},
        )

        for row_idx, fraction_bin in enumerate(data.fraction_bins):
            for col_idx, column in enumerate(data.columns):
                ax = axes[row_idx][col_idx]
                x_bins = data.x_bins_by_column[column]

                self._style_axes(ax)
                ax.set_xlim(-0.7, len(x_bins) - 0.3)
                ax.set_ylim(0.0, y_top)
                ax.set_yticks(yticks)
                self._int_yaxis(ax)
                ax.set_xticks(list(range(len(x_bins))))
                ax.set_xticklabels(x_bins, rotation=30, ha="right", fontsize=7.5)

                drew_any = False
                for pos, x_bin in enumerate(x_bins):
                    cell = data.panel_relerr.get((column, fraction_bin, x_bin))
                    if cell is None or cell.n_runs == 0:
                        continue
                    drew_any = True

                    ax.bar(
                        pos - pair_offset, cell.mean_relerr, width=pair_width,
                        color=_SERIES_RELERR_MEAN, edgecolor=_SURFACE, linewidth=1.0,
                        zorder=2, label="Mean rel.err(%)",
                    )
                    ax.errorbar(
                        pos - pair_offset, cell.mean_relerr,
                        yerr=[[cell.mean_relerr_err_minus], [cell.mean_relerr_err_plus]],
                        fmt="none", ecolor=_INK_SECONDARY,
                        elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
                    )
                    ax.bar(
                        pos + pair_offset, cell.covar_relerr, width=pair_width,
                        color=_SERIES_RELERR_COVAR, edgecolor=_SURFACE, linewidth=1.0,
                        zorder=2, label="Covariance rel.err(%)",
                    )
                    ax.errorbar(
                        pos + pair_offset, cell.covar_relerr,
                        yerr=[[cell.covar_relerr_err_minus], [cell.covar_relerr_err_plus]],
                        fmt="none", ecolor=_INK_SECONDARY,
                        elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
                    )
                    top = max(
                        cell.mean_relerr + cell.mean_relerr_err_plus,
                        cell.covar_relerr + cell.covar_relerr_err_plus,
                    )
                    ax.text(
                        pos, top * 1.03, f"n={cell.n_runs}",
                        ha="center", va="bottom", fontsize=6.5, color=_INK_MUTED,
                    )

                if not drew_any:
                    self._annotate_empty(ax)

                if row_idx == 0:
                    ax.set_title(column, fontsize=9.5, color=_INK_PRIMARY, pad=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel(data.x_axis_label_by_column[column], fontsize=8)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    ax.set_ylabel(f"alt proportion {fraction_bin}", fontsize=8.5)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        unique = {}
        for row_axes in axes:
            for ax in row_axes:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label and label not in unique:
                        unique[label] = handle
        if unique:
            fig.legend(
                unique.values(), unique.keys(),
                loc="upper right", bbox_to_anchor=(0.995, 0.97),
                fontsize=7.5, framealpha=0.9,
            )

        fig.suptitle(
            "Phlag EM fit quality: mean |relative error(%)| of fitted mean/covariance vs. ground truth",
            fontsize=12, color=_INK_PRIMARY, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])

        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path


    def render_em_divergence(self, filename="em_divergence.png"):
        """
        Same grid as f1.png/relerr.png: one paired bar per x_bin. The left
        (bin-colored, same ramp as f1.png) bar is the mean Hellinger distance
        between the fitted Null/Alt states' Gaussians
        (hmm.PhlagHMMEmissions.em_divergence) -- bounded [0, 1], where 0 means
        the two states are statistically identical and 1 means fully
        separated (higher is better separation). This is state separation,
        not detection performance or fit quality.

        The right (fixed teal) bar is the same distance computed directly on
        the ground-truth Null/Alt split instead of the fitted states -- the
        per-topology marginal average described at RunRecord.em_gt_bd -- so a
        cell where the model's fitted states are less separated than the
        ground truth's inherent separation (fitted bar shorter than ground
        truth) is visible at a glance, not just inferable from F1.
        """
        import matplotlib.patches as mpatches

        data = self.data
        n_rows, n_cols = len(data.fraction_bins), len(data.columns)

        pair_offset = _BAR_WIDTH * 0.27
        pair_width = _BAR_WIDTH * 0.42

        self._apply_theme()
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.3 * n_cols, 3.1 * n_rows + 0.8),
            squeeze=False,
            gridspec_kw={"width_ratios": self._width_ratios()},
        )

        for row_idx, fraction_bin in enumerate(data.fraction_bins):
            for col_idx, column in enumerate(data.columns):
                ax = axes[row_idx][col_idx]
                x_bins = data.x_bins_by_column[column]
                color_for_bin = self._bin_colors_for_column(column)

                self._style_axes(ax)
                ax.set_xlim(-0.7, len(x_bins) - 0.3)
                ax.set_ylim(0.0, 1.18)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xticks(list(range(len(x_bins))))
                ax.set_xticklabels(x_bins, rotation=30, ha="right", fontsize=7.5)

                drew_any = False
                for pos, x_bin in enumerate(x_bins):
                    cell = data.panel_em_divergence.get((column, fraction_bin, x_bin))
                    if cell is None or cell.n_runs == 0:
                        continue
                    drew_any = True

                    ax.bar(
                        pos - pair_offset, cell.mean_distance, width=pair_width,
                        color=color_for_bin[x_bin],
                        edgecolor=_SURFACE, linewidth=1.0, zorder=2,
                    )
                    ax.errorbar(
                        pos - pair_offset, cell.mean_distance,
                        yerr=[[cell.err_minus], [cell.err_plus]],
                        fmt="none", ecolor=_INK_SECONDARY,
                        elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
                    )
                    top = cell.mean_distance + cell.err_plus

                    if cell.n_gt_runs > 0:
                        ax.bar(
                            pos + pair_offset, cell.gt_mean_distance, width=pair_width,
                            color=_SERIES_GT_DIVERGENCE,
                            edgecolor=_SURFACE, linewidth=1.0, zorder=2,
                        )
                        ax.errorbar(
                            pos + pair_offset, cell.gt_mean_distance,
                            yerr=[[cell.gt_err_minus], [cell.gt_err_plus]],
                            fmt="none", ecolor=_INK_SECONDARY,
                            elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
                        )
                        top = max(top, cell.gt_mean_distance + cell.gt_err_plus)

                    ax.text(
                        pos, top * 1.03, f"n={cell.n_runs}",
                        ha="center", va="bottom", fontsize=6.5, color=_INK_MUTED,
                    )

                if not drew_any:
                    self._annotate_empty(ax)

                if row_idx == 0:
                    ax.set_title(column, fontsize=9.5, color=_INK_PRIMARY, pad=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel(data.x_axis_label_by_column[column], fontsize=8)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    ax.set_ylabel(f"alt proportion {fraction_bin}", fontsize=8.5)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fitted_proxy = mpatches.Patch(color=_SERIES_PHLAG, alpha=0.85, label="Fitted (Null/Alt EM states, bin-colored)")
        gt_proxy = mpatches.Patch(color=_SERIES_GT_DIVERGENCE, label="Ground truth (Null/Alt split)")
        fig.legend(
            handles=[fitted_proxy, gt_proxy],
            loc="upper right", bbox_to_anchor=(0.995, 0.97),
            fontsize=7.5, framealpha=0.9,
        )

        fig.suptitle(
            "Phlag EM state separation: Hellinger distance, fitted vs. ground truth",
            fontsize=12, color=_INK_PRIMARY, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])

        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path


    def render(self):
        """Renders all figures; returns the list of written PNG paths."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.render_panel_a(), self.render_panel_b(),
            self.render_relerr(), self.render_em_divergence(),
        ]



def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark: run caster and phlag over every dataset in simulations/ "
                    "(rerunning every stage by default), then summarize every "
                    "finished run into the Figure-3 panels."
    )
    parser.add_argument(
        "--skip",
        dest="skip",
        nargs="+",
        default=[],
        choices=["caster", "phlag"],
        help="Reuse existing output instead of rerunning, for the named stage(s) "
             "(default: rerun everything -- config.json flags can change what a stage "
             "does without touching any source file, so existing output is never "
             "assumed to still be current unless you say so here)",
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
        "--run",
        dest="run",
        type=int,
        default=None,
        help="Rerun summary stats into an existing numbered run folder "
             "(<phlag_base>/<dist-type>/w<W>_s<S>/benchmark/<run>/) instead of "
             "allocating a new one -- tables/figures are overwritten in place "
             "and the change message is appended to that folder's report.txt",
    )
    parser.add_argument(
        "--errorbar",
        dest="errorbar",
        default="sd",
        choices=list(ERRORBAR_KINDS),
        help="Error bar shown on each aggregated cell (default: sd)",
    )
    parser.add_argument(
        "change",
        nargs="?",
        default=None,
        help="Short description of what's different in this run compared to earlier ones; "
             "written to report.txt in the run's output folder, alongside how long the run took",
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
    from phlag.utils import get_data_dir, get_phlag_output_base, get_simulation_categories, get_short_sim_name
    from phlag.caster import format_val

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
    from phlag.utils import clean_locus_name

    sim_output_dir = get_expected_sim_output_dir(
        fasta_path, leaf_dir, dist_type=dist_type,
        window_size=window_size, step_size=step_size,
    )
    pattern_stem = clean_locus_name(fasta_path.stem)
    return sim_output_dir / pattern_stem / "caster" / "scores.tsv"


def get_expected_report_path(fasta_path, leaf_dir, dist_type=DEFAULT_DIST_TYPE,
                             window_size=DEFAULT_WINDOW_SIZE, step_size=DEFAULT_STEP_SIZE):
    from phlag.utils import clean_locus_name

    sim_output_dir = get_expected_sim_output_dir(
        fasta_path, leaf_dir, dist_type=dist_type,
        window_size=window_size, step_size=step_size,
    )
    pattern_stem = clean_locus_name(fasta_path.stem)
    return sim_output_dir / pattern_stem / "phlag" / "report.tsv"


def run_step(cmd, label, cwd=None, env=None):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if result.returncode != 0:
        return False, result.stderr
    return True, result.stdout


def create_source_snapshot():
    """
    Copies both phlag/ (the package caster/phlag run out of) and bench/ (this
    module, phlagster.py, and config.json) into a temp dir, so a run_all()
    call's subprocess invocations -- including the phlagster path, and
    config.json's CLI defaults -- stay frozen against whatever the source
    tree looked like when the run started, immune to concurrent edits.
    """
    import os
    import tempfile

    from phlag.utils import get_repo_root

    repo_root = get_repo_root()
    snapshot_root = pathlib.Path(tempfile.mkdtemp(prefix="phlag_snapshot_"))
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
    shutil.copytree(repo_root / "phlag", snapshot_root / "phlag", ignore=ignore)
    shutil.copytree(repo_root / "bench", snapshot_root / "bench", ignore=ignore)
    snapshot_env = dict(os.environ)
    snapshot_env["PHLAG_REPO_ROOT"] = str(repo_root)
    return snapshot_root, snapshot_env


def _fmt_metric(value):
    return f"{value:.4f}" if value is not None else "N/A"


def _group_cli_tokens(tokens):
    lines = []
    for tok in tokens:
        is_flag = tok.startswith("-") and not (len(tok) > 1 and tok[1].isdigit())
        if is_flag or not lines:
            lines.append(tok)
        else:
            lines[-1] += " " + tok
    return lines


def run_all(args, sim_root):
    """
    Runs phlagster (caster + phlag) over every complete simulation leaf x concat-fasta
    combination, restricted to the current affected-fraction sweep (DEFAULT_CONCAT_PATTERNS).
    Every default-pattern FASTA under a leaf's concat/ directory is its own unit of work.
    Anything else found there -- older one-off intervals, stray pattern-token files -- is
    left alone: it is neither run here nor counted by summarize().

    Caster and phlag are rerun by default, since both draw their actual CLI flags
    from config.json rather than only from source code -- a flag change there
    doesn't touch any .py file's mtime, so file-mtime-based staleness detection
    can't be trusted to notice it. --skip caster and/or --skip phlag opt back
    into reusing existing output for the named stage(s), if present. When both
    stages are running, they run as a single `phlagster` invocation (in-process
    caster-then-phlag, avoiding two separate interpreter/JAX startups); when
    only phlag is running (caster skipped), `phlag` alone is invoked directly
    against the existing scores.tsv.

    Neither invocation path passes --np/--ap: both phlag and phlagster fall
    through to phlag's own emission-parameterization default, so the sweep
    always matches whatever that default currently is without needing to be
    kept in sync here.
    """
    from phlag.utils import find_mapping_file, load_cli_config

    snapshot_root, snapshot_env = create_source_snapshot()

    leaf_dirs = discover_leaf_dirs(sim_root)

    work_items = []
    skipped_incomplete = 0

    for leaf_dir in leaf_dirs:
        rel_path = leaf_dir.relative_to(sim_root).as_posix()

        mapping_path, mapping_reason = find_mapping_file(leaf_dir)
        if mapping_path is None:
            print(f"[skip] {rel_path}: incomplete ({mapping_reason})")
            skipped_incomplete += 1
            continue

        concat_dir = leaf_dir / "concat"
        fasta_paths = sorted(
            p for p in concat_dir.glob("*.fa")
            if p.is_file() and p.stem in DEFAULT_CONCAT_PATTERNS
        ) if concat_dir.is_dir() else []
        if not fasta_paths:
            print(f"[skip] {rel_path}: incomplete (missing default-pattern concat/*.fa)")
            skipped_incomplete += 1
            continue

        for fasta_path in fasta_paths:
            work_items.append((leaf_dir, rel_path, fasta_path))

    caster_variable, _ = load_cli_config("caster")
    phlag_variable, _ = load_cli_config("phlag")

    print(
        f"\n{len(work_items)} file(s) to process\n"
        f"{'skipping ' + ', '.join(args.skip) if args.skip else 'rerunning all outputs'}\n"
        f"dist-type={args.dist_type}, window={DEFAULT_WINDOW_SIZE}, step={DEFAULT_STEP_SIZE}\n"
        "caster variable args:\n" + "\n".join(_group_cli_tokens(caster_variable)) + "\n"
        "phlag variable args:\n" + "\n".join(_group_cli_tokens(phlag_variable))
    )

    processed = 0
    skipped_present = 0
    failures = []

    for leaf_dir, rel_path, fasta_path in work_items:
        pattern_rel = f"{rel_path}/{fasta_path.stem}"

        scores_path = get_expected_scores_path(fasta_path, leaf_dir, dist_type=args.dist_type)
        report_path = get_expected_report_path(fasta_path, leaf_dir, dist_type=args.dist_type)

        caster_done = scores_path.exists() and "caster" in args.skip
        phlag_done = report_path.exists() and "phlag" in args.skip

        if caster_done and phlag_done:
            print(f"[skip] {pattern_rel}: already present at {report_path.parent}")
            skipped_present += 1
            continue

        if caster_done:
            print(f"[skip caster] {pattern_rel}: scores already at {scores_path}")
            phlag_ok, phlag_out = run_step(
                [sys.executable, "-m", "phlag.phlag", str(scores_path), "-d", args.dist_type,
                 "--plot", "-s", "1000"],
                "phlag",
                cwd=str(snapshot_root), env=snapshot_env,
            )
            if not phlag_ok:
                msg = f"{pattern_rel}: phlag failed\n{phlag_out}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue
        else:
            phlagster_ok, phlagster_out = run_step(
                [sys.executable, "-m", "bench.phlagster", str(fasta_path), "-d", args.dist_type,
                 "--no-plots"],
                "phlagster",
                cwd=str(snapshot_root), env=snapshot_env,
            )
            if not phlagster_ok:
                msg = f"{pattern_rel}: phlagster failed\n{phlagster_out}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue
            if not scores_path.exists():
                msg = f"{pattern_rel}: phlagster reported success but scores file not found at {scores_path}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue
            if not report_path.exists():
                msg = f"{pattern_rel}: phlagster reported success but report file not found at {report_path}"
                print(f"{RED}[fail] {msg}{RESET}")
                failures.append(msg)
                continue

        parsed = parse_report(report_path)
        print(
            f"[done] {pattern_rel}: TPR={_fmt_metric(parsed['tpr'])} "
            f"FPR={_fmt_metric(parsed['fpr'])} F1={_fmt_metric(parsed['f1'])}"
        )
        processed += 1

    shutil.rmtree(snapshot_root, ignore_errors=True)

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


ANALYSIS_METRICS = ["accuracy", "tpr", "fpr", "f1"] + [
    f"relerr_{topo}_{state}_{stat}"
    for topo in TOPOLOGY_NAMES
    for state in RELERR_STATES
    for stat in RELERR_STATISTICS
] + ["em_bd", "em_gt_bd"] + [
    "transition_null_to_null", "transition_null_to_alt",
    "transition_alt_to_null", "transition_alt_to_alt",
] + ["bic"]


def compute_analysis(runs_path, out_path=None):
    """
    Reads runs.tsv into a pandas DataFrame and reduces every metric
    in ANALYSIS_METRICS (accuracy/tpr/fpr/f1, every per-topology EM
    mean/covariance relative-error column, EM state divergence, and the fitted
    Null/Alt transition probabilities) to its mean plus the run with the
    maximum and the minimum value -- so an outlier is traceable straight back
    to its run_id without re-scanning every report.tsv.

    Writes analysis.tsv next to runs_path (or to out_path if given) and
    returns (analysis_df, out_path).
    """
    import pandas as pd

    runs_path = pathlib.Path(runs_path)
    df = pd.read_csv(runs_path, sep="\t")

    rows = []
    for metric in ANALYSIS_METRICS:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        valid = values.notna()
        n = int(valid.sum())
        if n == 0:
            rows.append({
                "metric": metric, "n": 0, "mean_value": None,
                "max_value": None, "max_run_id": None,
                "min_value": None, "min_run_id": None,
            })
            continue
        max_idx = values[valid].idxmax()
        min_idx = values[valid].idxmin()
        rows.append({
            "metric": metric, "n": n, "mean_value": float(values[valid].mean()),
            "max_value": float(values[max_idx]), "max_run_id": df.loc[max_idx, "run_id"],
            "min_value": float(values[min_idx]), "min_run_id": df.loc[min_idx, "run_id"],
        })

    analysis_df = pd.DataFrame(
        rows,
        columns=["metric", "n", "mean_value", "max_value", "max_run_id", "min_value", "min_run_id"],
    )

    if out_path is None:
        out_path = runs_path.parent / "analysis.tsv"
    out_path = pathlib.Path(out_path)
    analysis_df.to_csv(out_path, sep="\t", index=False, float_format="%.6g")
    return analysis_df, out_path


def save_reports(stats, out_dir):
    """
    Snapshots every discovered report.tsv (same set as runs.tsv, including
    excluded runs) into out_dir/reports/<category>/<subcategory>/<leaf>/<pattern>.tsv
    -- a per-run copy of the exact source reports this run's tables/figures were
    built from, so two numbered benchmark runs can be diffed report-by-report
    after a code change instead of only comparing the aggregated stats.

    <leaf> is the basename of the leaf directory under simulations/ (e.g. an
    admixture pair's directory name, or a plain node id for recombination/10X)
    -- not the "Clade" header value, which can be shared by several leaves.
    """
    reports_dir = out_dir / "reports"
    n_saved = 0
    for record in stats.runs:
        src = pathlib.Path(record.report_path)
        if not src.exists():
            continue
        leaf_name = pathlib.PurePosixPath(record.source_leaf).name
        dest_dir = reports_dir / record.category / record.subcategory / leaf_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / f"{record.pattern}.tsv")
        n_saved += 1
    return reports_dir, n_saved


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def summarize(args, sim_root, reuse_dir=False, start_time=None):
    """
    Aggregates every finished run into the Figure-3 panels and writes them out.

    reuse_dir: True when the preceding run_all() processed nothing new (no
    reruns, no new reports) -- in that case there's nothing to compare against
    a fresh numbered folder for, so the latest existing run dir is modified in
    place (tables/figures overwritten, report.txt appended to) instead of
    allocating a new one. Falls back to a fresh dir if none exist yet.

    args.run: explicit run number to target (from --run), e.g. to rerun
    stats/figures into an old folder after a code change without disturbing
    other runs' numbering. Same in-place/append semantics as reuse_dir, and
    takes priority over it. If the given run number doesn't exist yet, it's
    created fresh at that exact number rather than falling back to the next
    auto-allocated one.

    start_time: time.perf_counter() value from when the `benchmark` command
    started (main()'s first line) -- used to record how long the run took
    in report.txt alongside the change note.
    """
    stats = BenchmarkStats(
        sim_root=sim_root,
        dist_type=args.dist_type,
        errorbar=args.errorbar,
    )
    stats.collect()
    figure_data = stats.aggregate()
    stats.print_diagnostics(figure_data)

    if args.run is not None:
        out_dir = stats.default_stats_dir() / str(args.run)
        if out_dir.exists():
            reuse = True
        else:
            print(
                f"{RED}--run {args.run}: no existing run at {out_dir} "
                f"-- creating it fresh.{RESET}"
            )
            reuse = False
    elif reuse_dir:
        out_dir = stats.latest_run_dir() or stats.next_run_dir()
        reuse = reuse_dir
    else:
        out_dir = stats.next_run_dir()
        reuse = False
    out_dir.mkdir(parents=True, exist_ok=True)

    config_src = pathlib.Path(__file__).parent / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, out_dir / "config.json")
        print(f"Wrote config:         {out_dir / 'config.json'}")

    report_lines = []
    if args.change:
        report_lines.append(args.change)
    if start_time is not None:
        report_lines.append(f"Benchmark took {_format_duration(time.perf_counter() - start_time)}")

    if report_lines:
        report_txt_path = out_dir / "report.txt"
        text = "\n".join(report_lines) + "\n"
        if reuse and report_txt_path.exists():
            with open(report_txt_path, "a") as f:
                f.write(text)
            print(f"Appended to report:  {report_txt_path}")
        else:
            report_txt_path.write_text(text)
            print(f"Wrote report:         {report_txt_path}")

    runs_path, bins_path = stats.write_tables(out_dir, figure_data)
    print(f"\nWrote per-run table:  {runs_path}")
    print(f"Wrote per-cell table: {bins_path}")

    _analysis_df, analysis_path = compute_analysis(runs_path)
    print(f"Wrote analysis table: {analysis_path}")

    reports_dir, n_reports_saved = save_reports(stats, out_dir)
    print(f"Saved {n_reports_saved} report(s) to: {reports_dir}")

    if (
        figure_data.n_runs_panel_a == 0 and figure_data.n_runs_panel_b == 0
        and figure_data.n_runs_relerr == 0 and figure_data.n_runs_em_divergence == 0
    ):
        print(f"{RED}No runs qualified for any panel -- skipping figures "
              f"(see the exclusions above).{RESET}")
        return figure_data

    plotter = BenchmarkFigurePlotter(figure_data, out_dir)
    for figure_path in plotter.render():
        print(f"Wrote figure:         {figure_path}")

    return figure_data


def main(argv=None):
    start_time = time.perf_counter()
    args = parse_arguments(argv)

    from phlag.utils import get_data_dir

    sim_root = get_data_dir() / "simulations"
    if not sim_root.exists():
        sys.exit(f"Error: simulations directory not found at '{sim_root}'")

    processed, _skipped_incomplete, _skipped_present, _failures = run_all(args, sim_root)
    summarize(args, sim_root, reuse_dir=(processed == 0), start_time=start_time)


if __name__ == "__main__":
    main()
