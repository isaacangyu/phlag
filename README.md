# Phlag

**Phlag** detects and flags phylogenetic anomalies across the genome.

Given CASTER scores across the genome, Phlag detects strong deviations from the multi-species coalescent (MSC) using a hidden Markov model (HMM).

---

## Installation

Phlag requires Python 3.9.

```shell
micromamba create -n phlag python=3.9 -y
micromamba activate phlag
git clone https://github.com/isaacangyu/phlag.git
cd phlag
pip install .
```

## Environment Configuration

All scripts require `$CONNECTION_DIR`. This repository reads from `$CONNECTION_DIR/simulations/` and writes to `$CONNECTION_DIR/phlag/`; simulation repositories read from and write to `$CONNECTION_DIR/simulations/`. Set it via a `.env` file in the repo root:

```env
CONNECTION_DIR=/path/to/shared_directory
```

## 1. Caster CLI Utility

The `caster` command-line utility calculates CASTER scores, support for each of the three quartet topologies, on substrings of multiple sequence alignments, and applies an average sliding windows function.

### Usage

```shell
caster caster/data/ape.fa -l 0 -R 200000 -w 1000 -s 100 -n -m mapping/ape_mapping.tsv
```

### Arguments

* **`fasta_file`**: Input FASTA (or scores TSV) file path.
* **`-l`** (default: `0`): Left coordinate range bound (0-indexed, inclusive).
* **`-R, --right`**: Right coordinate range bound (0-indexed, exclusive).
* **`-w`** (default: `50000`): Sliding window size (bp) for calculating D* statistic.
* **`-s`** (default: `1000`): Step size (stride translation step) for sliding window.
* **`-n`**: Apply min-max normalization to D* scores.
* **`-m`**: Population/clade structure mapping file path.
* **`--plot`** (default: `scores dist`): Plots to generate (`scores`, `dist`).
* **`-t, --topologies`**: List of topologies to plot (default: all).
* **`-d, --dist-type`** (default: `gaussian`): Distribution type (`gaussian`, `gmm`) for the output directory structure.
* **`--tree`**: Optional species tree file (default: `store/63K.tre`).

---

## 2. Phlag HMM CLI Utility

The `phlag` utility fits a Gaussian HMM to the CASTER topology scores to flag anomalous genomic regions.

### Usage

```shell
phlag caster/data/ape_0_200k_w1k_s100_n.tsv -L 10 -s 100
```

### Arguments

* **`caster_scores`**: Path to TSV containing the CASTER scores.
* **`-o`** (optional): Path to save the output report (defaults to a directory derived from the input's window/step parameters; model design is inferred from the input/output filename, defaulting to `gaussian`).
* **`--plot`** (default: `em states`): Plots to generate (`em`, `states`).
* **`-L, --n-iters`** (default: `10`): Number of outer EM iterations.
* **`-s, --step-size`**: Genomic step size (in rows/positions) to compute a text-based ASCII histogram and save a visual bar chart plot. Required if `--plot` is supplied.

### HMM Parameters

* **`--np`** (default: `free`): Null state's emission parameterization (`free` or `repulsion`).
* **`--ap`** (default: `free`): Alt state's emission parameterization (`free` or `repulsion`).
* **`--lam`** (default: `1.0`): Emission penalty regularizer lambda.
* **`--annealing`**: Anneal the repulsion penalty lambda across EM iterations instead of holding it fixed, keeping the run's time-average pinned to `--lam`.
* **`--double-variance-init`**: Seed the alt state's initial variance at 2x the null state's.
* **`--repulsion-optimizer`** (default: `lm`): Optimizer for the repulsion parameterization's MAP fit (`lm` or `gd`).
* **`--mu`** (default: `1.0`): Initial damping for the `lm` repulsion optimizer.
* **`-t, --silhouette-threshold`** (default: `0.5`): Silhouette score threshold to determine optimal GMM mixture counts.
* **`-p, --best-paths`** (default: `1`): Number of best Viterbi paths to calculate and plot.
* **`--correct-transition`**: Set final transition matrix to ground truth automatically, or via custom values (`p0,p1`).

---

## 3. Benchmark CLI Utility

The `benchmark` utility runs `caster`/`phlag` over every simulation leaf under `<data-dir>/simulations`, then aggregates finished runs into summary tables and figures.

### Usage

```shell
benchmark --create store/phlag/gaussian/w50k_s1k/benchmark/my-run -d gaussian --errorbar sd
```

### Arguments

* **`--create PATH`**: Creates a fresh run at `PATH` (a repo-relative path starting with `store/`); errors out if `PATH` already exists. Exactly one of `--create`/`--rerun`/`--copy` is required.
* **`--rerun PATH`**: Reruns an existing run at `PATH` using that run's own recorded `args.json` (or, if `PATH` holds several runs nested under it, reruns all of them). Exactly one of `--create`/`--rerun`/`--copy` is required.
* **`--copy POS1 POS2`**: Copies every run under `POS1` into `POS2`, overriding any mirrored flag passed directly on this invocation.
* **`--sweep FLAG=V1,V2,...`**: Only with `--create` -- runs one leaf per value of `FLAG` under `PATH` as a container (e.g. `--sweep=-w=500k,250k,100k`).
* **`--skip`**: Reuse existing output instead of rerunning, for the named stage(s), comma-separated (e.g. `caster,phlag`).
* **`-d, --dist-type`** (default: `gaussian`): Distribution type (`gaussian`, `gmm`), threaded through to `caster`/`phlag`.
* **`--errorbar`** (default: `sd`): Error bar shown on each aggregated cell (`sd`, `sem`, or `ci95`).
* **`change`** (optional): Short description of what's different in this run, written to the run's report.
* Every other `caster`/`phlag` flag (`-w`/`--window-size`, `-s`/`--step-size`, `--np`, `--ap`, `--lam`, `--annealing`, `--pair`, etc.) is also mirrored directly onto `benchmark` -- run `benchmark --help` for the full list.
