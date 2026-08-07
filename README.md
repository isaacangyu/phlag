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

* **`fasta_file`** (optional): Input FASTA (or scores TSV) file path. If omitted, falls back to `-r/--recent`.
* **`-r, --recent`**: Use the most recently created FASTA file in `store/msa/concat`.
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

* **`caster_scores`** (optional): Path to TSV containing the CASTER scores. If omitted, falls back to `-r/--recent`.
* **`-r, --recent`**: Use the most recently created score file in the model parameter directories.
* **`-o`** (optional): Path to save the output report (defaults to a directory derived from the input's window/step parameters and `-d` dist type).
* **`--plot`** (default: `em states`): Plots to generate (`em`, `states`).
* **`-L, --n-iters`** (default: `10`): Number of outer EM iterations.
* **`-s, --step-size`**: Genomic step size (in rows/positions) to compute a text-based ASCII histogram and save a visual bar chart plot. Required if `--plot` is supplied.

### HMM Parameters

* **`-e`** (default: `attraction`): Parameterization of the emission probabilities of the default state (`free`, `attraction`, or `anchor`).
* **`--emission-lambda`** (default: `1.0`): Emission penalty regularizer parameter lambda.
* **`-d`** (default: `gaussian`): Type of HMM emissions (`gaussian`, `beta`, or `gmm`).
* **`-c, --cluster-topologies`**: For GMM, cluster topologies together instead of independently.
* **`-t, --silhouette-threshold`** (default: `0.5`): Silhouette score threshold to determine optimal GMM mixture counts.
* **`-p, --best-paths`** (default: `1`): Number of best Viterbi paths to calculate and plot.
* **`--correct-transition`**: Set final transition matrix to ground truth automatically, or via custom values (`p0,p1`).

---

## 3. Benchmark CLI Utility

The `benchmark` utility runs `caster`/`phlag` (via `phlagster`) over every simulation leaf under `<data-dir>/simulations`, skipping stages whose output already exists, then aggregates all finished runs into the Figure-3 summary tables and figures.

### Usage

```shell
benchmark -d gaussian --errorbar sd
```

### Arguments

* **`--rerun`**: Force `caster` and `phlag` to re-run even where output already exists (default: skip what's already there).
* **`-d, --dist-type`** (default: `gaussian`): Distribution type (`gaussian`, `gmm`), threaded through to `caster`/`phlag` and used to locate outputs when summarizing.
* **`--stats-out`** (optional): Directory for the summary tables and figures (default: `<phlag_base>/<dist-type>/w<W>_s<S>/benchmark/`).
* **`--errorbar`** (default: `sd`): Error bar shown on each aggregated cell (`sd`, `sem`, or `ci95`).
