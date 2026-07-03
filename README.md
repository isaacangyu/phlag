# Phlag

**Phlag** detects and flags phylogenetic anomalies across the genome.

Given CASTER topology scores across genomic positions ordered along a chromosome, Phlag detects strong deviations from the multi-species coalescent (MSC) using a hidden Markov model (HMM).

---

## Installation

Phlag requires Python 3.9.

```shell
micromamba create -n phlag python=3.9 -y
micromamba activate phlag
git clone https://github.com/bo1929/phlag.git
cd phlag
pip install .
```

---

## 1. Caster CLI Utility

The `caster` command-line utility slices range coordinates and runs the D* statistic calculation on multiple sequence alignments.

### Usage

```shell
caster caster/data/ape.fa -l 0 -r 200000 -w 1000 -s 100 -n -m caster/data/ape_mapping.tsv
```

### Options

* **`fasta_file`** (positional, required): Input multiple sequence alignment FASTA file path.
* **`-l, --left`** (required): Left coordinate range bound (0-indexed, inclusive).
* **`-r, --right`** (required): Right coordinate range bound (0-indexed, exclusive).
* **`-w, --window-size`** (default: `10000`): Sliding window size (bp) for calculating D* statistic.
* **`-s, --step-size`** (default: `10000`): Step size (stride translation step) for sliding window.
* **`-n, --normalize`**: Apply min-max normalization to D* scores.
* **`-m, --mapping`**: Optional population/clade structure mapping file path (required for alignments with >4 taxons).

### Output

The output TSV files are stored inside `caster/data/`. The utility automatically removes the letter `C` from the input FASTA filename stem (e.g. `apeC` -> `ape`), appends the coordinate bounds, formats/abbreviates large numbers using `k` and `m`, and adds `_n` if normalized.

* **Example output file name**: `caster/data/ape_0_200k_w1k_s100_n.tsv`

---

## 2. Phlag HMM CLI Utility

The `phlag` utility fits a Gaussian HMM to the CASTER topology scores to flag anomalous genomic regions.

### Usage

```shell
phlag -c caster/data/ape_0_200k_w1k_s100_n.tsv -L 1 -w 100
```

### Options

* **`-c, --caster-scores`** (required): TSV file containing the CASTER scores.
* **`-L, --n-iters`** (default: `5`): Number of outer EM iterations.
* **`-l, --increment-steps`** (default: `50`): Number of inner EM iterations per outer iteration.
* **`-w, --window-size`**: Genomic window size (in rows/positions) to compute a text-based ASCII histogram and save a visual bar chart plot.
* **`-o, --output-file`** (optional): Custom path to save the output report (automatically defaults to saving in the `test/` directory).
* **`--ilr-transform`**: Apply isometric log-ratio transformation to CASTER scores.
* **`--emission-parameterization`** (default: `attraction`): Parameterization of emission probabilities (`free`, `attraction`, or `anchor`).

### Output

The output report and visual histogram plot are automatically saved in the `test/` directory, named `results_` + `input_filename` (e.g., [test/results_ape_0_200k_w1k_s100_n.tsv](file:///c:/Users/isaac/phlag/test/results_ape_0_200k_w1k_s100_n.tsv)).

The report file contains:
* **Header lines** (prefixed with `#`): Including the command run and HMM state emission divergence.
* **State path**: A comma-separated sequence of discrete binary states (`0` for standard MSC, `1` for anomalous flagged states).
* **ASCII Histogram**: Counts of anomalous positions in non-overlapping genomic windows.
