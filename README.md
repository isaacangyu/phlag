# Phlag

**Phlag** detects and flags phylogenetic anomalies across the genome.

Given CASTER scores across the genome, Phlag detects strong deviations from the multi-species coalescent (MSC) using a hidden Markov model (HMM).

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

The `caster` command-line utility calculates CASTER scores, support for each of the three quartet topologies, on substrings of multiple sequence alignments, and applies an average sliding windows function. 

### Usage

```shell
caster caster/data/ape.fa -l 0 -r 200000 -w 1000 -s 100 -n -m caster/data/ape_mapping.tsv
```

### Arguments

* **`fasta_file`**: Input multiple sequence alignment FASTA file path.
* **`-l, --left`**: Left coordinate range bound (0-indexed, inclusive).
* **`-r, --right`**: Right coordinate range bound (0-indexed, exclusive).
* **`-w, --window-size`** (default: `10000`): Sliding window size (bp) for calculating D* statistic.
* **`-s, --step-size`** (default: `10000`): Step size (stride translation step) for sliding window.
* **`-n, --normalize`**: Apply min-max normalization to scores.
* **`-m, --mapping`**: Population/clade structure mapping file path. 

### Output

The output TSV files are stored inside `caster/data/`. Genomic bounds, window parameters, and normalization are appended to the output filename.

* **Example output file name**: `caster/data/ape_0_200k_w1k_s100_n.tsv`

---

## 2. Phlag HMM CLI Utility

The `phlag` utility fits a Gaussian HMM to the CASTER topology scores to flag anomalous genomic regions.

### Usage

```shell
phlag -c caster/data/ape_0_200k_w1k_s100_n.tsv -L 1 -w 100
```

### Required Arguments

* **`-c, --caster-scores`**: Path to TSV containing the CASTER scores.
* **`-L, --n-iters`** (default: `5`): Number of outer EM iterations.
* **`-l, --increment-steps`** (default: `50`): Number of inner EM iterations per outer iteration.
* **`-s, --step-size`**: Genomic step size for counts of anomalies.

### Optional Arguments

* **`-o, --output-file`** (default: extracts input filename): Custom path to save the output report.
* **`--ilr-transform`**: Apply isometric log-ratio transformation to CASTER scores.
* **`--emission-parameterization`** (default: `attraction`): Parameterization of emission probabilities (`free`, `attraction`, or `anchor`).

### Output

The output report and histogram are automatically saved in the `test/` directory, named `results_` + `input_filename` (e.g., [test/results_ape_0_200k_w1k_s100_n.tsv](file:///c:/Users/isaac/phlag/test/results_ape_0_200k_w1k_s100_n.tsv)).

The report contains:
* **Header lines** (prefixed with `#`): Including the command run and HMM state emission divergence.
* **State path**: A sequence of binary states (`0` for standard MSC, `1` for anomalous states).

and the histogram contains counts of anomalous positions in non-overlapping genomic windows.
