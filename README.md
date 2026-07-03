# Phlag

**Phlag** detects and flags phylogenetic anomalies across the genome.

Given a species tree and CASTER scores across genomic positions, Phlag detects strong deviations from the multi-species coalescent (MSC) using a hidden Markov model (HMM).

The input is CASTER topology scores at genomic positions ordered along a chromosome. The output is a set of flagged subsequences corresponding to regions with anomalous topology distributions.

## Installation

Phlag requires Python 3.9.

```shell
micromamba create -n phlag python=3.9 -y
micromamba activate phlag
git clone https://github.com/bo1929/phlag.git
cd phlag
pip install .
```

<!-- You can simply use pip. -->
<!-- ```shell -->
<!-- pip install phlag -->
<!-- ``` -->

<!-- Alternatively, install Phlag from the source. -->
<!-- ```shell -->
<!-- git clone https://github.com/bo1929/phlag.git -->
<!-- cd phlag -->
<!-- pip install . -->
<!-- ``` -->

## Quickstart with a toy example

The `test/` directory contains a simulated dataset based on a Neoaves species tree with 191 taxa. We use it to demonstrate how Phlag detects a genomic region where the MSC model is violated.

We have a species tree (`test/neoaves.nwk`) and 1500 CASTER score positions (`test/qqs.tsv`) ordered along a chromosome. The sequence contains a mixture of regions with two different topology distributions:

- **Background**: Positions following the standard MSC with the original species tree parameters.
- **Anomalous**: Positions with altered topology distributions due to a 10-fold increase in effective population size on the branch leading to the *Charadriiformes* clade (labeled `N159`).

Out of the 1500 consecutive positions, 150 (10%) are anomalous. They form one contiguous block at indices [913, 1063) (0-indexed). The goal is to recover that region with Phlag.

### Input

- **Species tree** (`-s`): A Newick species tree with labeled internal nodes.
- **CASTER scores** (`-c`): TSV file with CASTER topology scores (pos, ABBA, BABA, AABB) at genomic positions.
- **Focal edge** (`-e`): Label of an internal node defining the edge to target. In this example, the clade under suspicion is `N159`.

### Usage

```shell
phlag \
  -s test/neoaves.nwk \
  -c test/qqs.tsv \
  -e N159 \
  -L 10 \
  -o results-neoaves-N159.txt
```

For all options, run `phlag --help`.

### Options

- **`-L, --n-iters`** (default: `5`): Number of outer EM iterations.
- **`-l, --increment-steps`** (default: `50`): Number of inner EM iterations per outer iteration.
- **`--expand-edges`**: Include signal from neighboring/incident edges in addition to the focal edge.
- **`--ilr-transform`**: Apply isometric log-ratio transformation to CASTER scores.
- **`--emission-parameterization`** (default: `attraction`): Parameterization of emission probabilities (`free`, `attraction`, or `anchor`).

### Output

The output file contains:

- **Header lines** (prefixed with `#`):
  - The invoked command.
  - The initial species tree with branch lengths in coalescent units (CU).
  - Labels and branch lengths of the focal edge.
  - State divergence: distance between emission distributions of the two HMM states.
- **State labels**: A comma-separated sequence where `1` is for flagged (anomalous), `0` is for standard MSC.
- **Posterior probabilities**: Smoothed posterior probability of the anomalous state for each position.
