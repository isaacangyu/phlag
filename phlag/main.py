import sys
import pathlib
import os
import argparse

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.distributions as tfd

from collections import defaultdict
from functools import partial
from tqdm import tqdm
from skbio.stats.composition import ilr, multi_replace

from . import hmm
from . import utils

E_STEP_EPS = 0.0001
PSI_EPS = 0.001
NUM_STATES = 2
BETA_PRIME = 0.0025
INITIAL_PROBS = jnp.array([1.0000, 0.0000], dtype=jnp.float32)


class Phlag:
    def __init__(self, args):
        self.args = args
        # Inject defaults for removed CLI flags
        self.args.n_iters = 5
        self.args.step_size = None

        self.validate_parameters()
        
        # Purely ingest CASTER scores instead of computing or reading QQS
        self.read_caster_scores(self.args.caster_scores)
        
        self.configure_emissions()
        self.compute_emissions()
        self.initialize_hmm()
        self.initialize_output()

    def validate_parameters(self):
        pass

    def read_caster_scores(self, path):
        """
        Reads CASTER scores file with guaranteed schema:
        pos  avg*ABBA  avg*BABA  avg*AABB  sliding_D* QuartetCnt
        Drops sliding_D* and QuartetCnt, mapping pos to the three topology scores.
        Preserves the partial-based defaultdict structure for JAX consistency.
        Raises FileNotFoundError if the file path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CASTER scores file not found at: {path}")

        self.pos_to_caster = defaultdict(partial(jnp.zeros, 3))

        with open(path, "r") as f:
            header = f.readline()
            if not header:
                return

            for line in f:
                if not line.strip():
                    continue
                
                values = line.strip().split("\t") if "\t" in line else line.strip().split()
                
                try:
                    pos_key = int(values[0])
                    scores = jnp.array([float(values[1]), float(values[2]), float(values[3])], dtype=jnp.float32)
                    self.pos_to_caster[pos_key] = scores
                except (ValueError, IndexError):
                    continue

    def configure_emissions(self):
        self.ilr_transform = self.args.ilr_transform

    def compute_emissions(self):
        # Convert CASTER scores dictionary values into sequential matrix positions
        sorted_positions = sorted(self.pos_to_caster.keys())
        raw_caster_matrix = jnp.stack([self.pos_to_caster[pos] for pos in sorted_positions], axis=0)

        if self.ilr_transform:
            self.Y = ilr(multi_replace(raw_caster_matrix, delta=1e-7))
        else:
            self.Y = raw_caster_matrix

    def initialize_output(self):
        input_path = pathlib.Path(self.args.caster_scores)
        repo_root = pathlib.Path(__file__).parent.parent.resolve()
        test_dir = repo_root / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = test_dir / f"report_{input_path.name}"
        headers = [f"# {' '.join(sys.argv)}"]
        self.output_str = "\n".join(headers)

    def compute_output(self):
        divergence = self.hmm.state_emission_divergence(self.params)
        try:
            emission_divergence_str = ", ".join(map(str, divergence.tolist()))
        except TypeError:
            emission_divergence_str = str(float(divergence))
        most_likely_states = self.hmm.most_likely_states(self.params, self.Y)
        ps = self.hmm.smoother(self.params, self.Y).smoothed_probs[:, 1]
        
        headers = []
        headers.append("# State divergence: " + emission_divergence_str)
        
        self.output_str += "\n" + "\n".join(headers)
        self.output_str += "\n" + ",".join(map(str, most_likely_states.astype(int).tolist()))
        
        # Output histogram counting the number of flagged positions in a given window size if parameter is passed
        hist_str = self.generate_histogram(most_likely_states)
        if hist_str:
            self.output_str += hist_str

    def generate_histogram(self, most_likely_states):
        step_size = self.args.step_size
        if step_size is None:
            return ""

        num_positions = len(most_likely_states)
        step_counts = []
        for start in range(0, num_positions, step_size):
            end = min(start + step_size, num_positions)
            count = int(jnp.sum(most_likely_states[start:end] == 1))
            step_counts.append((start, end, count))

        # 1. Text-based ASCII histogram
        lines = [
            "",
            f"# Step-based counts of flagged positions (step size = {step_size}):",
            "# Range       | Count | Bar",
        ]
        
        max_count = max([c for _, _, c in step_counts]) if step_counts else 0
        bar_max_width = 40
        
        for start, end, count in step_counts:
            bar_len = int((count / max_count) * bar_max_width) if max_count > 0 else 0
            bar = "*" * bar_len
            range_str = f"{start:<6}-{end:<6}"
            lines.append(f"# {range_str} | {count:<5} | {bar}")
            
        text_hist = "\n".join(lines) + "\n"

        # 2. Matplotlib visual plot saving
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            steps = [f"{start}-{end}" for start, end, _ in step_counts]
            counts = [c for _, _, c in step_counts]
            
            bars = ax.bar(steps, counts, color='#4c72b0', edgecolor='none')
            
            ax.set_title(f"Flagged Positions Count per Step (Size = {step_size})", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Windows", fontsize=12, labelpad=10)
            ax.set_ylabel("Count of Flagged (Anomalous) States", fontsize=12, labelpad=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight="semibold")
                                
            plt.tight_layout()
            
            output_path = pathlib.Path(self.output_file)
            input_path = pathlib.Path(self.args.caster_scores)
            plot_path = output_path.with_name(f"states_{input_path.stem}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved visual window states plot to: {plot_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate visual histogram plot: {e}")

        return text_hist

    def initialize_hmm(self):
        # Fixed prior hyperparameters
        self.emission_lambda = 1.0
        self.gamma = 1.1
        self.nu = 1.1
        eta = 0.5
        
        # Generic Dirichlet structure setup for continuous emission tracking
        self.psi = jnp.ones((NUM_STATES, NUM_STATES)) + PSI_EPS
        self.occupancy_bias = jnp.zeros(NUM_STATES)
        self.occupancy_bias = self.occupancy_bias.at[-1].set(
            -jnp.log((1 - eta) / (eta))
        )
        
        emission_parameterization_mode = self.args.emission_parameterization
        self.emission_parameterization = (
            hmm.EmissionParam(emission_parameterization_mode),
        ) + tuple(hmm.EmissionParam("free") for _ in range(NUM_STATES - 1))
        
        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.Y.shape[-1],
            emission_lambda=self.emission_lambda,
            emission_concentration=self.gamma,
            emission_parameterization=self.emission_parameterization,
            initial_probs_concentration=self.nu,
            transition_concentration=self.psi,
            occupancy_bias=self.occupancy_bias,
        )
        
        # Compute empirical moments from CASTER data over genomic positions
        data_mean = jnp.mean(self.Y, axis=0)
        data_std = jnp.std(self.Y, axis=0)
        
        # Implement symmetry breaking: Anchor state 0 to the baseline mean,
        # and seed state 1 slightly further along the alternative dimensions.
        # Initialize emissions using mean and variance for each state.
        state0_init = jnp.stack([data_mean, data_std ** 2], axis=-1)
        state1_init = jnp.stack([data_mean + data_std, data_std ** 2], axis=-1)

        # Shape: [num_states, emission_dim, 2] where the last axis is [mean, variance]
        init_emissions = jnp.stack([state0_init, state1_init], axis=0)
        p0, p1 = 0.2, 0.2
        initial_transition_matrix = jnp.array([[p0, 1-p0], [1-p1, p1]], dtype=jnp.float32)
        
        self.params, self.props = self.hmm.initialize(
            initial_probs=INITIAL_PROBS, emission_probs=init_emissions, transition_matrix=initial_transition_matrix
        )
        self.props.transitions.transition_matrix.trainable = True
        self.props.emissions.means.trainable = True
        self.props.emissions.covariances.trainable = True
        
        self.hmm.initialize_m_step_state(self.params, self.props)
        self.n_iters = self.args.n_iters
        self.increment_steps = self.args.increment_steps
        
    def run(self):
        for i in tqdm(range(self.n_iters)):
            self.params, log_probs = self.hmm.fit_em(
                self.params,
                self.props,
                self.Y,
                num_iters=(i + 1) * self.increment_steps + 1,
                verbose=False,
            )
        self.compute_output()

    def save_output(self):
        with open(self.output_file, "w") as f:
            f.write(self.output_str)


def int_or_abbrev(val_str):
    val_str = str(val_str).strip().lower()
    if val_str.endswith('k'):
        return int(float(val_str[:-1]) * 1000)
    elif val_str.endswith('m'):
        return int(float(val_str[:-1]) * 1000000)
    return int(val_str)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Phlag: Detecting genomic regions with unexplained phylogenetic heterogeneity using CASTER"
    )

    parser.add_argument(
        "-o", dest="output_file", type=pathlib.Path, required=False, default=None, help="Path to save the output"
    )
    parser.add_argument(
        "-l",
        dest="increment_steps",
        type=int_or_abbrev,
        default=50,
        help="Increment for inner EM iterations (default: 50)",
    )

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument(
        "-e",
        dest="emission_parameterization",
        type=str.lower,
        default="attraction",
        choices=["free", "attraction", "anchor"],
        help="Parameterization of the emission probabilities of the default state (default: attraction)",
    )

    discr_group = parser.add_argument_group("Transformation options")
    discr_group.add_argument(
        "-i",
        dest="ilr_transform",
        action="store_true",
        help="Apply isometric log-ratio transformation on CASTER score distributions",
    )

    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "-c",
        dest="caster_scores",
        type=pathlib.Path,
        required=True,
        help="Path to the CASTER scores TSV",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    phlag = Phlag(args)
    phlag.run()
    phlag.save_output()


if __name__ == "__main__":
    main()