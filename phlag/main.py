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
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as transforms

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
        if not hasattr(self.args, "step_size"):
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

            header_parts = header.strip().split("\t") if "\t" in header else header.strip().split()
            if header_parts and header_parts[0].lower() == "file":
                pos_idx = 1
                score_indices = [2, 3, 4]
            else:
                pos_idx = 0
                score_indices = [1, 2, 3]

            for line in f:
                if not line.strip():
                    continue
                
                values = line.strip().split("\t") if "\t" in line else line.strip().split()
                
                try:
                    pos_key = int(values[pos_idx])
                    scores = jnp.array([
                        float(values[score_indices[0]]), 
                        float(values[score_indices[1]]), 
                        float(values[score_indices[2]])
                    ], dtype=jnp.float32)
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
        if self.args.output_file:
            self.output_file = self.args.output_file
            # Ensure the output directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            test_dir = pathlib.Path.cwd() / "test"
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
        headers.append(f"# Outer EM iterations: {self.n_iters}")
        headers.append(f"# Inner EM iterations: {self.increment_steps}")
        headers.append(f"# ILR transform: {self.ilr_transform}")
        
        self.output_str += "\n" + "\n".join(headers)
        self.output_str += "\n" + ",".join(map(str, most_likely_states.astype(int).tolist()))
        
        # Output histogram counting the number of flagged positions in a given window size if parameter is passed
        hist_str = self.generate_histogram(most_likely_states)
        if hist_str:
            self.output_str += hist_str

    def generate_histogram(self, most_likely_states):
        step_size = self.args.step_size
        if step_size is None:
            step_size = 100

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
        if self.args.plot and "states" in self.args.plot:
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
        # Prior hyperparameters
        self.emission_lambda = self.args.emission_lambda
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
            emission_parameterization_mode,
        ) + tuple("free" for _ in range(NUM_STATES - 1))
        
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


class PhlagPlotter:
    """
    Object-oriented plotter for visualizing the probability distributions
    of standard multi-species coalescent (MSC) background (State 0, Null) vs.
    alternative/anomalous (State 1, Alternative) states before and after EM fitting.
    """
    def __init__(self, phlag, initial_means, initial_covariances):
        self.phlag = phlag
        self.initial_means = initial_means
        self.initial_covariances = initial_covariances
        
        # Ingest and configure metadata parameters
        self.extract_metadata()
        
        # Generate the visual distribution charts
        self.plot_distributions()

    def extract_metadata(self):
        """Extracts genomic filename, dimension, and styles configuration."""
        self.input_path = pathlib.Path(self.phlag.args.caster_scores)
        self.emission_dim = self.phlag.Y.shape[-1]
        
        # Map coordinates to topology names if we have the standard 3 topologies
        if not self.phlag.ilr_transform and self.emission_dim == 3:
            self.topology_names = ["ABBA", "BABA", "AABB"]
        else:
            self.topology_names = [
                f"ILR Coord {i+1}" if self.phlag.ilr_transform else f"Coord {i+1}" 
                for i in range(self.emission_dim)
            ]

        # Define premium color palette and labels
        self.colors = {
            0: {
                'line': '#2B4C7E',    # Deep Steel Blue
                'fill': '#2B4C7E',
                'label': 'State 0 (Null)'
            },
            1: {
                'line': '#E05A47',    # Warm Coral
                'fill': '#E05A47',
                'label': 'State 1 (Alternative)'
            }
        }

    def plot_distributions(self):
        """Prepares the subplot layout and runs plotting for Before and After EM states."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
            'axes.edgecolor': '#cccccc',
            'grid.color': '#f0f0f0'
        })
        
        # Create a 2x3 panel layout (or 2x1 if single dimension)
        fig, axes = plt.subplots(2, self.emission_dim, figsize=(5 * self.emission_dim, 9.0), sharey=False)
        
        if self.emission_dim == 1:
            axes = np.array([[axes[0]], [axes[1]]])
            
        # Determine consistent coordinate limits across Before/After plots
        ranges = {}
        for d in range(self.emission_dim):
            ymin, ymax = float(np.min(self.phlag.Y[:, d])), float(np.max(self.phlag.Y[:, d]))
            ypad = (ymax - ymin) * 0.20 or 0.1
            ranges[d] = np.linspace(ymin - ypad, ymax + ypad, 300)

        # Plot Row 0: Before EM (Initial theoretical setup)
        self._plot_row(axes[0], self.initial_means, self.initial_covariances, ranges, title_prefix="Before EM", plot_empirical=False)
        
        # Plot Row 1: After EM (Fitted theoretical setup and assigned empirical data)
        final_means = np.array(self.phlag.params.emissions.means)
        final_covariances = np.array(self.phlag.params.emissions.covariances)
        self._plot_row(axes[1], final_means, final_covariances, ranges, title_prefix="After EM", plot_empirical=True)
        
        plt.tight_layout()
        self.save_plot()

    def _plot_row(self, row_axes, means, covariances, ranges, title_prefix, plot_empirical=False):
        """Plots a single row (either Before EM or After EM) across all topologies."""
        if plot_empirical:
            most_likely_states = self.phlag.hmm.most_likely_states(self.phlag.params, self.phlag.Y)
            
        for d in range(self.emission_dim):
            ax = row_axes[d]
            x_vals = ranges[d]
            
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            
            # 1. Plot empirical step-histograms for Assigned State Data points
            if plot_empirical:
                y_state0 = np.array(self.phlag.Y[most_likely_states == 0, d])
                y_state1 = np.array(self.phlag.Y[most_likely_states == 1, d])
                
                if len(y_state0) > 0:
                    sns.histplot(
                        y_state0, ax=ax, color=self.colors[0]['fill'], 
                        stat="density", kde=False, alpha=0.12, 
                        element="step", label=f"{self.colors[0]['label']} data"
                    )
                if len(y_state1) > 0:
                    sns.histplot(
                        y_state1, ax=ax, color=self.colors[1]['fill'], 
                        stat="density", kde=False, alpha=0.12, 
                        element="step", label=f"{self.colors[1]['label']} data"
                    )
            
            # 2. Plot Gaussian PDF curves and Vertical Guideline Markers (Mean and +/- 1 Std)
            for state in [0, 1]:
                mu = means[state, d]
                sigma = np.sqrt(np.clip(covariances[state, d, d], a_min=1e-6, a_max=None))
                
                pdf_vals = stats.norm.pdf(x_vals, mu, sigma)
                
                color_config = self.colors[state]
                
                # Plot theoretical normal curve
                ax.plot(
                    x_vals, pdf_vals, color=color_config['line'], 
                    linewidth=2.2, label=f"{color_config['label']} PDF"
                )
                
                # Shading under curve
                ax.fill_between(
                    x_vals, pdf_vals, alpha=0.05, 
                    color=color_config['fill']
                )
                
                # Plot Central Tendency Guideline: Mean (E[X])
                ax.axvline(
                    x=mu, color=color_config['line'], linestyle='--', linewidth=1.5, alpha=0.8,
                    label=None
                )
                
                # Plot Dispersion Guidelines: +/- 1 Std bounds
                ax.axvline(
                    x=mu - sigma, color=color_config['line'], linestyle=':', linewidth=1.0, alpha=0.6,
                    label=None
                )
                ax.axvline(
                    x=mu + sigma, color=color_config['line'], linestyle=':', linewidth=1.0, alpha=0.6,
                    label=None
                )
                
                # Label with symbols mu and sigma next to the lines
                y_pos_mean = 0.90 if state == 0 else 0.75
                y_pos_std = 0.83 if state == 0 else 0.68
                
                ax.text(
                    mu, y_pos_mean, f"$\\mu_{state} = {mu:.4f}$", transform=trans, color=color_config['line'],
                    fontsize=8.0, ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
                )
                ax.text(
                    mu + sigma, y_pos_std, f"$\\sigma_{state} = {sigma:.4f}$", transform=trans, color=color_config['line'],
                    fontsize=7.0, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
                )
                
            ax.set_title(f"{title_prefix} | Topology: {self.topology_names[d]}", fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel("Topology Score", fontsize=9, labelpad=4)
            ax.set_ylabel("Probability Density", fontsize=9, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # De-duplicate legend entries to keep layout clean and readable
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels[label] = handle
                    
            ax.legend(
                unique_labels.values(), unique_labels.keys(), 
                fontsize=7.5, loc='upper right', framealpha=0.9
            )

    def save_plot(self):
        """Saves generated plot as PNG."""
        if self.phlag.args.output_file:
            output_dir = pathlib.Path(self.phlag.args.output_file).parent
        else:
            output_dir = pathlib.Path.cwd() / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            
        suffix = "_ilr" if self.phlag.ilr_transform else ""
        plot_file = output_dir / f"em_{self.input_path.stem}{suffix}.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visual distributions plot to: {plot_file}")


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
        "caster_scores",
        type=pathlib.Path,
        help="Path to the CASTER scores TSV"
    )

    parser.add_argument(
        "-o", dest="output_file", type=pathlib.Path, required=False, default=None, help="Path to save the output"
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=["em", "states"],
        default=["em"],
        help="List of plots to generate (choices: em, states. Default: em)",
    )
    parser.add_argument(
        "-l",
        dest="increment_steps",
        type=int_or_abbrev,
        default=50,
        help="Increment for inner EM iterations (default: 50)",
    )
    parser.add_argument(
        "-s",
        dest="step_size",
        type=int_or_abbrev,
        required=False,
        default=None,
        help="Genomic step size (in rows/positions) to compute a text-based ASCII histogram and save a visual bar chart plot",
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
    hmm_group.add_argument(
        "-L",
        dest="emission_lambda",
        type=float,
        default=1.0,
        help="Emission penalty regularizer parameter lambda (default: 1.0)",
    )

    discr_group = parser.add_argument_group("Transformation options")
    discr_group.add_argument(
        "-i",
        dest="ilr_transform",
        action="store_true",
        help="Apply isometric log-ratio transformation on CASTER score distributions",
    )

    args = parser.parse_args()

    # Check if --plot is supplied
    plot_supplied = any(arg == "--plot" or arg.startswith("--plot=") for arg in sys.argv)
    if plot_supplied and args.step_size is None:
        parser.error("argument -s/--step-size is required if --plot is supplied")

    return args


def main():
    args = parse_arguments()
    phlag = Phlag(args)
    
    # Ingest baseline setup properties before fitting
    initial_means = np.array(phlag.params.emissions.means)
    initial_covariances = np.array(phlag.params.emissions.covariances)
    
    phlag.run()
    phlag.save_output()
    
    if args.plot and "em" in args.plot:
        PhlagPlotter(phlag, initial_means, initial_covariances)


if __name__ == "__main__":
    main()