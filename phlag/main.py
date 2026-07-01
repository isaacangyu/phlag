import sys
import pathlib
import os
import argparse

import jax
import dendropy
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax.distributions as tfd

from collections import defaultdict
from functools import partial
from io import StringIO
from tqdm import tqdm
from skbio.stats.composition import ilr, multi_replace

from . import hmm
from . import utils

E_STEP_EPS = 0.0001
PSI_EPS = 0.001
NUM_STATES = 2
MAX_INCIDENT_LENGTH = 2.0
BETA_PRIME = 0.0025
INITIAL_PROBS = jnp.array([1.0000, 0.0000], dtype=jnp.float32)


class Phlag:
    def __init__(self, args):
        self.args = args

        self.read_species_tree()
        self.validate_parameters()
        self.determine_focal_edges()
        self.st.deroot()
        self.st.encode_bipartitions()
        
        # Purely ingest CASTER scores instead of computing or reading QQS
        self.read_caster_scores(self.args.caster_scores)
        
        self.configure_emissions()
        self.compute_emissions()
        self.initialize_hmm()
        self.initialize_output()

    def read_species_tree(self):
        self.taxa = utils.get_canonical_taxon_namespace(self.args.species_tree)
        self.st = dendropy.Tree.get(
            path=self.args.species_tree,
            schema="newick",
            preserve_underscores=True,
            taxon_namespace=self.taxa,
        )
        self.st.suppress_unifurcations()
        self.st.collapse_basal_bifurcation()
        self.st.encode_bipartitions(
            collapse_unrooted_basal_bifurcation=True, suppress_unifurcations=True
        )
        self.lbl_to_nd = utils.map_label_to_node(self.st)

    def validate_parameters(self):
        if self.args.beta is None:
            raise ValueError("--beta hyperparameter must be explicitly set when running alignment mode.")
        if not (0 < self.args.rho < 1):
            raise ValueError(f"--rho must be in (0, 1), got {self.args.rho}")
        if self.args.beta <= 0:
            raise ValueError(f"--beta must be positive, got {self.args.beta}")
        if not (0 < self.args.eta < 1):
            raise ValueError(f"--eta must be in (0, 1), got {self.args.eta}")
        if self.args.n_iters < 1:
            raise ValueError(f"--n-iters must be >= 1, got {self.args.n_iters}")
        for lbl in self.args.focal_edges:
            if lbl not in self.lbl_to_nd:
                raise ValueError(f"Focal edge label '{lbl}' not found in species tree")

    def determine_focal_edges(self):
        if self.args.focal_edges:
            self.focal_edges = []
            for lbl in self.args.focal_edges:
                edge = utils.focal_edge_from_label(self.st, lbl, self.lbl_to_nd)
                if self.args.expand_edges:
                    self.focal_edges.extend(
                        incident
                        for incident in utils.get_incident_edges(self.st, edge, self.lbl_to_nd)
                        if incident.length < MAX_INCIDENT_LENGTH or incident == edge
                    )
                else:
                    self.focal_edges.append(edge)
        self.num_edges = len(self.focal_edges)

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
            
        self.num_classes = self.Y.shape[-1]

    def focal_edge_lengths(self):
        return [edge.head_node.label + ": " + str(edge.length) for edge in self.focal_edges]

    def initialize_output(self):
        self.output_file = self.args.output_file
        self.output_str = f"# {' '.join(sys.argv)}"
        self.output_str += "\n# Initial tree: " + self.st.as_string(schema="newick")
        self.output_str += "# Initial focal edge lengths: " + ", ".join(self.focal_edge_lengths())

    def compute_output(self):
        emission_divergence_str = ", ".join(
            list(map(lambda x: str(x), self.hmm.state_emission_divergence(self.params).tolist()))
        )
        most_likely_states = self.hmm.most_likely_states(self.params, self.Y)
        ps = self.hmm.smoother(self.params, self.Y).smoothed_probs[:, 1]
        
        self.output_str += "\n# Final tree: " + self.st.as_string(schema="newick")
        self.output_str += "# Final focal edge lengths: " + ", ".join(self.focal_edge_lengths())
        self.output_str += "\n# State divergence: " + emission_divergence_str
        self.output_str += "\n" + ",".join(map(str, most_likely_states.astype(int).tolist()))
        self.output_str += "\n" + ",".join(map(lambda x: str(x), jnp.round(ps, decimals=3).tolist()))

    def initialize_hmm(self):
        self.emission_lambda = self.args.emission_lambda
        self.gamma = self.args.emission_concentration
        self.nu = self.args.initial_probs_concentration
        
        # Generic Dirichlet structure setup for continuous emission tracking
        self.psi = jnp.ones((NUM_STATES, NUM_STATES)) + PSI_EPS
        self.occupancy_bias = jnp.zeros(NUM_STATES)
        self.occupancy_bias = self.occupancy_bias.at[-1].set(
            -jnp.log((1 - self.args.eta) / (self.args.eta))
        )
        
        self.emission_parameterization = (
            hmm.EmissionParam(self.args.emission_parameterization),
        ) + tuple(hmm.EmissionParam("free") for _ in range(NUM_STATES - 1))
        
        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.num_edges,
            self.num_classes,
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
        # and seed state 1 slightly further along the alternative dimensions
        state0_init = data_mean
        state1_init = data_mean + (1.0 * data_std)
        
        # Stack the perturbed matrices to form distinct starting emission spaces
        init_emissions = jnp.stack([state0_init, state1_init], axis=0)
        
        self.params, self.props = self.hmm.initialize(
            initial_probs=INITIAL_PROBS, emission_probs=init_emissions
        )
        self.props.transitions.transition_matrix.trainable = True
        self.props.emissions.probs.trainable = True
        self.props.initial.probs.trainable = True
        
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Phlag: Detecting genomic regions with unexplained phylogenetic heterogeneity using CASTER"
    )

    parser.add_argument(
        "-s",
        "--species-tree",
        type=pathlib.Path,
        required=True,
        help="Path to species tree in Newick format",
    )
    parser.add_argument(
        "-o", "--output-file", type=pathlib.Path, required=True, help="Path to save the output"
    )
    parser.add_argument(
        "-L", "--n-iters", type=int, default=5, help="Number of (outer) iterations (default: 5)"
    )
    parser.add_argument(
        "-l",
        "--increment-steps",
        type=int,
        default=50,
        help="Increment for inner EM iterations (default: 50)",
    )
    parser.add_argument(
        "-e",
        "--focal-edges",
        nargs="+",
        type=str,
        required=True,
        help="Focal edge(s) specified by inner node label(s).",
    )
    parser.add_argument(
        "--expand-edges",
        action="store_true",
        help="Incorporate the signal from neighboring/incident edges.",
    )

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument(
        "--rho",
        type=float,
        default=0.9,
        help="Hyperparameter to control sensitivity (default 0.9)",
    )
    hmm_group.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="Hyperparameter to control contiguity of flagged regions (default: 5.0)",
    )
    hmm_group.add_argument(
        "--emission-lambda",
        "--lambda",
        type=float,
        default=1.0,
        help="Hyperparameter to control deviation of anomalies from baseline (default: 1.0)",
    )
    hmm_group.add_argument(
        "--initial-probs-concentration",
        type=float,
        default=1.1,
        help="Initial probabilities concentration (default: 1.1)",
    )
    hmm_group.add_argument(
        "--emission-concentration",
        type=float,
        default=1.1,
        help="Emission prior concentration (default: 1.1)",
    )
    hmm_group.add_argument(
        "--eta",
        "--occupancy-bias",
        type=float,
        default=0.5,
        help="A global occupancy penalty on the marginal log-likelihood (default: 0.5)",
    )
    hmm_group.add_argument(
        "--emission-parameterization",
        type=str.lower,
        default="attraction",
        choices=["free", "attraction", "anchor"],
        help="Parameterization of the emission probabilities of the default state (default: attraction)",
    )

    discr_group = parser.add_argument_group("Transformation options")
    discr_group.add_argument(
        "--ilr-transform",
        action="store_true",
        help="Apply isometric log-ratio transformation on CASTER score distributions",
    )

    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "-c",
        "--caster-scores",
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