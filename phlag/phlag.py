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
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import hmm
from . import utils

E_STEP_EPS = 0.0001
PSI_EPS = 0.001
NUM_STATES = 2
BETA_PRIME = 0.0025
INITIAL_PROBS = jnp.array([1.0000, 0.0000], dtype=jnp.float32)


def get_topology_names(dim):
    """Maps an emission dimension count to human-readable topology labels."""
    if dim == 3:
        return ["ABBA", "BABA", "AABB"]
    return [f"Coord {i+1}" for i in range(dim)]


def get_state_mu_sigma_pdf(params, model_design, state, dim, x_vals):
    """Extracts (mu, sigma, pdf_vals) for a given HMM state/dimension, branching on model_design."""
    if model_design == "gmm":
        w = np.array(params.emissions.mixture_weights[state, dim])
        m_means = np.array(params.emissions.means[state, dim])
        m_stds = np.array(params.emissions.stds[state, dim])
        pdf_vals = np.zeros_like(x_vals)
        for m in range(len(w)):
            pdf_vals += w[m] * stats.norm.pdf(x_vals, m_means[m], m_stds[m])
        mu = float(np.sum(w * m_means))
        var = float(np.sum(w * (m_stds ** 2 + m_means ** 2)) - mu ** 2)
        sigma = np.sqrt(max(1e-6, var))
    else:
        mu = float(params.emissions.means[state, dim])
        sigma = np.sqrt(np.clip(float(params.emissions.covariances[state, dim, dim]), a_min=1e-6, a_max=None))
        pdf_vals = stats.norm.pdf(x_vals, mu, sigma)
    return mu, sigma, pdf_vals


def format_rel_err(value):
    """Formats a relative-error percentage to 4 decimals."""
    return f"{value:.4f}"


def count_trainable_params(params, props):
    """
    Total scalar element count across every leaf marked trainable=True in props,
    for a BIC parameter penalty. Counts declared array sizes (e.g. the 2x2
    transition matrix counts as 4), not row-stochastic-constrained degrees of
    freedom, so this is an upper bound on true DOF rather than an exact count.
    """
    from dynamax.parameters import ParameterProperties

    is_leaf = lambda node: isinstance(node, ParameterProperties)

    def _count(p, prop):
        return int(np.prod(p.shape)) if prop.trainable else 0

    counts = jax.tree_util.tree_map(_count, params, props, is_leaf=is_leaf)
    return int(sum(jax.tree_util.tree_leaves(counts)))


def determine_optimal_mixtures(caster_scores_path, Y, pos_to_caster, silhouette_threshold, output_dir, cluster_topologies=False):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_positions = sorted(pos_to_caster.keys())
    positions_kb = np.array(sorted_positions) / 1000.0

    num_mixtures_matrix = np.zeros((NUM_STATES, Y.shape[-1]), dtype=int)
    dim_cluster_info = {}

    print("\n=== K-means Clustering & Silhouette Scores ===")

    topology_names = ["ABBA", "BABA", "AABB"] if Y.shape[-1] == 3 else [f"Coord_{i+1}" for i in range(Y.shape[-1])]

    dims_to_iterate = [None] if cluster_topologies else list(range(Y.shape[-1]))

    for d in dims_to_iterate:
        if d is None:
            y_d = Y
            topo_name = "All"
            y_plot = np.linalg.norm(Y, axis=1)
        else:
            y_d = np.array(Y[:, [d]])
            topo_name = topology_names[d]
            y_plot = y_d.ravel()

        k_values = list(range(2, 11))
        scores = {}
        k_opt = 2
        max_score = -2.0

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(y_d)
            score = silhouette_score(y_d, labels)
            scores[k] = score
            if score > max_score:
                max_score = score
                k_opt = k

        s_2 = scores[2]

        # Print global table
        print(f"\nTopology / Dimension: {topo_name}")
        print(f"{'k':<5} | {'Global Silhouette Score':<25}")
        print("-" * 35)
        for k in k_values:
            marker = " *" if k == k_opt else ""
            print(f"{k:<5} | {scores[k]:<25.6f}{marker}")
        print(f"Global Optimal k* = {k_opt} (Score: {max_score:.6f})")
        print(f"Global k=2 Score = {s_2:.6f}")

        if s_2 > silhouette_threshold:
            print(f"s_2 ({s_2:.4f}) > threshold ({silhouette_threshold:.4f}) -> Partitioning into Null and Alternative clusters:")

            # Fit 2-means to partition the data
            kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
            labels_2 = kmeans_2.fit_predict(y_d)

            # Assign cluster labels to Null vs Alternative based on distance from global mean
            global_mean = np.mean(y_d, axis=0)
            mean_c0 = np.mean(y_d[labels_2 == 0], axis=0)
            mean_c1 = np.mean(y_d[labels_2 == 1], axis=0)

            dist_c0 = np.linalg.norm(mean_c0 - global_mean)
            dist_c1 = np.linalg.norm(mean_c1 - global_mean)

            if dist_c0 < dist_c1:
                null_lbl, alt_lbl = 0, 1
            else:
                null_lbl, alt_lbl = 1, 0

            y_sub_N = y_d[labels_2 == null_lbl]
            pos_sub_N = positions_kb[labels_2 == null_lbl]
            y_sub_A = y_d[labels_2 == alt_lbl]
            pos_sub_A = positions_kb[labels_2 == alt_lbl]

            n_N = len(y_sub_N)
            n_A = len(y_sub_A)


            # Search for best split kn + ka = k_opt minimizing total within-cluster sum of squares (inertia)
            best_kn = 1
            best_ka = k_opt - 1
            min_inertia = float('inf')

            print(f"\nEvaluating partitions (kn + ka = k* = {k_opt}):")
            print(f"{'Split':<12} | {'Null Inertia':<15} | {'Alt Inertia':<15} | {'Total Inertia':<15}")
            print("-" * 65)

            for kn in range(1, k_opt):
                ka = k_opt - kn
                if kn > n_N or ka > n_A:
                    continue

                if kn == 1:
                    w_n = float(np.sum((y_sub_N - np.mean(y_sub_N, axis=0)) ** 2))
                else:
                    km_n = KMeans(n_clusters=kn, random_state=42, n_init="auto")
                    km_n.fit(y_sub_N)
                    w_n = float(km_n.inertia_)

                if ka == 1:
                    w_a = float(np.sum((y_sub_A - np.mean(y_sub_A, axis=0)) ** 2))
                else:
                    km_a = KMeans(n_clusters=ka, random_state=42, n_init="auto")
                    km_a.fit(y_sub_A)
                    w_a = float(km_a.inertia_)

                total_w = w_n + w_a
                marker = ""
                if total_w < min_inertia:
                    min_inertia = total_w
                    best_kn = kn
                    best_ka = ka
                    marker = " *"

                print(f"{kn:<2} + {ka:<2} = {k_opt:<2}  | {w_n:<15.6f} | {w_a:<15.6f} | {total_w:<15.6f}{marker}")

            print(f"Optimal split: Null count = {best_kn}, Alternative count = {best_ka} (Inertia: {min_inertia:.6f})")

            if d is None:
                num_mixtures_matrix[0, :] = best_kn
                num_mixtures_matrix[1, :] = best_ka
            else:
                num_mixtures_matrix[0, d] = best_kn
                num_mixtures_matrix[1, d] = best_ka

            # Save Null sub-cluster plot if kn > 1
            if best_kn > 1:
                sub_kmeans = KMeans(n_clusters=best_kn, random_state=42, n_init="auto")
                sub_labels = sub_kmeans.fit_predict(y_sub_N)

                plt.figure(figsize=(10, 5))
                sns.set_theme(style="whitegrid")
                sns.scatterplot(x=pos_sub_N, y=y_plot[labels_2 == null_lbl], hue=sub_labels, palette="tab10", alpha=0.8, legend="full")
                plt.title(f"{topo_name} | Null Sub-cluster {best_kn}-means Clustering", fontsize=12, fontweight='bold')
                plt.xlabel("Position (kb)", fontsize=10)
                plt.ylabel("Normalized score", fontsize=10)
                plt.tight_layout()
                plot_path_sub_N = output_dir / f"kmeans_kstar_{topo_name}_null.png"
                plt.savefig(plot_path_sub_N, dpi=150)
                plt.close()
                print(f"Saved Null sub-cluster optimal plot to: {plot_path_sub_N}")

            # Save Alternative sub-cluster plot if ka > 1
            if best_ka > 1:
                sub_kmeans = KMeans(n_clusters=best_ka, random_state=42, n_init="auto")
                sub_labels = sub_kmeans.fit_predict(y_sub_A)

                plt.figure(figsize=(10, 5))
                sns.set_theme(style="whitegrid")
                sns.scatterplot(x=pos_sub_A, y=y_plot[labels_2 == alt_lbl], hue=sub_labels, palette="tab10", alpha=0.8, legend="full")
                plt.title(f"{topo_name} | Alternative Sub-cluster {best_ka}-means Clustering", fontsize=12, fontweight='bold')
                plt.xlabel("Position (kb)", fontsize=10)
                plt.ylabel("Normalized score", fontsize=10)
                plt.tight_layout()
                plot_path_sub_A = output_dir / f"kmeans_kstar_{topo_name}_alternative.png"
                plt.savefig(plot_path_sub_A, dpi=150)
                plt.close()
                print(f"Saved Alternative sub-cluster optimal plot to: {plot_path_sub_A}")

            # Save combined optimal plot (kmeans_kstar_{topo_name}.png)
            plt.figure(figsize=(10, 5))
            sns.set_theme(style="whitegrid")
            sns.scatterplot(x=positions_kb, y=y_plot, hue=labels_2, palette="tab10", alpha=0.8, legend="full")
            plt.title(f"{topo_name} | Optimal GMM Mixture Partitions (Null count={best_kn}, Alt count={best_ka})", fontsize=11, fontweight='bold')
            plt.xlabel("Position (kb)", fontsize=10)
            plt.ylabel("Normalized score", fontsize=10)
            plt.tight_layout()
            plot_path_opt = output_dir / f"kmeans_kstar_{topo_name}.png"
            plt.savefig(plot_path_opt, dpi=150)
            plt.close()
            print(f"Saved combined optimal plot to: {plot_path_opt}")

            # Compute Null sub-cluster mixture parameters
            null_params = []
            if best_kn == 1:
                mu_n = np.mean(y_sub_N, axis=0)
                std_n = np.clip(np.std(y_sub_N, axis=0), a_min=1e-4, a_max=None)
                null_params.append((1.0, mu_n, std_n))
            else:
                km_n = KMeans(n_clusters=best_kn, random_state=42, n_init="auto")
                sub_labels_n = km_n.fit_predict(y_sub_N)
                for m in range(best_kn):
                    pts = y_sub_N[sub_labels_n == m]
                    if len(pts) > 0:
                        w = len(pts) / len(y_sub_N)
                        mu = np.mean(pts, axis=0)
                        std = np.clip(np.std(pts, axis=0), a_min=1e-4, a_max=None)
                    else:
                        w = 1.0 / best_kn
                        mu = km_n.cluster_centers_[m]
                        std = np.full(y_d.shape[-1], 1e-4)
                    null_params.append((w, mu, std))

            # Compute Alt sub-cluster mixture parameters
            alt_params = []
            if best_ka == 1:
                mu_a = np.mean(y_sub_A, axis=0)
                std_a = np.clip(np.std(y_sub_A, axis=0), a_min=1e-4, a_max=None)
                alt_params.append((1.0, mu_a, std_a))
            else:
                km_a = KMeans(n_clusters=best_ka, random_state=42, n_init="auto")
                sub_labels_a = km_a.fit_predict(y_sub_A)
                for m in range(best_ka):
                    pts = y_sub_A[sub_labels_a == m]
                    if len(pts) > 0:
                        w = len(pts) / len(y_sub_A)
                        mu = np.mean(pts, axis=0)
                        std = np.clip(np.std(pts, axis=0), a_min=1e-4, a_max=None)
                    else:
                        w = 1.0 / best_ka
                        mu = km_a.cluster_centers_[m]
                        std = np.full(y_d.shape[-1], 1e-4)
                    alt_params.append((w, mu, std))

            dim_cluster_info[d] = (null_params, alt_params)

        else:
            m_val = max(1, int(round(k_opt / 2.0)))
            print(f"s_2 ({s_2:.4f}) <= threshold ({silhouette_threshold:.4f}) -> Use k*/2 = {m_val} mixtures for both states.")
            if d is None:
                num_mixtures_matrix[0, :] = m_val
                num_mixtures_matrix[1, :] = m_val
            else:
                num_mixtures_matrix[0, d] = m_val
                num_mixtures_matrix[1, d] = m_val

            # Save ONLY 2-means plot, NOT kmeans_kstar plot!
            kmeans_plot = KMeans(n_clusters=2, random_state=42, n_init="auto")
            labels_plot = kmeans_plot.fit_predict(y_d)

            plt.figure(figsize=(10, 5))
            sns.set_theme(style="whitegrid")
            sns.scatterplot(x=positions_kb, y=y_plot, hue=labels_plot, palette="tab10", alpha=0.8, legend="full")
            plt.title(f"{topo_name} | 2-means Clustering", fontsize=12, fontweight='bold')
            plt.xlabel("Position (kb)", fontsize=10)
            plt.ylabel("Normalized score", fontsize=10)
            plt.legend(title="Cluster")
            plt.tight_layout()

            plot_path = output_dir / f"kmeans_2_{topo_name}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved diagnostic clustering plot to: {plot_path}")

            null_params = []
            alt_params = []
            if m_val == 1:
                mu_base = np.mean(y_d, axis=0)
                std_base = np.clip(np.std(y_d, axis=0), a_min=1e-4, a_max=None)
                null_params.append((1.0, mu_base, std_base))
                alt_params.append((1.0, mu_base + std_base, std_base))
            else:
                km = KMeans(n_clusters=m_val, random_state=42, n_init="auto")
                labels_m = km.fit_predict(y_d)
                for m in range(m_val):
                    pts = y_d[labels_m == m]
                    if len(pts) > 0:
                        w = len(pts) / len(y_d)
                        mu = np.mean(pts, axis=0)
                        std = np.clip(np.std(pts, axis=0), a_min=1e-4, a_max=None)
                    else:
                        w = 1.0 / m_val
                        mu = km.cluster_centers_[m]
                        std = np.full(y_d.shape[-1], 1e-4)
                    null_params.append((w, mu, std))
                    alt_params.append((w, mu + std, std))

            dim_cluster_info[d] = (null_params, alt_params)

    D = Y.shape[-1]
    max_m = int(np.max(num_mixtures_matrix))
    init_means = np.zeros((NUM_STATES, D, max_m), dtype=np.float32)
    init_stds = np.ones((NUM_STATES, D, max_m), dtype=np.float32)
    init_weights = np.zeros((NUM_STATES, D, max_m), dtype=np.float32)

    if cluster_topologies:
        null_params, alt_params = dim_cluster_info[None]
        for s_idx, state_params in enumerate([null_params, alt_params]):
            m_cnt = len(state_params)
            for m_idx, (w, mu, std) in enumerate(state_params):
                for d_idx in range(D):
                    init_weights[s_idx, d_idx, m_idx] = w
                    init_means[s_idx, d_idx, m_idx] = mu[d_idx]
                    init_stds[s_idx, d_idx, m_idx] = std[d_idx]
            for d_idx in range(D):
                sum_w = np.sum(init_weights[s_idx, d_idx, :m_cnt])
                if sum_w > 0:
                    init_weights[s_idx, d_idx, :m_cnt] /= sum_w
    else:
        for d_idx in range(D):
            null_params, alt_params = dim_cluster_info[d_idx]
            for s_idx, state_params in enumerate([null_params, alt_params]):
                m_cnt = len(state_params)
                for m_idx, (w, mu, std) in enumerate(state_params):
                    init_weights[s_idx, d_idx, m_idx] = w
                    init_means[s_idx, d_idx, m_idx] = mu[0]
                    init_stds[s_idx, d_idx, m_idx] = std[0]
                sum_w = np.sum(init_weights[s_idx, d_idx, :m_cnt])
                if sum_w > 0:
                    init_weights[s_idx, d_idx, :m_cnt] /= sum_w

    gmm_init_params = (init_weights, init_means, init_stds)
    return num_mixtures_matrix, gmm_init_params


class Phlag:
    def __init__(self, args):
        self.args = args
        if not hasattr(self.args, "n_iters"):
            self.args.n_iters = 10
        self.args.increment_steps = 5
        if not hasattr(self.args, "step_size"):
            self.args.step_size = None

        # Auto-extract model_design from filename if not explicitly passed
        self.extract_distribution_type_from_filename()

        self.validate_parameters()
        
        # Purely ingest CASTER scores instead of computing or reading QQS
        self.read_caster_scores(self.args.caster_scores)
        
        self.compute_emissions()
        if getattr(self.args, "model_design", "gaussian") == "gmm":
            self.determine_optimal_mixtures()
        self.initialize_hmm()
        self.initialize_output()

    def extract_distribution_type_from_filename(self):
        # Check if the user specified model_design explicitly on CLI
        T_supplied = any(arg.startswith("-d") or arg.startswith("--emission-type") for arg in sys.argv)
        if not T_supplied:
            # Check input filename (caster_scores) and output filename
            filenames_to_check = []
            if hasattr(self.args, "caster_scores") and self.args.caster_scores:
                filenames_to_check.append(str(self.args.caster_scores))
            if hasattr(self.args, "output_file") and self.args.output_file:
                filenames_to_check.append(str(self.args.output_file))
                
            for fname in filenames_to_check:
                fname_lower = os.path.basename(fname).lower()
                if "gmm" in fname_lower:
                    self.args.model_design = "gmm"
                    break
                elif "gaussian" in fname_lower:
                    self.args.model_design = "gaussian"
                    break

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
        from .utils import resolve_input_file
        path_obj = pathlib.Path(path)
        resolved = resolve_input_file(path_obj, default_subdirs=["scores", "msa"], default_exts=[".tsv", ".txt"])
        if resolved and resolved.exists():
            path_obj = resolved

        path = str(path_obj)
        if hasattr(self, "args") and self.args:
            self.args.caster_scores = path_obj

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

        if len(self.pos_to_caster) == 0:
            sys.exit(f"Error: No valid window scores parsed from CASTER score file '{path}'. Please check that sequence headers in the FASTA match the species in the mapping file.")

    def compute_emissions(self):
        # Convert CASTER scores dictionary values into sequential matrix positions
        sorted_positions = sorted(self.pos_to_caster.keys())
        raw_caster_matrix = jnp.stack([self.pos_to_caster[pos] for pos in sorted_positions], axis=0)
        self.Y = raw_caster_matrix

    def get_default_out_dir(self):
        input_path = pathlib.Path(self.args.caster_scores)
        dist_type = getattr(self.args, "model_design", "gaussian")
        from .utils import parse_filename_to_dir_structure, get_repo_root, get_data_dir, get_phlag_output_base
        parsed = parse_filename_to_dir_structure(input_path.stem)
        
        conn_env = os.environ.get("CONNECTION_DIR")
        if conn_env:
            base_dir = pathlib.Path(conn_env)
        else:
            base_dir = get_data_dir()
            if base_dir == get_repo_root() / "caster" / "data":
                base_dir = get_repo_root() / "connection_dir"
        
        phlag_base = get_phlag_output_base(base_dir)
        
        if parsed:
            rel_dir = parsed["relative_dir"]
            out_dir = phlag_base / dist_type / rel_dir / "phlag"
        else:
            if input_path.parent.name == "caster":
                # Find path relative to model design or phlag root if input is in a caster subfolder
                # e.g., .../phlag/gaussian/w50k_s1k/admixture/high/Columbiformes_.../45-55/caster/scores.tsv
                # should preserve w50k_s1k/admixture/high/Columbiformes_.../45-55 under phlag/<dist_type>/
                locus_dir = input_path.parent.parent
                parts = locus_dir.parts
                # Check if model design or 'phlag' is in parent path parts
                known_models = ["gaussian", "gmm"]
                rel_parts = []
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i] in known_models or parts[i] == "phlag":
                        rel_parts = list(parts[i+1:])
                        break
                if rel_parts:
                    sub_path = pathlib.Path(*rel_parts)
                    out_dir = phlag_base / dist_type / sub_path / "phlag"
                else:
                    from .utils import get_simulation_categories, get_short_sim_name
                    cats = get_simulation_categories(locus_dir.name)
                    pattern_name = get_short_sim_name(locus_dir.name)
                    cat_prefix = pathlib.Path(*cats) if cats else pathlib.Path()
                    out_dir = phlag_base / dist_type / cat_prefix / pattern_name / "phlag"
            else:
                from .utils import get_simulation_categories, get_short_sim_name
                cats = get_simulation_categories(input_path.stem)
                pattern_name = get_short_sim_name(input_path.stem)
                cat_prefix = pathlib.Path(*cats) if cats else pathlib.Path()
                out_dir = phlag_base / dist_type / cat_prefix / pattern_name / "phlag"
        return out_dir

    def determine_optimal_mixtures(self):
        if self.args.output_file:
            output_dir = pathlib.Path(self.args.output_file).parent
        else:
            output_dir = self.get_default_out_dir()

        output_dir.mkdir(parents=True, exist_ok=True)

        num_mixtures_matrix, self.gmm_init_params = determine_optimal_mixtures(
            self.args.caster_scores,
            self.Y,
            self.pos_to_caster,
            self.args.silhouette_threshold,
            output_dir,
            getattr(self.args, "cluster_topologies", False)
        )
        
        self.num_mixtures = int(np.max(num_mixtures_matrix))
        self.mixture_masks = np.zeros((NUM_STATES, self.Y.shape[-1], self.num_mixtures), dtype=np.float32)
        for s in range(NUM_STATES):
            for d in range(self.Y.shape[-1]):
                m_count = num_mixtures_matrix[s, d]
                self.mixture_masks[s, d, :m_count] = 1.0
                
        self.mixture_masks = jnp.array(self.mixture_masks)
        
        print(f"\nFinal configuration: num_mixtures = {self.num_mixtures}, mixture_masks = \n{self.mixture_masks}\n")

    def initialize_output(self):
        if getattr(self.args, "output_file", None):
            self.output_file = pathlib.Path(self.args.output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = self.get_default_out_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            self.output_file = out_dir / "report.tsv"
            
        headers = [f"{' '.join(sys.argv)}"]
        self.output_str = "\n".join(headers)

    def get_n_best_viterbi_paths(self, initial_probs, transition_matrix, log_likelihoods, n):
        T, K = log_likelihoods.shape
        log_V = np.full((T, K, n), -np.inf)
        BP = np.zeros((T, K, n, 2), dtype=int)
        
        log_pi = np.log(initial_probs + 1e-12)
        log_A = np.log(transition_matrix + 1e-12)
        
        for s in range(K):
            log_V[0, s, 0] = log_pi[s] + log_likelihoods[0, s]
            
        for t in range(1, T):
            for s in range(K):
                candidates = []
                for s_prev in range(K):
                    for k_prev in range(n):
                        score = log_V[t-1, s_prev, k_prev] + log_A[s_prev, s] + log_likelihoods[t, s]
                        if score > -np.inf:
                            candidates.append((score, s_prev, k_prev))
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[0], reverse=True)
                for i in range(min(n, len(candidates))):
                    log_V[t, s, i] = candidates[i][0]
                    BP[t, s, i] = [candidates[i][1], candidates[i][2]]
                    
        final_candidates = []
        for s in range(K):
            for k in range(n):
                score = log_V[T-1, s, k]
                if score > -np.inf:
                    final_candidates.append((score, s, k))
        final_candidates.sort(key=lambda x: x[0], reverse=True)
        
        paths = []
        path_likelihoods = []
        for rank in range(min(n, len(final_candidates))):
            score, s_last, k_last = final_candidates[rank]
            path = np.zeros(T, dtype=int)
            path[T-1] = s_last
            curr_k = k_last
            likelihoods = np.zeros(T)
            likelihoods[T-1] = score
            for t in range(T-2, -1, -1):
                s_next = path[t+1]
                bp = BP[t+1, s_next, curr_k]
                path[t] = bp[0]
                curr_k = bp[1]
                likelihoods[t] = log_V[t, path[t], curr_k]
            paths.append(path)
            path_likelihoods.append(likelihoods)
        return paths, path_likelihoods

    def compute_output(self):
        divergence = self.hmm.state_emission_divergence(self.params)
        try:
            emission_divergence_str = ", ".join(map(str, divergence.tolist()))
        except TypeError:
            emission_divergence_str = str(float(divergence))
            
        # Get the emission distributions for each state to compute log likelihoods
        log_likelihoods = []
        for state in range(self.hmm.num_states):
            dist = self.hmm.emission_component.distribution(self.params.emissions, state)
            log_likelihoods.append(dist.log_prob(self.Y))
        log_likelihoods = jnp.stack(log_likelihoods, axis=-1)
        
        # Convert values to numpy arrays for Viterbi calculation
        initial_probs_np = np.array(self.params.initial.probs)
        transition_matrix_np = np.array(self.params.transitions.transition_matrix)
        log_likelihoods_np = np.array(log_likelihoods)
        
        # Extract the required ground truth pattern from the input filename/path.
        # Format example: ...a1n5a2a3n8n1...
        # 'a' = anomaly locus block of 500Kb, 'n' = normal locus block of 500Kb
        # Each token (e.g., 'n1', 'n8', 'a1', 'n5') represents one 500Kb locus block.
        import re
        input_path_obj = pathlib.Path(self.args.caster_scores)

        sorted_positions = sorted(self.pos_to_caster.keys())
        y_true = np.zeros(len(sorted_positions), dtype=int)

        pattern_str = None
        # 1. Try to find the pattern in the path parts (matches [an]\d+ blocks, range syntax, or start-end coords)
        pattern_full_match_regex = r'(?:[an]\d+)+(?:[_,]\d+-\d+(?:[_,]\d+-\d+)*)?|(?:[an]\d+(?:-[an]?\d+)?(?:[_,])?)+|\d+-\d+(?:[_,]\d+-\d+)*'
        for part in reversed(input_path_obj.parts):
            if re.fullmatch(pattern_full_match_regex, part):
                pattern_str = part
                break

        # 2. Try the file stem
        if not pattern_str:
            pattern_str_match = re.search(r'((?:[an]\d+)+(?:[_,]\d+-\d+(?:[_,]\d+-\d+)*)?|(?:[an]\d+(?:-[an]?\d+)?(?:[_,])?){2,}|\d+-\d+(?:[_,]\d+-\d+)*)', input_path_obj.stem)
            if pattern_str_match:
                pattern_str = pattern_str_match.group(1)

        if not pattern_str:
            sys.exit(f"Error: No ground truth locus pattern (e.g. 'n1a1n5...') found in '{input_path_obj}'. Phlag requires a ground truth pattern in the score file's path or filename.")

        from .utils import parse_pattern_string
        total_span = sorted_positions[-1] if sorted_positions else None
        blocks, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)
        if not blocks:
            sys.exit(f"Error: Could not parse ground truth locus pattern '{pattern_str}' from '{input_path_obj}'.")

        for idx, pos in enumerate(sorted_positions):
            for start_bp, end_bp in anomaly_intervals:
                if start_bp <= pos <= end_bp:
                    y_true[idx] = 1
                    break

        # Store ground truth info and compute the ground-truth-split empirical fits
        # (Null vs Alt), shared by the report's relative-error stats and the em.png top row
        self.y_true = y_true
        self.ground_truth_fits = {}
        Y_np = np.array(self.Y)
        for d in range(Y_np.shape[-1]):
            null_vals = Y_np[y_true == 0, d]
            alt_vals = Y_np[y_true == 1, d]
            if len(null_vals) > 1 and len(alt_vals) > 1:
                mu_null, std_null = stats.norm.fit(null_vals)
                mu_alt, std_alt = stats.norm.fit(alt_vals)
                self.ground_truth_fits[d] = (float(mu_null), float(std_null), float(mu_alt), float(std_alt))

        # Check if --correct-transition was requested to manually/automatically set ground truth transition matrix
        correct_trans_arg = getattr(self.args, "correct_transition", None)
        if correct_trans_arg is not None:
            transition_corrected = False
            gt_tm = None
            if isinstance(correct_trans_arg, str) and correct_trans_arg.lower() != "auto":
                nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", correct_trans_arg)]
                if len(nums) == 2:
                    p0, p1 = nums[0], nums[1]
                    gt_tm = np.array([[p0, 1.0 - p0], [1.0 - p1, p1]], dtype=np.float32)
                    transition_corrected = True
                elif len(nums) == 4:
                    gt_tm = np.array([[nums[0], nums[1]], [nums[2], nums[3]]], dtype=np.float32)
                    transition_corrected = True
                else:
                    print(f"Warning: Could not parse transition values from '{correct_trans_arg}'. Using auto ground-truth estimation.")
                    correct_trans_arg = "auto"
            
            if correct_trans_arg == "auto" or (isinstance(correct_trans_arg, str) and correct_trans_arg.lower() == "auto"):
                N_00 = np.sum((y_true[:-1] == 0) & (y_true[1:] == 0))
                N_01 = np.sum((y_true[:-1] == 0) & (y_true[1:] == 1))
                N_10 = np.sum((y_true[:-1] == 1) & (y_true[1:] == 0))
                N_11 = np.sum((y_true[:-1] == 1) & (y_true[1:] == 1))

                A00 = float(N_00 / (N_00 + N_01)) if (N_00 + N_01) > 0 else 0.5
                A01 = 1.0 - A00
                A10 = float(N_10 / (N_10 + N_11)) if (N_10 + N_11) > 0 else 0.5
                A11 = 1.0 - A10

                gt_tm = np.array([[A00, A01], [A10, A11]], dtype=np.float32)
                transition_corrected = True

            if transition_corrected and gt_tm is not None:
                from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
                transition_matrix_np = gt_tm
                self.params = self.params._replace(
                    transitions=ParamsStandardHMMTransitions(transition_matrix=jnp.array(gt_tm, dtype=jnp.float32))
                )
                print(f"\n[Ground Truth Transition Matrix Override] Applied: {gt_tm.tolist()}\n")

        # Calculate n-best Viterbi paths
        n_paths = getattr(self.args, "best_paths", 1)
        paths, path_likelihoods = self.get_n_best_viterbi_paths(
            initial_probs_np, transition_matrix_np, log_likelihoods_np, n_paths
        )

        # Calculate metrics for primary Viterbi path (Path 1)
        y_pred = np.array(paths[0])
        flipped_for_eval = False
        
        # Flip the state assignments according to whichever has smaller hamming distance
        hamming_dist = np.sum(y_pred != y_true)
        hamming_dist_flipped = np.sum((1 - y_pred) != y_true)

        if hamming_dist_flipped < hamming_dist:
            y_pred = 1 - y_pred
            flipped_for_eval = True

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

        metrics_str = f"TPR: {tpr:.4f}, FPR: {fpr:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}"
        print(f"\n[Evaluation Metrics] {metrics_str}\n")

        # Build headers
        from .utils import get_simulation_clade, get_cu_branch_length_from_population_info, get_population_info
        _, clade_name, clade_number = get_simulation_clade(self.args.caster_scores)
        headers = []
        headers.append(f"Clade: {clade_name if clade_name else 'N/A'}")
        if clade_number:
            headers.append(f"Clade number: {clade_number}")

        pop_info = get_population_info(clade_name) if clade_name else None

        is_admixture = "admixture" in pathlib.Path(self.args.caster_scores).parts
        if is_admixture:
            # Admixture events reference an internal simulator node distinct from the
            # named donor clade, not a single tree edge -- no branch length to report.
            headers.append("Branch length (CU): N/A (not applicable for admixture)")
        elif clade_name:
            branch_len = get_cu_branch_length_from_population_info(clade_name, pop_info=pop_info)
            if branch_len is None:
                headers.append(f"Branch length (CU): N/A (no population-information.tsv row for clade '{clade_name}')")
            else:
                headers.append(f"Branch length (CU, node '{clade_name}'): {branch_len:.6f}")
        else:
            headers.append("Branch length (CU): N/A")

        if pop_info:
            try:
                headers.append(f"Height (Ngen): {float(pop_info.get('HEIGHT_NGEN')):.4f}")
            except (TypeError, ValueError):
                headers.append("Height (Ngen): N/A")
            try:
                headers.append(f"Clade size: {int(float(pop_info.get('CLADE_SIZE')))}")
            except (TypeError, ValueError):
                headers.append("Clade size: N/A")
        else:
            headers.append("Height (Ngen): N/A")
            headers.append("Clade size: N/A")

        # Self-describing ground-truth/evaluation summary. These lines let downstream
        # consumers (phlag.benchmark) read the anomaly fraction and the eval-time
        # label polarity straight out of the report, instead of re-deriving them from
        # the locus pattern -- the fraction in particular is not a pure function of
        # the pattern string, since it depends on where the actual window grid
        # (sorted_positions) falls inside the anomaly intervals.
        n_windows = int(len(y_true))
        n_anomaly = int(np.sum(y_true == 1))
        anomaly_fraction = (n_anomaly / n_windows) if n_windows > 0 else 0.0
        headers.append(f"Anomaly fraction: {anomaly_fraction:.6f} ({n_anomaly}/{n_windows} windows)")
        headers.append(f"Label polarity flipped for evaluation: {flipped_for_eval}")

        headers.append("State divergence: " + emission_divergence_str)
        headers.append(f"Outer EM iterations: {self.n_iters}")
        headers.append(f"Inner EM iterations: {self.increment_steps}")
        if hasattr(self, "initial_transition_matrix") and self.initial_transition_matrix is not None:
            tm_before_str = ", ".join(f"[{', '.join(f'{x:.6f}' for x in row)}]" for row in self.initial_transition_matrix.tolist())
            headers.append(f"Initial transition matrix (before EM): [{tm_before_str}]")
        tm_after_str = ", ".join(f"[{', '.join(f'{x:.6f}' for x in row)}]" for row in transition_matrix_np.tolist())
        headers.append(f"Final transition matrix (after EM): [{tm_after_str}]")
        if correct_trans_arg is not None:
            headers.append("Corrected transition matrix applied: True")

        if hasattr(self, "final_em_log_prob"):
            headers.append(f"EM final joint log-likelihood: {self.final_em_log_prob:.6f}")
        marginal_ll = float(self.hmm.marginal_log_prob(self.params, self.Y))
        k_params = count_trainable_params(self.params, self.props)
        n_obs = int(self.Y.shape[0])
        bic = k_params * np.log(n_obs) - 2 * marginal_ll
        headers.append(
            f"BIC: {bic:.6f} (log-likelihood={marginal_ll:.6f}, k={k_params} trainable params, n={n_obs} windows)"
        )

        headers.append(f"{metrics_str}")
        headers.append(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
        if self.ground_truth_fits:
            topology_names = get_topology_names(self.Y.shape[-1])
            headers.append("Topology\tState\tStatistic\tRel.err(%)")
            for d in sorted(self.ground_truth_fits.keys()):
                mu_null_gt, std_null_gt, mu_alt_gt, std_alt_gt = self.ground_truth_fits[d]
                mu_null_fit, std_null_fit, _ = get_state_mu_sigma_pdf(self.params, self.args.model_design, 0, d, np.array([0.0]))
                mu_alt_fit, std_alt_fit, _ = get_state_mu_sigma_pdf(self.params, self.args.model_design, 1, d, np.array([0.0]))
                rel_mu_null = (mu_null_fit - mu_null_gt) / mu_null_gt * 100 if mu_null_gt != 0 else float('nan')
                rel_std_null = (std_null_fit - std_null_gt) / std_null_gt * 100 if std_null_gt != 0 else float('nan')
                rel_mu_alt = (mu_alt_fit - mu_alt_gt) / mu_alt_gt * 100 if mu_alt_gt != 0 else float('nan')
                rel_std_alt = (std_alt_fit - std_alt_gt) / std_alt_gt * 100 if std_alt_gt != 0 else float('nan')
                topo_name = topology_names[d] if d < len(topology_names) else f"Coord {d+1}"
                headers.append(f"{topo_name}\tNull\tmean\t{format_rel_err(rel_mu_null)}")
                headers.append(f"{topo_name}\tNull\tstd\t{format_rel_err(rel_std_null)}")
                headers.append(f"{topo_name}\tAlt\tmean\t{format_rel_err(rel_mu_alt)}")
                headers.append(f"{topo_name}\tAlt\tstd\t{format_rel_err(rel_std_alt)}")
        for idx, l in enumerate(path_likelihoods):
            headers.append(f"Path {idx + 1} final joint log-likelihood: {l[-1]:.6f}")

        self.output_str += "\n" + "\n".join(headers)

        # Add the state paths as comma-separated rows in the report
        for path in paths:
            effective_path = (1 - path) if flipped_for_eval else path
            self.output_str += "\n" + ",".join(map(str, effective_path.tolist()))

        # Generate the visual plot if configured: states
        if self.args.plot and "states" in self.args.plot:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                sns.set_theme(style="white")
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                input_path = pathlib.Path(self.args.caster_scores)
                sorted_positions = sorted(self.pos_to_caster.keys())
                positions_kb = np.array(sorted_positions) / 1000.0  # in kb
                
                colors = sns.color_palette("tab10", len(paths))
                
                for idx in range(len(paths)):
                    path = paths[idx]
                    plot_path_data = (1 - path) if flipped_for_eval else path
                    color = colors[idx]
                    line_style = "-" if idx == 0 else ("--" if idx == 1 else "-.")
                    ax1.step(positions_kb, plot_path_data, where="mid", color=color, linestyle=line_style, linewidth=1.5, label=f"Path {idx+1}")
                
                ax1.step(positions_kb, y_true, where="mid", color='black', linestyle='--', linewidth=2.0, label="Ground Truth", alpha=0.8)

                ax1.set_xlabel("Genomic Position (kb)", fontsize=12, labelpad=10)
                ax1.set_ylabel("HMM State", fontsize=12, labelpad=10)
                ax1.set_ylim(-0.05, 1.05)
                ax1.set_yticks([0, 1])
                from .utils import get_locus_description
                locus_desc = get_locus_description(input_path)
                if locus_desc:
                    plt.title(f"Genomic Profile: {locus_desc}\nTop {len(paths)} Viterbi Paths", fontsize=12, fontweight="bold", pad=12)
                else:
                    plt.title(f"Genomic Profile: Top {len(paths)} Viterbi Paths", fontsize=14, fontweight="bold", pad=15)
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                ax1.legend(lines1, labels1, loc="upper left", framealpha=0.9)
                
                fig.tight_layout()
                
                output_path = pathlib.Path(self.output_file)
                plot_path = output_path.with_name("states.png")
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"Saved visual HMM states plot to: {plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate visual states plot: {e}")

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
            f"Step-based counts of flagged positions (step size = {step_size}):",
            "Range       | Count | Bar",
        ]
        
        max_count = max([c for _, _, c in step_counts]) if step_counts else 0
        bar_max_width = 40
        
        for start, end, count in step_counts:
            bar_len = int((count / max_count) * bar_max_width) if max_count > 0 else 0
            bar = "*" * bar_len
            range_str = f"{start:<6}-{end:<6}"
            lines.append(f"# {range_str} | {count:<5} | {bar}")
            
        text_hist = "\n".join(lines) + "\n"

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
        
        self.emission_parameterization = (
            self.args.null_emission_parameterization,
            self.args.alt_emission_parameterization,
        )
        
        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.Y.shape[-1],
            emission_lambda=self.emission_lambda,
            emission_concentration=self.gamma,
            emission_parameterization=self.emission_parameterization,
            initial_probs_concentration=self.nu,
            transition_concentration=self.psi,
            occupancy_bias=self.occupancy_bias,
            model_design=self.args.model_design,
            num_mixtures=getattr(self, "num_mixtures", 2),
        )
        if self.args.model_design == "gmm":
            self.hmm.emission_component.mixture_masks = self.mixture_masks
        
        # Compute empirical moments from CASTER data over genomic positions
        data_mean = jnp.mean(self.Y, axis=0)
        data_std = jnp.std(self.Y, axis=0)
        
        # alt variance = double null variance
        state0_init = jnp.stack([data_mean, 2 * data_std ** 2], axis=-1)
        state1_init = jnp.stack([data_mean, 2 * data_std ** 2], axis=-1)

        # Shape: [num_states, emission_dim, 2] where the last axis is [mean, variance]
        init_emissions = jnp.stack([state0_init, state1_init], axis=0)
        p0, p1 = 0.99, 0.99
        initial_transition_matrix = jnp.array([[p0, 1-p0], [1-p1, p1]], dtype=jnp.float32)
        
        if self.args.model_design == "gmm":
            self.params, self.props = self.hmm.initialize(
                initial_probs=INITIAL_PROBS,
                emission_probs=init_emissions,
                transition_matrix=initial_transition_matrix,
                initial_gmm_params=getattr(self, "gmm_init_params", None),
            )
        else:
            self.params, self.props = self.hmm.initialize(
                initial_probs=INITIAL_PROBS,
                emission_probs=init_emissions,
                transition_matrix=initial_transition_matrix,
            )
        self.initial_transition_matrix = np.array(self.params.transitions.transition_matrix)
        self.props.transitions.transition_matrix.trainable = True
        if self.args.model_design == "gmm":
            self.props.emissions.mixture_weights.trainable = True
            self.props.emissions.means.trainable = True
            self.props.emissions.stds.trainable = True
        else:
            self.props.emissions.means.trainable = True
            self.props.emissions.covariances.trainable = True
        
        self.hmm.initialize_m_step_state(self.params, self.props)
        self.n_iters = self.args.n_iters
        self.increment_steps = self.args.increment_steps
        
    def run(self):
        tm_init = self.params.transitions.transition_matrix
        tm_init_str = ", ".join(f"[{', '.join(f'{x:.6f}' for x in row)}]" for row in tm_init.tolist())
        tqdm.write(f"Initial Transition matrix: {tm_init_str}")

        for i in tqdm(range(self.n_iters)):
            num_inner = (i + 1) * self.increment_steps
            self.params, log_probs = self.hmm.fit_em(
                self.params,
                self.props,
                self.Y,
                num_iters=num_inner,
                verbose=False,
            )
            tm = self.params.transitions.transition_matrix
            tm_str = ", ".join(f"[{', '.join(f'{x:.6f}' for x in row)}]" for row in tm.tolist())
            tqdm.write(f"Outer EM iteration {i + 1}/{self.n_iters} ({num_inner} inner steps) - Transition matrix: {tm_str}")
        # log_probs is dynamax's own "joint log probability" trace for this fit_em call
        # (log_prior(params) + sum of per-step data log-likelihoods); its last entry is
        # the final joint log-likelihood EM converged to.
        self.final_em_log_prob = float(log_probs[-1])
        self.compute_output()

    def save_output(self):
        with open(self.output_file, "w") as f:
            f.write(self.output_str)
        print(f"Saved PHLAG output report to: {self.output_file}")


def backfill_branch_length(root_dir=None):
    """
    Patches the 'Branch length (CU...)' header line in every already-written
    report.tsv under root_dir (default: the same CONNECTION_DIR-resolved phlag
    output base that get_default_out_dir() uses) to use
    get_cu_branch_length_from_population_info() instead of whatever it was
    computed with before -- without touching anything else in the file (no EM
    rerun needed, since this line never depended on the HMM/Viterbi output).

    Never overwrites a line that already carries a real numeric branch length,
    and leaves admixture reports (no single branch to report) untouched.
    Returns a dict of counts: patched / already_resolved / admixture / unresolved / no_clade.
    """
    import re
    import pathlib
    from . import utils
    from .benchmark import parse_report

    branch_length_line_re = re.compile(r'^Branch length \(CU[^)]*\):\s*(.+?)\s*$')

    if root_dir is None:
        conn_env = os.environ.get("CONNECTION_DIR")
        if conn_env:
            base_dir = pathlib.Path(conn_env)
        else:
            base_dir = utils.get_data_dir()
            if base_dir == utils.get_repo_root() / "caster" / "data":
                base_dir = utils.get_repo_root() / "connection_dir"
        root_dir = utils.get_phlag_output_base(base_dir)
    root_dir = pathlib.Path(root_dir)

    counts = {"patched": 0, "already_resolved": 0, "admixture": 0, "unresolved": 0, "no_clade": 0}

    for report_path in sorted(root_dir.rglob("report.tsv")):
        parsed = parse_report(report_path)
        clade_name = parsed["clade_name"]

        if "admixture" in report_path.parts:
            counts["admixture"] += 1
            continue
        if not clade_name:
            counts["no_clade"] += 1
            continue
        if parsed["branch_length_present"] and parsed["branch_length"] is not None:
            counts["already_resolved"] += 1
            continue

        branch_len = utils.get_cu_branch_length_from_population_info(clade_name)
        if branch_len is None:
            counts["unresolved"] += 1
            continue

        with open(report_path, "r") as f:
            lines = f.read().splitlines()

        new_line = f"Branch length (CU, node '{clade_name}'): {branch_len:.6f}"
        for idx, line in enumerate(lines):
            if branch_length_line_re.match(line.strip()):
                lines[idx] = new_line
                break
        else:
            continue

        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        counts["patched"] += 1

    print(
        f"Backfilled branch lengths under {root_dir}: "
        f"{counts['patched']} patched, {counts['already_resolved']} already resolved, "
        f"{counts['admixture']} admixture (skipped), {counts['unresolved']} still unresolved, "
        f"{counts['no_clade']} with no clade name."
    )
    return counts


class PhlagPlotter:
    """
    Object-oriented plotter for visualizing the probability distributions
    of standard multi-species coalescent (MSC) background (State 0, Null) vs.
    alternative/anomalous (State 1, Alternative) states.

    Row 0 shows the ground-truth-split empirical histogram and independent Null/Alt
    gaussian fits (or a single ungrounded distribution when no ground truth is available).
    Row 1 shows the empirical HMM-assigned histogram and EM-fitted emission curves.
    """
    def __init__(self, phlag):
        self.phlag = phlag

        # Ingest and configure metadata parameters
        self.extract_metadata()

        # Generate the visual distribution charts
        self.plot_distributions()
        self.plot_correlations()

    def extract_metadata(self):
        """Extracts genomic filename, dimension, and styles configuration."""
        self.input_path = pathlib.Path(self.phlag.args.caster_scores)
        self.emission_dim = self.phlag.Y.shape[-1]
        self.topology_names = get_topology_names(self.emission_dim)

        # Define premium color palette and labels
        self.colors = {
            0: {
                'line': '#2B4C7E',    # Deep Steel Blue
                'fill': '#2B4C7E',
                'label': 'Null'
            },
            1: {
                'line': '#E05A47',    # Warm Coral
                'fill': '#E05A47',
                'label': 'Alt'
            }
        }

    def plot_distributions(self):
        """Prepares the subplot layout and runs plotting for the ground-truth and After-EM rows."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
            'axes.edgecolor': '#cccccc',
            'grid.color': '#f0f0f0'
        })

        # Create a 2x3 panel layout (or 2x1 if single dimension).
        fig, axes = plt.subplots(2, self.emission_dim, figsize=(5.5 * self.emission_dim, 9.5), sharey=False)

        if self.emission_dim == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        # Determine consistent coordinate limits across both rows
        ranges = {}
        for d in range(self.emission_dim):
            ymin, ymax = float(np.min(self.phlag.Y[:, d])), float(np.max(self.phlag.Y[:, d]))
            ypad = (ymax - ymin) * 0.20 or 0.1
            ranges[d] = np.linspace(ymin - ypad, ymax + ypad, 300)

        # Row 0: GMM initialization seed (gmm) or ground-truth-split empirical distribution (gaussian)
        if self.phlag.args.model_design == "gmm":
            self._plot_gmm_init_row(axes[0], ranges)
        else:
            self._plot_ground_truth_row(axes[0], ranges)

        # Row 1: After EM (fitted emission curves and HMM-assigned empirical data)
        self._plot_em_row(axes[1], ranges)

        self._finalize_legend(fig, axes)

        from .utils import get_locus_description
        locus_desc = get_locus_description(self.phlag.args.caster_scores)
        if locus_desc:
            fig.suptitle(f"EM Distributions | {locus_desc}", fontsize=13, fontweight="bold", y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.save_plot()

    def _finalize_legend(self, fig, axes):
        """Collects de-duplicated handles/labels across every axis in the figure (both
        rows share the same Null/Alt Fit/Histogram labeling) and places a single legend
        in the figure's upper-right corner."""
        unique = {}
        for ax in np.array(axes).flat:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in unique:
                    unique[label] = handle
        if unique:
            fig.legend(
                unique.values(), unique.keys(),
                loc='upper right', bbox_to_anchor=(0.995, 0.97),
                fontsize=7.5, framealpha=0.9
            )

    def _plot_ground_truth_row(self, row_axes, ranges):
        """Plots the ground-truth-split empirical histogram and independent Null/Alt gaussian fits."""
        y_true = self.phlag.y_true
        ground_truth_fits = self.phlag.ground_truth_fits
        title_prefix = "Ground Truth Split"
        Y_np = np.array(self.phlag.Y)

        for d in range(self.emission_dim):
            ax = row_axes[d]
            x_vals = ranges[d]
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            vals = Y_np[:, d]

            mu_null, std_null, mu_alt, std_alt = ground_truth_fits[d]
            null_vals = vals[y_true == 0]
            alt_vals = vals[y_true == 1]

            if len(null_vals) > 0:
                sns.histplot(null_vals, ax=ax, stat='density', element='step', kde=False, alpha=0.35, color=self.colors[0]['fill'], label='Null Histogram', bins=30)
            if len(alt_vals) > 0:
                sns.histplot(alt_vals, ax=ax, stat='density', element='step', kde=False, alpha=0.35, color=self.colors[1]['fill'], label='Alt Histogram', bins=30)

            pdf_null = stats.norm.pdf(x_vals, mu_null, std_null)
            ax.plot(x_vals, pdf_null, color=self.colors[0]['line'], linewidth=2.2, label='Null Fit')
            ax.axvline(mu_null, color=self.colors[0]['line'], linestyle='--', linewidth=1.5)
            ax.text(mu_null, 0.90, f"$\\mu_{{null}}={mu_null:.4f}$", transform=trans, color=self.colors[0]['line'], fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            ax.text(mu_null + std_null, 0.82, f"$\\sigma_{{null}}={std_null:.4f}$", transform=trans, color=self.colors[0]['line'], fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            pdf_alt = stats.norm.pdf(x_vals, mu_alt, std_alt)
            ax.plot(x_vals, pdf_alt, color=self.colors[1]['line'], linewidth=2.2, label='Alt Fit')
            ax.axvline(mu_alt, color=self.colors[1]['line'], linestyle=':', linewidth=1.5)
            ax.text(mu_alt, 0.75, f"$\\mu_{{alt}}={mu_alt:.4f}$", transform=trans, color=self.colors[1]['line'], fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            ax.text(mu_alt + std_alt, 0.67, f"$\\sigma_{{alt}}={std_alt:.4f}$", transform=trans, color=self.colors[1]['line'], fontsize=7, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            ax.set_title(f"{title_prefix} | Topology: {self.topology_names[d]}", fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel("Topology Score", fontsize=9, labelpad=4)
            ax.set_ylabel("Density" if d == 0 else "", fontsize=9, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=8)

    def _plot_gmm_init_row(self, row_axes, ranges):
        """Plots the pre-EM GMM initialization seed: weighted-sum-of-Gaussians curves per
        state, derived from the k-means-based mixture seed (self.phlag.gmm_init_params)."""
        init_weights, init_means, init_stds = self.phlag.gmm_init_params
        title_prefix = "GMM Initialization"

        for d in range(self.emission_dim):
            ax = row_axes[d]
            x_vals = ranges[d]
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

            for state in [0, 1]:
                color_config = self.colors[state]
                w = np.array(init_weights[state, d])
                m_means = np.array(init_means[state, d])
                m_stds = np.array(init_stds[state, d])
                pdf_vals = np.zeros_like(x_vals)
                for m in range(len(w)):
                    pdf_vals += w[m] * stats.norm.pdf(x_vals, m_means[m], m_stds[m])
                mu = float(np.sum(w * m_means))
                var = float(np.sum(w * (m_stds ** 2 + m_means ** 2)) - mu ** 2)
                sigma = np.sqrt(max(1e-6, var))

                ax.plot(x_vals, pdf_vals, color=color_config['line'], linewidth=2.2, label=f"{color_config['label']} Fit")
                ax.fill_between(x_vals, pdf_vals, alpha=0.05, color=color_config['fill'])
                ax.axvline(x=mu, color=color_config['line'], linestyle='--', linewidth=1.5, alpha=0.8)
                ax.axvline(x=mu - sigma, color=color_config['line'], linestyle=':', linewidth=1.0, alpha=0.6)
                ax.axvline(x=mu + sigma, color=color_config['line'], linestyle=':', linewidth=1.0, alpha=0.6)

                y_pos_mean = 0.90 if state == 0 else 0.75
                y_pos_std = 0.83 if state == 0 else 0.68
                ax.text(mu, y_pos_mean, f"$\\mu_{state} = {mu:.4f}$", transform=trans, color=color_config['line'], fontsize=8.0, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))
                ax.text(mu + sigma, y_pos_std, f"$\\sigma_{state} = {sigma:.4f}$", transform=trans, color=color_config['line'], fontsize=7.0, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))

            ax.set_title(f"{title_prefix} | Topology: {self.topology_names[d]}", fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel("Topology Score", fontsize=9, labelpad=4)
            ax.set_ylabel("Density" if d == 0 else "", fontsize=9, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=8)

    def _plot_em_row(self, row_axes, ranges):
        """Plots the empirical HMM-assigned histogram and EM-fitted emission curves."""
        params = self.phlag.params
        title_prefix = "After EM"
        most_likely_states = self.phlag.hmm.most_likely_states(params, self.phlag.Y)

        for d in range(self.emission_dim):
            ax = row_axes[d]
            x_vals = ranges[d]
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

            # 1. Plot empirical step-histograms for Assigned State Data points
            y_state0 = np.array(self.phlag.Y[most_likely_states == 0, d])
            y_state1 = np.array(self.phlag.Y[most_likely_states == 1, d])

            if len(y_state0) > 0:
                sns.histplot(
                    y_state0, ax=ax, color=self.colors[0]['fill'],
                    stat="density", kde=False, alpha=0.12,
                    element="step", label=f"{self.colors[0]['label']} Histogram"
                )
            if len(y_state1) > 0:
                sns.histplot(
                    y_state1, ax=ax, color=self.colors[1]['fill'],
                    stat="density", kde=False, alpha=0.12,
                    element="step", label=f"{self.colors[1]['label']} Histogram"
                )

            # 2. Plot PDF curves and Vertical Guideline Markers (Mean and +/- 1 Std)
            for state in [0, 1]:
                color_config = self.colors[state]
                mu, sigma, pdf_vals = get_state_mu_sigma_pdf(params, self.phlag.args.model_design, state, d, x_vals)

                # Plot theoretical curve
                ax.plot(
                    x_vals, pdf_vals, color=color_config['line'],
                    linewidth=2.2, label=f"{color_config['label']} Fit"
                )

                # Shading under curve
                ax.fill_between(
                    x_vals, pdf_vals, alpha=0.05,
                    color=color_config['fill']
                )

                # Plot Central Tendency Guideline: Mean
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
            ax.set_ylabel("Probability Density" if d == 0 else "", fontsize=9, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=8)

    def _covariance_ellipse(self, mean_xy, cov2x2, n_std=1.0, **kwargs):
        """Builds an n_std confidence-region Ellipse patch for a 2D Gaussian, via
        eigendecomposition of its 2x2 covariance submatrix (eigenvectors give the
        ellipse's orientation, sqrt(eigenvalues) its axis lengths)."""
        from matplotlib.patches import Ellipse

        eigenvalues, eigenvectors = np.linalg.eigh(np.array(cov2x2))
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(np.clip(eigenvalues, a_min=0, a_max=None))
        return Ellipse(xy=mean_xy, width=width, height=height, angle=angle, **kwargs)

    def plot_correlations(self):
        """
        Pairwise cross-topology correlation plot: for every pair of topology
        dimensions, scatters the HMM-assigned data and draws 1-sigma/2-sigma
        ellipses from the fitted full covariance matrix (PhlagHMMEmissions now
        uses MultivariateNormalFullCovariance, not just the diagonal -- see
        hmm.py), so the correlation structure the EM fit actually captures is
        visible instead of only implied by the independent per-dimension curves
        in em.png. Gaussian-only: gmm emissions have no cross-dimension
        covariance to show.
        """
        if self.emission_dim < 2 or self.phlag.args.model_design != "gaussian":
            return

        import itertools

        pairs = list(itertools.combinations(range(self.emission_dim), 2))
        params = self.phlag.params
        Y_np = np.array(self.phlag.Y)
        most_likely_states = np.array(self.phlag.hmm.most_likely_states(params, self.phlag.Y))

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5.5), squeeze=False)
        axes = axes[0]

        for col_idx, (i, j) in enumerate(pairs):
            ax = axes[col_idx]

            for state in [0, 1]:
                color_config = self.colors[state]
                mask = most_likely_states == state
                ax.scatter(
                    Y_np[mask, i], Y_np[mask, j],
                    s=8, alpha=0.25, color=color_config['fill'], linewidths=0,
                    label=f"{color_config['label']} Windows",
                )

            for state in [0, 1]:
                color_config = self.colors[state]
                mean_xy = np.array(params.emissions.means[state])[[i, j]]
                cov2x2 = np.array(params.emissions.covariances[state])[np.ix_([i, j], [i, j])]

                for n_std, alpha, linestyle, sigma_label in [(1.0, 0.9, '-', '1σ'), (2.0, 0.5, '--', '2σ')]:
                    ellipse = self._covariance_ellipse(
                        mean_xy, cov2x2, n_std=n_std,
                        edgecolor=color_config['line'], facecolor='none',
                        linewidth=1.6, linestyle=linestyle, alpha=alpha,
                        label=f"{color_config['label']} {sigma_label}",
                    )
                    ax.add_patch(ellipse)
                ax.plot(mean_xy[0], mean_xy[1], marker='x', color=color_config['line'], markersize=8, markeredgewidth=2)

                denom = np.sqrt(cov2x2[0, 0] * cov2x2[1, 1])
                corr = float(cov2x2[0, 1] / denom) if denom > 0 else 0.0
                y_text = 0.95 if state == 0 else 0.88
                ax.text(
                    0.03, y_text, f"$\\rho_{{{color_config['label'].lower()}}} = {corr:.3f}$",
                    transform=ax.transAxes, color=color_config['line'], fontsize=9,
                    fontweight='bold', va='top',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1),
                )

            ax.set_title(f"{self.topology_names[i]} vs {self.topology_names[j]}", fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel(f"{self.topology_names[i]} Score", fontsize=9, labelpad=4)
            ax.set_ylabel(f"{self.topology_names[j]} Score", fontsize=9, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=8)

        self._finalize_legend(fig, axes.reshape(1, -1))

        from .utils import get_locus_description
        locus_desc = get_locus_description(self.phlag.args.caster_scores)
        if locus_desc:
            fig.suptitle(f"Cross-Topology Correlation (After EM) | {locus_desc}", fontsize=13, fontweight="bold", y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.88])
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.92])

        self.save_plot("correlations.png")

    def save_plot(self, filename="em.png"):
        """Saves the currently active matplotlib figure as PNG."""
        output_dir = getattr(self.phlag, "output_file", None)
        if output_dir:
            output_dir = output_dir.parent
        else:
            if self.phlag.args.output_file:
                output_dir = pathlib.Path(self.phlag.args.output_file).parent
            else:
                output_dir = pathlib.Path.cwd() / "test" / self.input_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)

        plot_file = output_dir / filename

        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to: {plot_file}")


def int_or_abbrev(val_str):
    val_str = str(val_str).strip().lower()
    if val_str.endswith('k'):
        return int(float(val_str[:-1]) * 1000)
    elif val_str.endswith('m'):
        return int(float(val_str[:-1]) * 1000000)
    return int(val_str)

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Phlag: Detecting genomic regions with unexplained phylogenetic heterogeneity using CASTER"
    )

    parser.add_argument(
        "caster_scores",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help="Path to the CASTER scores TSV"
    )
    parser.add_argument(
        "-r",
        "--recent",
        dest="recent",
        action="store_true",
        help="Use the most recently created score file in model parameter directories"
    )

    parser.add_argument(
        "-o", dest="output_file", type=pathlib.Path, required=False, default=None, help="Path to save the output"
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=["em", "states"],
        default=["em", "states"],
        help="List of plots to generate (choices: em, states. Default: both em and states)",
    )
    parser.add_argument(
        "-L",
        "--n-iters",
        dest="n_iters",
        type=int_or_abbrev,
        default=10,
        help="Number of outer EM iterations (default: 10)",
    )
    parser.add_argument(
        "-s",
        "--step-size",
        dest="step_size",
        type=int_or_abbrev,
        required=False,
        default=None,
        help="Genomic step size (in rows/positions) to compute a text-based ASCII histogram and save a visual bar chart plot",
    )

    hmm_group = parser.add_argument_group("HMM parameters")
    hmm_group.add_argument(
        "--np",
        dest="null_emission_parameterization",
        type=str.lower,
        default="free",
        choices=["free", "repulsion"],
        help="""Parameterization of the null (background) state's emission distribution
                    (default: free): free (unconstrained MLE) or repulsion (pushed away
                    from the alt state's currently fitted distribution).""",
    )
    hmm_group.add_argument(
        "--ap",
        dest="alt_emission_parameterization",
        type=str.lower,
        default="repulsion",
        choices=["free", "repulsion"],
        help="""Parameterization of the alt (anomalous) state's emission distribution
                    (default: repulsion): free (unconstrained MLE) or repulsion (pushed
                    away from the null state's currently fitted distribution).""",
    ) 
    hmm_group.add_argument(
        "--lam",
        dest="emission_lambda",
        type=float,
        default=1.0,
        help="Emission penalty regularizer parameter lambda (default: 1.0)",
    )
    hmm_group.add_argument(
        "-d",
        dest="model_design",
        type=str.lower,
        default="gaussian",
        choices=["gaussian", "gmm"],
        help="Type of HMM emissions (gaussian or gmm. Default: gaussian)",
    )
    hmm_group.add_argument(
        "-c",
        "--cluster-topologies",
        dest="cluster_topologies",
        action="store_true",
        help="For GMM, cluster topologies together instead of independently",
    )
    hmm_group.add_argument(
        "-t",
        "--silhouette-threshold",
        dest="silhouette_threshold",
        type=float,
        default=0.5,
        help="Silhouette score threshold to determine optimal GMM mixture counts (default: 0.5)",
    )
    hmm_group.add_argument(
        "-p",
        "--best-paths",
        dest="best_paths",
        type=int,
        default=1,
        help="Number of best Viterbi paths to calculate and plot (default: 1)",
    )
    hmm_group.add_argument(
        "--correct-transition",
        dest="correct_transition",
        nargs="?",
        const="auto",
        default=None,
        help="Manually set final transition matrix to ground truth (or pass explicit probabilities p0,p1)",
    )

    args = parser.parse_args(argv)

    from .utils import get_data_dir, get_repo_root, resolve_input_file, get_most_recent_file
    repo_root = get_repo_root()
    data_dir = get_data_dir()

    dist_type = getattr(args, "model_design", "gaussian")

    def resolve_model_scores(target_name=None):
        from .utils import get_phlag_output_base
        search_bases = [
            get_phlag_output_base(data_dir) / dist_type,
            get_phlag_output_base(repo_root / "store" / "phlag") / dist_type,
            get_phlag_output_base(data_dir),
            get_phlag_output_base(repo_root / "store" / "phlag"),
        ]
        candidates = []
        seen = set()
        for b in search_bases:
            if not b.exists():
                continue
            if target_name:
                # Look for target_name subdirectory
                target_dirs = list(b.glob(f"**/{target_name}/**/caster")) + list(b.glob(f"**/{target_name}"))
                for td in target_dirs:
                    if td.is_dir():
                        for sfile in td.rglob("scores.tsv"):
                            if sfile.parent.name == "caster" and sfile.resolve() not in seen:
                                seen.add(sfile.resolve())
                                candidates.append(sfile)
                        for sfile in td.rglob("*.tsv"):
                            if sfile.parent.name == "caster" and not any(sfile.name.startswith(p) for p in ["report_", "em_", "states_"]) and sfile.resolve() not in seen:
                                seen.add(sfile.resolve())
                                candidates.append(sfile)
            else:
                for sfile in b.rglob("scores.tsv"):
                    if sfile.parent.name == "caster" and sfile.resolve() not in seen:
                        seen.add(sfile.resolve())
                        candidates.append(sfile)
        if candidates:
            candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return candidates[0].resolve()
        return None

    if args.recent or args.caster_scores == pathlib.Path("-r") or args.caster_scores is None:
        recent_file = resolve_model_scores()
        if recent_file is None:
            recent_file = get_most_recent_file(
                default_subdirs=["store/phlag", "phlag"],
                default_exts=[".tsv", ".txt"],
                exclude_prefixes=["report_", "em_", "walkthrough", "implementation_plan"],
                target_dir_name="caster"
            )
        if recent_file is None or not recent_file.exists():
            sys.exit("Error: No score file found in store/phlag or candidate data directories.")
        print(f"Using most recent score file: {recent_file}")
        args.caster_scores = recent_file
    else:
        resolved = resolve_input_file(args.caster_scores, default_subdirs=["scores", "msa", "store/scores", "store/phlag"], default_exts=[".tsv", ".txt"])
        if resolved.exists() and resolved.is_file():
            args.caster_scores = resolved
        else:
            sys.exit(f"Error: Score file not found for '{args.caster_scores}' under model output directories or relative paths.")

    # Check if --plot is supplied
    check_argv = argv if argv is not None else sys.argv[1:]
    plot_supplied = any(arg == "--plot" or arg.startswith("--plot=") for arg in check_argv)
    if plot_supplied and args.step_size is None:
        parser.error("argument -s/--step-size is required if --plot is supplied")

    return args


def organize_existing_test_files():
    import pathlib
    import shutil
    test_dir = pathlib.Path.cwd() / "test"
    if not test_dir.exists():
        return
    for item in list(test_dir.iterdir()):
        if not item.is_file() or item.name == "check_output.py":
            continue
        fname = item.name
        stem = None
        if fname.startswith("report_") or fname.startswith("em_") or fname.startswith("states_"):
            parts = fname.split("_", 2)
            if len(parts) >= 3:
                stem = pathlib.Path(parts[2]).stem
        elif fname.startswith("kmeans_"):
            stem = "null-neoaves_alt-Nyctibiidae_10X_up_n1n8a1n5_0_2m_w50k_s1k"
        
        if stem:
            target_dir = test_dir / stem
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(item), str(target_dir / fname))
            except Exception:
                pass


organize_existing_test_files()


def main(argv=None):
    organize_existing_test_files()
    args = parse_arguments(argv)
    phlag = Phlag(args)

    phlag.run()
    phlag.save_output()

    if args.plot and "em" in args.plot:
        PhlagPlotter(phlag)


if __name__ == "__main__":
    main()