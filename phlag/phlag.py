import sys
import pathlib
import os
import argparse

import jax
import numpy as np
import jax.numpy as jnp

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
# Chosen independently of any observed run's data: clipping engaging on more
# than a quarter of all LM step attempts over a fit means the optimizer is
# being bottlenecked by the step-norm safety cap rather than occasionally
# catching a rare bad step.
GRADIENT_CLIP_UNSAFE_RATE_THRESHOLD = 0.25


def get_topology_names(dim):
    """Maps an emission dimension count to human-readable topology labels."""
    if dim == 3:
        return ["ABBA", "BABA", "AABB"]
    return [f"Coord {i+1}" for i in range(dim)]


def get_state_mu_sigma_pdf(params, model_design, state, dim, x_vals):
    """Extracts (mu, sigma, pdf_vals) for a given HMM state/dimension, branching on model_design."""
    if model_design == "gmm":
        w = np.array(params.emissions.mixture_weights[state])
        m_means = np.array(params.emissions.means[state, :, dim])
        m_vars = np.array(params.emissions.covariances[state, :, dim, dim])
        m_stds = np.sqrt(np.clip(m_vars, a_min=1e-6, a_max=None))
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
    """Formats a report table numeric value: up to 3 decimals, none if >=1000."""
    from .utils import format_number
    return format_number(value)


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


def _cluster_mean_cov(pts, D, eps=1e-4):
    """Mean vector and (jitter-regularized) full covariance matrix of a point cluster."""
    mu = np.mean(pts, axis=0)
    if len(pts) > 1:
        cov = np.atleast_2d(np.cov(pts, rowvar=False))
    else:
        cov = np.zeros((D, D))
    cov = cov + np.eye(D) * eps
    return mu, cov


def determine_optimal_mixtures(caster_scores_path, Y, pos_to_caster, silhouette_threshold, output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_positions = sorted(pos_to_caster.keys())
    positions_kb = np.array(sorted_positions) / 1000.0

    D = Y.shape[-1]
    num_mixtures_matrix = np.zeros((NUM_STATES, D), dtype=int)
    dim_cluster_info = {}

    print("\n=== K-means Clustering & Silhouette Scores ===")

    topology_names = ["ABBA", "BABA", "AABB"] if Y.shape[-1] == 3 else [f"Coord_{i+1}" for i in range(Y.shape[-1])]

    # Topologies are always clustered together (shared multivariate mixture
    # components with full covariance across dims), never per-dimension.
    dims_to_iterate = [None]

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

            null_params = []
            if best_kn == 1:
                mu_n, cov_n = _cluster_mean_cov(y_sub_N, D)
                null_params.append((1.0, mu_n, cov_n))
            else:
                km_n = KMeans(n_clusters=best_kn, random_state=42, n_init="auto")
                sub_labels_n = km_n.fit_predict(y_sub_N)
                for m in range(best_kn):
                    pts = y_sub_N[sub_labels_n == m]
                    if len(pts) > 0:
                        w = len(pts) / len(y_sub_N)
                        mu, cov = _cluster_mean_cov(pts, D)
                    else:
                        w = 1.0 / best_kn
                        mu = km_n.cluster_centers_[m]
                        cov = np.eye(D) * 1e-4
                    null_params.append((w, mu, cov))

            alt_params = []
            if best_ka == 1:
                mu_a, cov_a = _cluster_mean_cov(y_sub_A, D)
                alt_params.append((1.0, mu_a, cov_a))
            else:
                km_a = KMeans(n_clusters=best_ka, random_state=42, n_init="auto")
                sub_labels_a = km_a.fit_predict(y_sub_A)
                for m in range(best_ka):
                    pts = y_sub_A[sub_labels_a == m]
                    if len(pts) > 0:
                        w = len(pts) / len(y_sub_A)
                        mu, cov = _cluster_mean_cov(pts, D)
                    else:
                        w = 1.0 / best_ka
                        mu = km_a.cluster_centers_[m]
                        cov = np.eye(D) * 1e-4
                    alt_params.append((w, mu, cov))

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
                mu_base, cov_base = _cluster_mean_cov(y_d, D)
                std_base = np.sqrt(np.clip(np.diag(cov_base), a_min=1e-4, a_max=None))
                null_params.append((1.0, mu_base, cov_base))
                alt_params.append((1.0, mu_base + std_base, cov_base))
            else:
                km = KMeans(n_clusters=m_val, random_state=42, n_init="auto")
                labels_m = km.fit_predict(y_d)
                for m in range(m_val):
                    pts = y_d[labels_m == m]
                    if len(pts) > 0:
                        w = len(pts) / len(y_d)
                        mu, cov = _cluster_mean_cov(pts, D)
                    else:
                        w = 1.0 / m_val
                        mu = km.cluster_centers_[m]
                        cov = np.eye(D) * 1e-4
                    std = np.sqrt(np.clip(np.diag(cov), a_min=1e-4, a_max=None))
                    null_params.append((w, mu, cov))
                    alt_params.append((w, mu + std, cov))

            dim_cluster_info[d] = (null_params, alt_params)

    max_m = int(np.max(num_mixtures_matrix))
    init_means = np.zeros((NUM_STATES, max_m, D), dtype=np.float32)
    init_covariances = np.zeros((NUM_STATES, max_m, D, D), dtype=np.float32)
    init_weights = np.zeros((NUM_STATES, max_m), dtype=np.float32)

    null_params, alt_params = dim_cluster_info[None]
    for s_idx, state_params in enumerate([null_params, alt_params]):
        m_cnt = len(state_params)
        for m_idx, (w, mu, cov) in enumerate(state_params):
            init_weights[s_idx, m_idx] = w
            init_means[s_idx, m_idx, :] = mu
            init_covariances[s_idx, m_idx, :, :] = cov
        sum_w = np.sum(init_weights[s_idx, :m_cnt])
        if sum_w > 0:
            init_weights[s_idx, :m_cnt] /= sum_w

    gmm_init_params = (init_weights, init_means, init_covariances)
    return num_mixtures_matrix, gmm_init_params


class Phlag:
    def __init__(self, args):
        self.args = args
        self.args.increment_steps = 5

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
        # model_design has no CLI flag -- inferred from the input/output
        # filename, falling back to the "gaussian" default (see parse_arguments)
        # if neither filename carries a hint.
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
        For --pair's chunk_scores.tsv (which also carries q1/q2/q3, the
        normalized ABBA/BABA/AABB proportions -- see CasterPlotter's own
        preference for the same columns), q1/q2/q3 are used instead of the
        unnormalized c*ABBA/c*BABA/c*AABB sums, whose scale grows with window
        size and isn't comparable to a Gaussian/GMM emission model's expectations.
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
        self.source_fasta_path = None

        with open(path, "r") as f:
            header = f.readline()
            if not header:
                return

            header_parts = header.strip().split("\t") if "\t" in header else header.strip().split()
            lower_parts = [h.lower() for h in header_parts]
            is_file_header = bool(header_parts) and lower_parts[0] == "file"
            is_locus_header = bool(header_parts) and lower_parts[0] == "locus"

            if all(q in lower_parts for q in ("q1", "q2", "q3")):
                score_indices = [lower_parts.index("q1"), lower_parts.index("q2"), lower_parts.index("q3")]
            else:
                abba_idx = next((i for i, n in enumerate(lower_parts) if 'abba' in n), None)
                baba_idx = next((i for i, n in enumerate(lower_parts) if 'baba' in n), None)
                aabb_idx = next((i for i, n in enumerate(lower_parts) if 'aabb' in n), None)
                if abba_idx is not None and baba_idx is not None and aabb_idx is not None:
                    score_indices = [abba_idx, baba_idx, aabb_idx]
                else:
                    score_indices = [2, 3, 4] if (is_file_header or is_locus_header) else [1, 2, 3]

            pos_idx = lower_parts.index("pos") if "pos" in lower_parts else (1 if (is_file_header or is_locus_header) else 0)

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
                    if self.source_fasta_path is None and is_file_header:
                        self.source_fasta_path = values[0]
                except (ValueError, IndexError):
                    continue

        if len(self.pos_to_caster) == 0:
            sys.exit(f"Error: No valid window scores parsed from CASTER score file '{path}'. Please check that sequence headers in the FASTA match the species in the mapping file.")

    def compute_emissions(self):
        sorted_positions = sorted(self.pos_to_caster.keys())
        raw_caster_matrix = jnp.stack([self.pos_to_caster[pos] for pos in sorted_positions], axis=0)
        self.Y = raw_caster_matrix

    def get_default_out_dir(self):
        import re
        input_path = pathlib.Path(self.args.caster_scores)
        dist_type = getattr(self.args, "model_design", "gaussian")
        from .utils import parse_filename_to_dir_structure, get_repo_root, get_data_dir, get_phlag_output_base, get_short_sim_name
        parsed = parse_filename_to_dir_structure(input_path.stem)

        if not getattr(self.args, "bench", False):
            # Standalone use (the default): flat <repo_root>/out/<node_name>/,
            # no dist_type/window/category/pattern nesting, so ad-hoc runs
            # never write into the tree benchmark runs share/read. The scores
            # file may come from either layout -- caster.py's own flat
            # <node_name>/scores.tsv (the common case; node_name is right
            # there as the parent dir), or (found via -r's broadened search,
            # or passed explicitly) the canonical --bench tree's
            # <node_name>/<pattern>/scores.tsv under a caster/ ancestor,
            # where node_name sits one level further up -- get_simulation_node_name
            # handles that fixed two-directories-up offset. parsed (synthetic
            # null/alt filenames) takes precedence over both.
            if parsed:
                node_name = get_short_sim_name(parsed["alt"])
            elif "caster" in input_path.parts:
                from .utils import get_simulation_node_name
                node_name = get_simulation_node_name(input_path) or input_path.parent.name
            else:
                node_name = input_path.parent.name
            # caster --pair nests its output one level deeper, under
            # out/c<chunk>_s<step>/<node_name>/ (see caster.py) -- preserve
            # that prefix so report.tsv/plots land alongside it instead of
            # flattening back to out/<node_name>/.
            grandparent_name = input_path.parent.parent.name
            if not parsed and re.fullmatch(r'c\w+_s\w+', grandparent_name):
                return get_repo_root() / "out" / grandparent_name / node_name
            return get_repo_root() / "out" / node_name

        # --bench (set only by benchmark's own subprocess invocations) keeps
        # output in the shared canonical tree.
        # --output-base replaces the usual '<model-design>/w<W>_s<S>' prefix
        # wholesale -- it already carries whatever variant/window/step
        # structure the caller wants.
        output_base = getattr(self.args, "output_base", None)
        dist_prefix = pathlib.PurePosixPath(output_base) if output_base else pathlib.PurePosixPath(dist_type)

        conn_env = os.environ.get("CONNECTION_DIR")
        if conn_env:
            base_dir = pathlib.Path(conn_env)
        else:
            base_dir = get_data_dir()
            if base_dir == get_repo_root() / "caster":
                base_dir = get_repo_root() / "connection_dir"

        phlag_base = get_phlag_output_base(base_dir)

        if parsed:
            rel_dir = parsed["relative_dir"]
            out_dir = phlag_base / dist_prefix / rel_dir
        else:
            # scores.tsv lives under a canonical, --output-base/dist_type-
            # independent store/caster/w<W>_s<S>/... tree (see caster.py's own
            # output-path derivation) -- report.tsv's location is still
            # base-dependent, so all phlag needs from the input path is the
            # w<W>_s<S> segment (immediately after caster/) plus the
            # category/subcategory/sim_name/pattern segments that follow it,
            # re-rooted under phlag's own dist_prefix. Where caster/ itself
            # sat is otherwise irrelevant, EXCEPT: when there's no
            # --output-base, dist_prefix is dist_type alone (no window/step),
            # so the 'w<W>_s<S>' segment has to be pulled from the path
            # explicitly to keep report.tsv under the usual
            # '<dist_type>/w<W>_s<S>/...' tree instead of silently losing it.
            parts = input_path.parts
            caster_idx = None
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "caster":
                    caster_idx = i
                    break
            if caster_idx is not None and caster_idx + 1 < len(parts) - 1:
                w_s_part = parts[caster_idx + 1]
                rel_parts = list(parts[caster_idx + 2:-1])
            else:
                w_s_part = None
                rel_parts = []
            if rel_parts:
                if not output_base and w_s_part is not None and re.match(r'w\w+_s\w+', w_s_part):
                    dist_prefix = pathlib.PurePosixPath(dist_type) / w_s_part
                sub_path = pathlib.Path(*rel_parts)
                out_dir = phlag_base / dist_prefix / sub_path
            else:
                from .utils import get_simulation_categories, get_short_sim_name
                cats = get_simulation_categories(input_path.stem)
                pattern_name = get_short_sim_name(input_path.stem)
                cat_prefix = pathlib.Path(*cats) if cats else pathlib.Path()
                out_dir = phlag_base / dist_prefix / cat_prefix / pattern_name
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
        )

        self.num_mixtures = int(np.max(num_mixtures_matrix))
        self.mixture_masks = np.zeros((NUM_STATES, self.num_mixtures), dtype=np.float32)
        for s in range(NUM_STATES):
            m_count = num_mixtures_matrix[s, 0]
            self.mixture_masks[s, :m_count] = 1.0

        self.mixture_masks = jnp.array(self.mixture_masks)
        
        print(f"\nFinal configuration: num_mixtures = {self.num_mixtures}, mixture_masks = \n{self.mixture_masks}\n")

    def initialize_output(self):
        if getattr(self.args, "output_file", None):
            self.output_file = pathlib.Path(self.args.output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = self.get_default_out_dir()
            if getattr(self.args, "bench", False):
                # Flat '<pattern>.tsv' -- out_dir.name is always the pattern
                # (every get_default_out_dir branch ends with it as the last
                # segment) -- matching the existing benchmark reports/ archive
                # shape exactly, so no migration is needed for prior runs.
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                self.output_file = out_dir.parent / f"{out_dir.name}.tsv"
            else:
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

        # Squared Hellinger distance between states -- gaussian-only (depends
        # on the full covariance matrix; beta/gmm emissions have no analog).
        em_hellinger2_distance = None
        if getattr(self.args, "model_design", "gaussian") == "gaussian":
            em_hellinger2_distance = float(self.hmm.em_divergence(self.params))


        log_likelihoods = []
        for state in range(self.hmm.num_states):
            dist = self.hmm.emission_component.distribution(self.params.emissions, state)
            log_likelihoods.append(dist.log_prob(self.Y))
        log_likelihoods = jnp.stack(log_likelihoods, axis=-1)
        
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

        # 0. Explicit --locus-pattern override (needed once caster.py's flat
        # standalone output stopped encoding the pattern in scores.tsv's path).
        pattern_str = getattr(self.args, "locus_pattern", None)
        # 1. Try to find the pattern in the path parts (matches [an]\d+ blocks, range syntax, or start-end coords)
        pattern_full_match_regex = r'(?:[an]\d+)+(?:[_,]\d+-\d+(?:[_,]\d+-\d+)*)?|(?:[an]\d+(?:-[an]?\d+)?(?:[_,])?)+|\d+-\d+(?:[_,]\d+-\d+)*'
        if not pattern_str:
            for part in reversed(input_path_obj.parts):
                if re.fullmatch(pattern_full_match_regex, part):
                    pattern_str = part
                    break

        # 2. Try the file stem
        if not pattern_str:
            pattern_str_match = re.search(r'((?:[an]\d+)+(?:[_,]\d+-\d+(?:[_,]\d+-\d+)*)?|(?:[an]\d+(?:-[an]?\d+)?(?:[_,])?){2,}|\d+-\d+(?:[_,]\d+-\d+)*)', input_path_obj.stem)
            if pattern_str_match:
                pattern_str = pattern_str_match.group(1)

        # 3. Try the source FASTA path caster recorded in scores.tsv's own
        # 'file' column (read_caster_scores captures it as
        # self.source_fasta_path) -- still encodes the pattern even once
        # caster_scores' own path/filename stops doing so, e.g. the flat
        # standalone out/<node>/scores.tsv layout.
        if not pattern_str and getattr(self, "source_fasta_path", None):
            source_path_obj = pathlib.Path(self.source_fasta_path)
            for part in reversed(source_path_obj.parts):
                if re.fullmatch(pattern_full_match_regex, part):
                    pattern_str = part
                    break
            if not pattern_str:
                pattern_str_match = re.search(r'((?:[an]\d+)+(?:[_,]\d+-\d+(?:[_,]\d+-\d+)*)?|(?:[an]\d+(?:-[an]?\d+)?(?:[_,])?){2,}|\d+-\d+(?:[_,]\d+-\d+)*)', source_path_obj.stem)
                if pattern_str_match:
                    pattern_str = pattern_str_match.group(1)

        # No pattern found anywhere (and none given explicitly): evaluation
        # against ground truth isn't possible -- degrade gracefully (skip
        # metrics/ground-truth plots) instead of erroring, since standalone
        # runs on non-simulation data or flat-output paths legitimately have
        # no ground truth to compare against.
        self.has_ground_truth = False
        if not pattern_str:
            print(f"Warning: No ground truth locus pattern found for '{input_path_obj}' (and no --locus-pattern given) -- skipping evaluation metrics and ground-truth plots.")
        else:
            from .utils import parse_pattern_string
            total_span = sorted_positions[-1] if sorted_positions else None
            blocks, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)
            if not blocks:
                print(f"Warning: Could not parse ground truth locus pattern '{pattern_str}' from '{input_path_obj}' -- skipping evaluation metrics and ground-truth plots.")
            else:
                self.has_ground_truth = True
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
            
            is_auto = correct_trans_arg == "auto" or (isinstance(correct_trans_arg, str) and correct_trans_arg.lower() == "auto")
            if is_auto and not self.has_ground_truth:
                print("Warning: --correct-transition auto requested but no ground truth pattern is available -- skipping.")
            elif is_auto:
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

        n_paths = getattr(self.args, "best_paths", 1)
        paths, path_likelihoods = self.get_n_best_viterbi_paths(
            initial_probs_np, transition_matrix_np, log_likelihoods_np, n_paths
        )

        from .utils import format_number

        # Calculate metrics for primary Viterbi path (Path 1) -- only
        # meaningful with a real ground truth to compare against; without
        # one, skip evaluation entirely (flipping to "match" an all-null
        # y_true would silently relabel states based on nothing) rather than
        # report misleading numbers.
        y_pred = np.array(paths[0])
        flipped_for_eval = False
        tp = fp = fn = tn = 0

        if self.has_ground_truth:
            # Flip the state assignments according to whichever has smaller hamming distance
            hamming_dist = np.sum(y_pred != y_true)
            hamming_dist_flipped = np.sum((1 - y_pred) != y_true)

            if hamming_dist_flipped < hamming_dist:
                y_pred = 1 - y_pred
                flipped_for_eval = True

            # Per-window posterior P(Alt) from the same forward-backward smoother
            # fit_em's E-step already runs -- reused here (one extra pass on the
            # final params) to sweep a real ROC AUC instead of a hard-decision
            # proxy. Oriented the same way y_pred was flipped above, so state
            # index 1's posterior always means "Alt" for evaluation purposes.
            inference_args = self.hmm._inference_args(self.params, self.Y, None)
            posterior = hmm.hmm_two_filter_smoother(*inference_args)
            alt_scores = np.array(posterior.smoothed_probs[:, 1])
            if flipped_for_eval:
                alt_scores = 1.0 - alt_scores

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
            f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_true, alt_scores))
            except ValueError:
                # y_true has only one class (all-Null or all-Alt window set) --
                # no ROC curve to sweep, fall back to the single-operating-point
                # estimate the hard Viterbi confusion matrix already gives.
                auc = (tpr + (1.0 - fpr)) / 2.0

            metrics_str = (
                f"TPR: {format_number(tpr)}, FPR: {format_number(fpr)}, "
                f"Precision: {format_number(precision)}, F1: {format_number(f1)}, "
                f"Accuracy: {format_number(accuracy)}, AUC: {format_number(auc)}"
            )
        else:
            metrics_str = "N/A (no ground truth pattern)"
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
                headers.append(f"Branch length (CU, node '{clade_name}'): {format_number(branch_len)}")
        else:
            headers.append("Branch length (CU): N/A")

        if pop_info:
            try:
                headers.append(f"Height (Ngen): {format_number(float(pop_info.get('HEIGHT_NGEN')))}")
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
        # consumers (bench.benchmark) read the anomaly fraction and the eval-time
        # label polarity straight out of the report, instead of re-deriving them from
        # the locus pattern -- the fraction in particular is not a pure function of
        # the pattern string, since it depends on where the actual window grid
        # (sorted_positions) falls inside the anomaly intervals.
        if self.has_ground_truth:
            n_windows = int(len(y_true))
            n_anomaly = int(np.sum(y_true == 1))
            anomaly_fraction = (n_anomaly / n_windows) if n_windows > 0 else 0.0
            headers.append(f"Anomaly fraction: {format_number(anomaly_fraction)} ({n_anomaly}/{n_windows} windows)")
            headers.append(f"Label polarity flipped for evaluation: {flipped_for_eval}")
        else:
            headers.append("Anomaly fraction: N/A (no ground truth pattern)")
            headers.append("Label polarity flipped for evaluation: N/A (no ground truth pattern)")

        # Source mtimes at report-generation time -- let bench.benchmark's run_all()
        # detect a stale report (caster.py or phlag.py/hmm.py edited since this report
        # was written) and rerun just the stage whose source actually changed, instead
        # of relying purely on report.tsv/scores.tsv presence.
        _pkg_dir = pathlib.Path(__file__).parent
        _caster_mtime = (_pkg_dir / "caster.py").stat().st_mtime
        _phlag_mtime = max((_pkg_dir / f).stat().st_mtime for f in ("phlag.py", "hmm.py"))
        headers.append(f"Caster source mtime: {_caster_mtime:.6f}")
        headers.append(f"Phlag source mtime: {_phlag_mtime:.6f}")

        headers.append("State divergence: " + emission_divergence_str)
        if em_hellinger2_distance is not None:
            headers.append(f"EM divergence (Hellinger^2): {format_number(em_hellinger2_distance)}")
        headers.append(f"Outer EM iterations: {self.n_iters}")
        headers.append(f"Inner EM iterations: {self.increment_steps}")
        if hasattr(self, "initial_transition_matrix") and self.initial_transition_matrix is not None:
            tm_before_str = ", ".join(f"[{', '.join(format_number(x) for x in row)}]" for row in self.initial_transition_matrix.tolist())
            headers.append(f"Initial transition matrix (before EM): [{tm_before_str}]")
        tm_after_str = ", ".join(f"[{', '.join(format_number(x) for x in row)}]" for row in transition_matrix_np.tolist())
        headers.append(f"Final transition matrix (after EM): [{tm_after_str}]")
        if correct_trans_arg is not None:
            headers.append("Corrected transition matrix applied: True")

        if hasattr(self, "final_em_log_prob"):
            headers.append(f"EM final joint log-likelihood: {format_number(self.final_em_log_prob)}")
        marginal_ll = float(self.hmm.marginal_log_prob(self.params, self.Y))
        k_params = count_trainable_params(self.params, self.props)
        n_obs = int(self.Y.shape[0])
        bic = k_params * np.log(n_obs) - 2 * marginal_ll
        headers.append(
            f"BIC: {format_number(bic)} (log-likelihood={format_number(marginal_ll)}, "
            f"k={k_params} trainable params, n={n_obs} windows)"
        )
        if hasattr(self, "clip_activation_count") and self.clip_activation_attempts > 0:
            clip_rate = self.clip_activation_count / self.clip_activation_attempts
            clip_unsafe = clip_rate > GRADIENT_CLIP_UNSAFE_RATE_THRESHOLD
            headers.append(
                f"Gradient clip activations: {self.clip_activation_count} "
                f"(rate: {format_number(clip_rate)}, unsafe: {clip_unsafe})"
            )

        headers.append(f"{metrics_str}")
        if self.ground_truth_fits:
            topology_names = get_topology_names(self.Y.shape[-1])
            headers.append("Topology\tState\tStatistic\tFitted\tGroundTruth\tRel.err")
            # State index 0/1 is an arbitrary EM cluster label, not a fixed
            # Null/Alt identity -- flipped_for_eval (established above from
            # which orientation matches y_true) says which physical state
            # behaves like Null vs Alt, so route each into the ground-truth
            # comparison it actually corresponds to instead of assuming
            # state 0 is always Null.
            null_state, alt_state = (1, 0) if flipped_for_eval else (0, 1)
            # Per-topology (marginal, univariate) ground-truth squared
            # Hellinger distance, averaged across topologies -- the
            # ground-truth split only gives per-topology mean/std (no
            # cross-topology covariance), so unlike the fitted EM divergence
            # (joint over all topologies at once) this is an average of
            # marginals. Same measure hmm.PhlagHMMEmissions.em_divergence
            # uses (bounded [0, 1], 0 = statistically identical, 1 = fully
            # separated), so the two numbers are on a comparable scale.
            gt_divergences = []
            for d in sorted(self.ground_truth_fits.keys()):
                mu_null_gt, std_null_gt, mu_alt_gt, std_alt_gt = self.ground_truth_fits[d]
                mu_null_fit, std_null_fit, _ = get_state_mu_sigma_pdf(self.params, self.args.model_design, null_state, d, np.array([0.0]))
                mu_alt_fit, std_alt_fit, _ = get_state_mu_sigma_pdf(self.params, self.args.model_design, alt_state, d, np.array([0.0]))
                rel_mu_null = (mu_null_fit - mu_null_gt) / mu_null_gt if mu_null_gt != 0 else float('nan')
                rel_std_null = (std_null_fit - std_null_gt) / std_null_gt if std_null_gt != 0 else float('nan')
                rel_mu_alt = (mu_alt_fit - mu_alt_gt) / mu_alt_gt if mu_alt_gt != 0 else float('nan')
                rel_std_alt = (std_alt_fit - std_alt_gt) / std_alt_gt if std_alt_gt != 0 else float('nan')
                topo_name = topology_names[d] if d < len(topology_names) else f"Coord {d+1}"
                headers.append(f"{topo_name}\tNull\tmean\t{format_rel_err(mu_null_fit)}\t{format_rel_err(mu_null_gt)}\t{format_rel_err(rel_mu_null)}")
                headers.append(f"{topo_name}\tNull\tstd\t{format_rel_err(std_null_fit)}\t{format_rel_err(std_null_gt)}\t{format_rel_err(rel_std_null)}")
                headers.append(f"{topo_name}\tAlt\tmean\t{format_rel_err(mu_alt_fit)}\t{format_rel_err(mu_alt_gt)}\t{format_rel_err(rel_mu_alt)}")
                headers.append(f"{topo_name}\tAlt\tstd\t{format_rel_err(std_alt_fit)}\t{format_rel_err(std_alt_gt)}\t{format_rel_err(rel_std_alt)}")
                hellinger2_gt = hmm.gaussian_hellinger2(
                    jnp.array([mu_null_gt]), jnp.array([[std_null_gt ** 2]]),
                    jnp.array([mu_alt_gt]), jnp.array([[std_alt_gt ** 2]]),
                )
                gt_divergences.append(float(hellinger2_gt))
            if gt_divergences:
                headers.append(f"Ground truth EM divergence (Hellinger^2): {format_number(sum(gt_divergences) / len(gt_divergences))}")
        for idx, l in enumerate(path_likelihoods):
            headers.append(f"Path {idx + 1} final joint log-likelihood: {format_number(l[-1])}")

        headers.append("--- Bookkeeping ---")
        headers.append(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

        self.output_str += "\n" + "\n".join(headers)

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
                
                if self.has_ground_truth:
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

    def initialize_hmm(self):
        # Prior hyperparameters
        self.emission_lambda = self.args.emission_lambda
        self.nu = 1.1

        # Generic Dirichlet structure setup for continuous emission tracking
        self.psi = jnp.ones((NUM_STATES, NUM_STATES)) + PSI_EPS

        self.emission_parameterization = (
            self.args.null_emission_parameterization,
            self.args.alt_emission_parameterization,
        )

        self.hmm = hmm.PhlagHMM(
            NUM_STATES,
            self.Y.shape[-1],
            emission_lambda=self.emission_lambda,
            emission_parameterization=self.emission_parameterization,
            initial_probs_concentration=self.nu,
            transition_concentration=self.psi,
            model_design=self.args.model_design,
            num_mixtures=getattr(self, "num_mixtures", 2),
            lm_damping=self.args.lm_damping,
            repulsion_optimizer=self.args.repulsion_optimizer,
            penalty_lambda_anneal=self.args.annealing,
            n_iters=self.args.n_iters,
            increment_steps=self.args.increment_steps,
        )
        if self.args.model_design == "gmm":
            self.hmm.emission_component.mixture_masks = self.mixture_masks

        # Compute empirical moments from CASTER data over genomic positions
        data_mean = jnp.mean(self.Y, axis=0)
        data_cov = jnp.cov(self.Y, rowvar=False)
        data_var = jnp.diag(data_cov)

        # alt variance = double null variance if --double-variance-init, else same as null
        alt_variance_mult = 2.0 if self.args.double_variance_init else 1.0
        p0, p1 = 0.99, 0.99
        initial_transition_matrix = jnp.array([[p0, 1-p0], [1-p1, p1]], dtype=jnp.float32)

        if self.args.model_design == "gmm":
            # Only used as a fallback seed if gmm_init_params wasn't computed
            # (determine_optimal_mixtures normally supplies it instead, full
            # covariance included) -- diagonal here is just a degenerate seed.
            state0_init = jnp.stack([data_mean, data_var], axis=-1)
            state1_init = jnp.stack([data_mean, alt_variance_mult * data_var], axis=-1)
            init_emissions = jnp.stack([state0_init, state1_init], axis=0)
            self.params, self.props = self.hmm.initialize(
                initial_probs=INITIAL_PROBS,
                emission_probs=init_emissions,
                transition_matrix=initial_transition_matrix,
                initial_gmm_params=getattr(self, "gmm_init_params", None),
            )
        else:
            # Shape: [num_states, emission_dim, 2] where column 0 is the mean and
            # column 1 is the variance -- hmm.py's initialize() builds the actual
            # (diagonal) covariance matrix from this itself.
            state0_init = jnp.concatenate([data_mean[:, None], data_cov], axis=-1) # CHANGED FROM data_cov and data_mean[:,None]
            state1_init = jnp.concatenate([data_mean[:, None], alt_variance_mult * data_cov], axis=-1)
            init_emissions = jnp.stack([state0_init, state1_init], axis=0)
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
            self.props.emissions.covariances.trainable = True
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

        cumulative_inner_steps = 0
        self.clip_activation_count = 0
        self.clip_activation_attempts = 0
        for i in tqdm(range(self.n_iters)):
            num_inner = (i + 1) * self.increment_steps
            self.params, log_probs, clip_count, clip_attempts = self.hmm.fit_em(
                self.params,
                self.props,
                self.Y,
                num_iters=num_inner,
                verbose=False,
                # Annealing's decay clock runs on actual EM steps completed,
                # not outer-loop index -- since num_inner grows every outer
                # iteration (arithmetic sequence), an outer index of e.g. 3
                # doesn't mean the same "amount of training" regardless of
                # where it falls in the schedule.
                outer_iter=cumulative_inner_steps if self.args.annealing else None,
            )
            cumulative_inner_steps += num_inner
            self.clip_activation_count += clip_count
            self.clip_activation_attempts += clip_attempts
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
        self.plot_topologies_3d()

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
        elif self.phlag.ground_truth_fits:
            self._plot_ground_truth_row(axes[0], ranges)
        else:
            for d in range(self.emission_dim):
                axes[0][d].text(0.5, 0.5, "No ground truth available", ha="center", va="center", transform=axes[0][d].transAxes, fontsize=10, color="gray")
                axes[0][d].set_xticks([])
                axes[0][d].set_yticks([])

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
        init_weights, init_means, init_covariances = self.phlag.gmm_init_params
        title_prefix = "GMM Initialization"

        for d in range(self.emission_dim):
            ax = row_axes[d]
            x_vals = ranges[d]
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

            for state in [0, 1]:
                color_config = self.colors[state]
                w = np.array(init_weights[state])
                m_means = np.array(init_means[state, :, d])
                m_vars = np.array(init_covariances[state, :, d, d])
                m_stds = np.sqrt(np.clip(m_vars, 1e-6, None))
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

    def _covariance_ellipsoid_surface(self, ax, mean, cov, color, n_std=1.0):
        eigenvalues, eigenvectors = np.linalg.eigh(np.array(cov))
        radii = n_std * np.sqrt(np.clip(eigenvalues, a_min=0, a_max=None))

        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 24)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        sphere = np.stack([x, y, z], axis=-1)
        transform = eigenvectors @ np.diag(radii)
        ellipsoid = sphere @ transform.T + np.array(mean)

        ax.plot_surface(
            ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
            color=color, alpha=0.15, linewidth=0, shade=True,
        )

    def plot_topologies_3d(self):
        if self.emission_dim != 3 or self.phlag.args.model_design != "gaussian":
            return

        params = self.phlag.params
        Y_np = np.array(self.phlag.Y)
        y_true = np.array(self.phlag.y_true)
        most_likely_states = np.array(self.phlag.hmm.most_likely_states(params, self.phlag.Y))

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

        stages = [
            ("Before EM", y_true),
            ("After EM", most_likely_states),
        ]

        for col_idx, (stage_label, state_labels) in enumerate(stages):
            ax = axes[col_idx]

            for state in [0, 1]:
                color_config = self.colors[state]
                mask = state_labels == state

                ax.scatter(
                    Y_np[mask, 0], Y_np[mask, 1], Y_np[mask, 2],
                    s=8, alpha=0.25, color=color_config['fill'], linewidths=0,
                    label=color_config['label'],
                )

                if mask.sum() > 0:
                    if stage_label == "After EM":
                        mean = np.array(params.emissions.means[state])
                        cov = np.array(params.emissions.covariances[state])
                    else:
                        mean = np.mean(Y_np[mask], axis=0)
                        cov = np.cov(Y_np[mask], rowvar=False)

                    self._covariance_ellipsoid_surface(ax, mean, cov, color_config['line'])

            ax.set_title(stage_label, fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel(self.topology_names[0], fontsize=8, labelpad=4)
            ax.set_ylabel(self.topology_names[1], fontsize=8, labelpad=4)
            ax.set_zlabel(self.topology_names[2], fontsize=8, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.legend(fontsize=8, loc='upper right')

        from .utils import get_locus_description
        locus_desc = get_locus_description(self.phlag.args.caster_scores)
        if locus_desc:
            fig.suptitle(f"Topology Score Space (3D) | {locus_desc}", fontsize=13, fontweight="bold", y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        self.save_plot("topologies_3d.png")

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

def build_parser():
    parser = argparse.ArgumentParser(
        description="Phlag: Detecting genomic regions with unexplained phylogenetic heterogeneity using CASTER"
    )
    # model_design has no CLI flag -- it's inferred from the input/output
    # filename (see Phlag.extract_distribution_type_from_filename), falling
    # back to this default if neither filename carries a "gaussian"/"gmm" hint.
    parser.set_defaults(model_design="gaussian")

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
        default="free",
        choices=["free", "repulsion"],
        help="""Parameterization of the alt (anomalous) state's emission distribution
                    (default: free): free (unconstrained MLE) or repulsion (pushed
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
        "--double-variance-init",
        dest="double_variance_init",
        action="store_true",
        help="Seed the alt state's initial variance at 2x the null state's (default: same variance as null).",
    )
    hmm_group.add_argument(
        "--repulsion-optimizer",
        dest="repulsion_optimizer",
        type=str.lower,
        default="lm",
        choices=["lm", "gd"],
        help="""Optimizer the REPULSION emission parameterization's joint MAP fit
                    uses (default: lm): lm (Levenberg-Marquardt/damped Newton, see
                    --mu) or gd (plain fixed-step gradient descent).""",
    )
    hmm_group.add_argument(
        "--annealing",
        dest="annealing",
        action="store_true",
        help="""Anneal the REPULSION emission parameterization's penalty lambda across
                    outer EM iterations, "budget mode": the schedule
                    penalty_lambda * (1 + 2*exp(-t/tau)) starts elevated and decays as t
                    grows (t = inner EM steps completed so far, not the outer iteration
                    index -- inner steps per outer iteration grow as an arithmetic
                    sequence, so this tracks actual training progress; tau is a third of
                    the run's total inner steps), then the whole schedule is rescaled so
                    its time-weighted average across the run equals exactly the base
                    --lam value -- --lam is a budget spent unevenly (more early, less
                    late), not a floor the schedule decays toward (default: off, fixed
                    --lam throughout).""",
    )
    hmm_group.add_argument(
        "--mu",
        dest="lm_damping",
        type=float,
        default=1.0,
        help="""Initial damping for the --repulsion-optimizer lm path's joint MAP
                    optimizer (default: 1.0; ignored under gd). Per-step damping
                    still adapts (relaxes toward 0 on accepted steps, grows on
                    rejected ones) starting from this value -- 0 starts at an
                    undamped Newton step, not gradient descent.""",
    )
    hmm_group.add_argument(
        "--output-base",
        dest="output_base",
        default=None,
        help="Override the '<model-design>/w<W>_s<S>' output-path prefix with an "
             "arbitrary relative path (e.g. 'gaussian/repulsion/w50k_s1k'), for "
             "writing into a relocated/variant-specific output tree instead of "
             "the default one",
    )
    hmm_group.add_argument(
        "--locus-pattern",
        dest="locus_pattern",
        default=None,
        help="Explicit ground-truth locus pattern (e.g. '37-62' or 'n1a1n5...') "
             "to evaluate against, for when the score file's path/filename doesn't "
             "encode one (e.g. caster.py's flat standalone output). Falls back to "
             "parsing one out of the score file's path/filename when not given; if "
             "neither succeeds, evaluation metrics and ground-truth plots are "
             "skipped instead of erroring.",
    )
    hmm_group.add_argument(
        "--bench",
        dest="bench",
        action="store_true",
        default=False,
        help="Set by benchmark's run_all() for its own subprocess invocations -- "
             "not meant to be passed by hand. When set, report.tsv is written as a "
             "flat '<output-base>/<pattern>.tsv' file instead of the default "
             "'<pattern>/report.tsv', and the output root is the shared canonical "
             "tree (default: off, writes report.tsv + plots to a flat "
             "<repo_root>/out/<node_name>/ instead of the shared tree, for "
             "standalone use, alongside caster.py's scores.tsv for that node).",
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
    return parser


def parse_arguments(argv=None):
    parser = build_parser()
    args = utils.apply_cli_config(parser, argv, "phlag")

    from .utils import get_data_dir, get_repo_root, resolve_input_file, get_most_recent_file
    repo_root = get_repo_root()
    data_dir = get_data_dir()

    dist_type = getattr(args, "model_design", "gaussian")

    def resolve_model_scores(target_name=None):
        from .utils import get_phlag_output_base
        # Canonical bases (--bench's shared tree) key scores.tsv under a
        # 'caster' ancestor directory, so candidates there are filtered on
        # that. scores.tsv lives in its own store/caster/w<W>_s<S>/... tree
        # (dist_type-independent, not nested under phlag_base at all -- see
        # caster.py). The flat 'out/' base is caster.py's standalone scratch
        # tree (see caster.py) -- everything under it is ours, no 'caster'
        # ancestor to filter on.
        canonical_bases = [
            data_dir / "caster",
            repo_root / "store" / "caster",
        ]
        flat_bases = [repo_root / "out"]

        # flat_candidates (out/) and canonical_candidates are kept in
        # separate pools rather than merged-then-sorted-by-mtime: out/ is the
        # standalone scratch tree a single interactive `phlag -r` session
        # writes into, while the canonical bases are shared with every
        # concurrently-running --bench sweep, whose scores.tsv files get
        # touched far more often -- a raw mtime merge would almost always
        # pick a canonical file that has nothing to do with what -r's user
        # just ran standalone. out/ wins whenever it has anything at all.
        flat_candidates = []
        canonical_candidates = []
        seen = set()

        def add(pool, sfile):
            if sfile.resolve() not in seen:
                seen.add(sfile.resolve())
                pool.append(sfile)

        for b in canonical_bases:
            if not b.exists():
                continue
            if target_name:
                # Look for target_name subdirectory anywhere under b -- under
                # the canonical caster/ layout that's always somewhere below
                # a caster/ ancestor, not a sibling/child of one, so there's
                # no separate '**/caster'-anchored search needed any more.
                target_dirs = [td for td in b.glob(f"**/{target_name}") if td.is_dir()]
                for td in target_dirs:
                    for sfile in td.rglob("scores.tsv"):
                        if "caster" in sfile.parts:
                            add(canonical_candidates, sfile)
                    for sfile in td.rglob("*.tsv"):
                        if "caster" in sfile.parts and not any(sfile.name.startswith(p) for p in ["report_", "em_", "states_"]):
                            add(canonical_candidates, sfile)
            else:
                for sfile in b.rglob("scores.tsv"):
                    if "caster" in sfile.parts:
                        add(canonical_candidates, sfile)

        for b in flat_bases:
            if not b.exists():
                continue
            if target_name:
                for td in b.glob(f"**/{target_name}"):
                    if td.is_dir():
                        for sfile in td.rglob("scores.tsv"):
                            add(flat_candidates, sfile)
                        for sfile in td.rglob("*.tsv"):
                            if not any(sfile.name.startswith(p) for p in ["report_", "em_", "states_"]):
                                add(flat_candidates, sfile)
            else:
                # chunk_scores.tsv is --pair's scores.tsv-equivalent (see
                # caster.py's run_caster_pair) -- included here so `-r` can
                # find it under out/c<chunk>_s<step>/<node_name>/ too.
                for pattern in ("scores.tsv", "chunk_scores.tsv"):
                    for sfile in b.rglob(pattern):
                        add(flat_candidates, sfile)

        for candidates in (flat_candidates, canonical_candidates):
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


def main(argv=None):
    args = parse_arguments(argv)

    if not args.bench:
        flags_str = " ".join(f"{k}={v}" for k, v in vars(args).items())
        print(f"[phlag] Effective flags: {flags_str}")

    phlag = Phlag(args)

    phlag.run()
    phlag.save_output()

    if args.plot and "em" in args.plot:
        PhlagPlotter(phlag)


if __name__ == "__main__":
    main()