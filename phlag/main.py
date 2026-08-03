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
        if not hasattr(self.args, "n_iters"):
            self.args.n_iters = 10
        self.args.increment_steps = 50
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
                if "beta" in fname_lower:
                    self.args.model_design = "beta"
                    break
                elif "gmm" in fname_lower:
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

        if getattr(self.args, "model_design", "gaussian") == "beta":
            raw_clipped = jnp.clip(raw_caster_matrix, a_min=1e-7)
            self.Y = raw_clipped / raw_clipped.sum(axis=-1, keepdims=True)
        else:
            self.Y = raw_caster_matrix

    def get_default_out_dir(self):
        input_path = pathlib.Path(self.args.caster_scores)
        dist_type = getattr(self.args, "model_design", "gaussian")
        from .utils import parse_filename_to_dir_structure, get_repo_root, get_data_dir
        parsed = parse_filename_to_dir_structure(input_path.stem)
        
        conn_env = os.environ.get("CONNECTION_DIR")
        if conn_env:
            base_dir = pathlib.Path(conn_env)
        else:
            base_dir = get_data_dir()
            if base_dir == get_repo_root() / "caster" / "data":
                base_dir = get_repo_root() / "connection_dir"
        
        if parsed:
            rel_dir = parsed["relative_dir"]
            out_dir = base_dir / "phlag" / dist_type / rel_dir / "phlag"
        else:
            if input_path.parent.name == "caster":
                # Find path relative to model design or phlag root if input is in a caster subfolder
                # e.g., .../phlag/gaussian/w50k_s1k/Nyctibiidae_10X_up/n1n2n3n4n5a1a2a3n6n8n10n11/caster/scores.tsv
                # should preserve w50k_s1k/Nyctibiidae_10X_up/n1n2n3n4n5a1a2a3n6n8n10n11 under phlag/<dist_type>/
                locus_dir = input_path.parent.parent
                parts = locus_dir.parts
                # Check if model design or 'phlag' is in parent path parts
                known_models = ["gaussian", "beta", "gmm"]
                rel_parts = []
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i] in known_models or parts[i] == "phlag":
                        rel_parts = list(parts[i+1:])
                        break
                if rel_parts:
                    from .utils import get_simulation_categories, get_short_sim_name
                    for idx, p in enumerate(rel_parts):
                        cats = get_simulation_categories(p)
                        if cats:
                            short_p = get_short_sim_name(p)
                            rel_parts[idx] = short_p
                            if not (cats[0] in rel_parts and cats[1] in rel_parts):
                                rel_parts = rel_parts[:idx] + list(cats) + rel_parts[idx:]
                            break
                    sub_path = pathlib.Path(*rel_parts)
                    out_dir = base_dir / "phlag" / dist_type / sub_path / "phlag"
                else:
                    from .utils import get_simulation_categories, get_short_sim_name
                    cats = get_simulation_categories(locus_dir.name)
                    short_name = get_short_sim_name(locus_dir.name)
                    cat_prefix = pathlib.Path(*cats) if cats else pathlib.Path()
                    out_dir = base_dir / "phlag" / dist_type / cat_prefix / short_name / "phlag"
            else:
                from .utils import get_simulation_categories, get_short_sim_name
                cats = get_simulation_categories(input_path.stem)
                short_name = get_short_sim_name(input_path.stem)
                cat_prefix = pathlib.Path(*cats) if cats else pathlib.Path()
                out_dir = base_dir / "phlag" / dist_type / cat_prefix / short_name / "phlag"
        return out_dir

    def determine_optimal_mixtures(self):
        import sys
        from .utils import get_repo_root
        repo_root = get_repo_root()
        caster_results_dir = repo_root / "caster" / "results"
        if str(caster_results_dir) not in sys.path:
            sys.path.append(str(caster_results_dir))
            
        import caster_plot
        
        if self.args.output_file:
            output_dir = pathlib.Path(self.args.output_file).parent
        else:
            output_dir = self.get_default_out_dir()
            
        output_dir.mkdir(parents=True, exist_ok=True)
            
        num_mixtures_matrix, self.gmm_init_params = caster_plot.determine_optimal_mixtures(
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
        
        # Extract ground truth pattern indices from input filename if present
        # Format example: ...a1n5a2a3n8n1...
        # 'a' = anomaly locus block of 500Kb, 'n' = normal locus block of 500Kb
        # Each token (e.g., 'n1', 'n8', 'a1', 'n5') represents one 500Kb locus block.
        import re
        input_path_obj = pathlib.Path(self.args.caster_scores)
        
        sorted_positions = sorted(self.pos_to_caster.keys())
        y_true = np.zeros(len(sorted_positions), dtype=int)
        has_ground_truth = False
        
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
                
        # 3. Try pattern.txt
        if not pattern_str:
            pattern_file = input_path_obj.parent / "pattern.txt"
            if pattern_file.exists():
                try:
                    content = pattern_file.read_text().strip()
                    p_match = re.search(r'((?:[an]\d+)+(?:[_,]\d+-\d+(?:[_,]\d+-\d+)*)?|(?:[an]\d+(?:-[an]?\d+)?(?:[_,])?)+|\d+-\d+(?:[_,]\d+-\d+)*)', content)
                    if p_match:
                        pattern_str = p_match.group(1)
                except Exception:
                    pass

        if pattern_str:
            from .utils import parse_pattern_string
            total_span = sorted_positions[-1] if sorted_positions else None
            blocks, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)
            if blocks:
                has_ground_truth = True
                for idx, pos in enumerate(sorted_positions):
                    for start_bp, end_bp in anomaly_intervals:
                        if start_bp <= pos <= end_bp:
                            y_true[idx] = 1
                            break

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
                if has_ground_truth:
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
                else:
                    print("Warning: --correct-transition auto requested, but ground truth locus pattern not found in filename or pattern.txt.")

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
        if has_ground_truth:
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
        headers = []
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
        if has_ground_truth:
            headers.append(f"{metrics_str}")
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
                
                # Plot ground truth if available
                if has_ground_truth:
                    ax1.step(positions_kb, y_true, where="mid", color='black', linestyle='--', linewidth=2.0, label="Ground Truth", alpha=0.8)

                ax1.set_xlabel("Genomic Position (kb)", fontsize=12, labelpad=10)
                ax1.set_ylabel("HMM State", fontsize=12, labelpad=10)
                ax1.set_ylim(-0.05, 1.05)
                ax1.set_yticks([0, 1])
                ax1.grid(True, axis='x', linestyle=':', alpha=0.5)
                
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
            model_design=self.args.model_design,
            num_mixtures=getattr(self, "num_mixtures", 2),
        )
        if self.args.model_design == "gmm":
            self.hmm.emission_component.mixture_masks = self.mixture_masks
        
        # Compute empirical moments from CASTER data over genomic positions
        data_mean = jnp.mean(self.Y, axis=0)
        data_std = jnp.std(self.Y, axis=0)
        
        # Implement symmetry breaking: Anchor state 0 to the baseline mean,
        # and seed state 1 slightly further along the alternative dimensions.
        # Initialize emissions using mean and variance for each state.
        if self.args.model_design == "beta":
            state1_mean = jnp.clip(data_mean + data_std, a_min=1e-4, a_max=0.95)
            state0_init = jnp.stack([data_mean, data_std ** 2], axis=-1)
            state1_init = jnp.stack([state1_mean, data_std ** 2], axis=-1)
        else:
            state0_init = jnp.stack([data_mean, data_std ** 2], axis=-1)
            state1_init = jnp.stack([data_mean + data_std, data_std ** 2], axis=-1)

        # Shape: [num_states, emission_dim, 2] where the last axis is [mean, variance]
        init_emissions = jnp.stack([state0_init, state1_init], axis=0)
        p0, p1 = 0.6, 0.6
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
        if self.args.model_design == "beta":
            self.props.emissions.concentration1.trainable = True
            self.props.emissions.concentration0.trainable = True
        elif self.args.model_design == "gmm":
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
        self.compute_output()

    def save_output(self):
        with open(self.output_file, "w") as f:
            f.write(self.output_str)
        print(f"Saved PHLAG output report to: {self.output_file}")



class PhlagPlotter:
    """
    Object-oriented plotter for visualizing the probability distributions
    of standard multi-species coalescent (MSC) background (State 0, Null) vs.
    alternative/anomalous (State 1, Alternative) states before and after EM fitting.
    """
    def __init__(self, phlag, initial_params):
        self.phlag = phlag
        self.initial_params = initial_params
        
        # Ingest and configure metadata parameters
        self.extract_metadata()
        
        # Generate the visual distribution charts
        self.plot_distributions()

    def extract_metadata(self):
        """Extracts genomic filename, dimension, and styles configuration."""
        self.input_path = pathlib.Path(self.phlag.args.caster_scores)
        self.emission_dim = self.phlag.Y.shape[-1]
        
        # Map coordinates to topology names if we have the standard 3 topologies
        if self.emission_dim == 3:
            self.topology_names = ["ABBA", "BABA", "AABB"]
        else:
            self.topology_names = [f"Coord {i+1}" for i in range(self.emission_dim)]

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
            if self.phlag.args.model_design == "beta":
                ranges[d] = np.linspace(max(1e-5, ymin - ypad), min(1.0 - 1e-5, ymax + ypad), 300)
            else:
                ranges[d] = np.linspace(ymin - ypad, ymax + ypad, 300)

        # Plot Row 0: Before EM (Initial theoretical setup)
        self._plot_row(axes[0], self.initial_params, ranges, title_prefix="Before EM", plot_empirical=False)
        
        # Plot Row 1: After EM (Fitted theoretical setup and assigned empirical data)
        self._plot_row(axes[1], self.phlag.params, ranges, title_prefix="After EM", plot_empirical=True)
        
        plt.tight_layout()
        self.save_plot()

    def _plot_row(self, row_axes, params, ranges, title_prefix, plot_empirical=False):
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
            
            # 2. Plot PDF curves and Vertical Guideline Markers (Mean and +/- 1 Std)
            for state in [0, 1]:
                color_config = self.colors[state]
                
                if self.phlag.args.model_design == "beta":
                    alpha = float(params.emissions.concentration1[state, d])
                    beta = float(params.emissions.concentration0[state, d])
                    pdf_vals = stats.beta.pdf(x_vals, alpha, beta)
                    mu = alpha / (alpha + beta)
                    sigma = np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
                elif self.phlag.args.model_design == "gmm":
                    w = np.array(params.emissions.mixture_weights[state, d])
                    m_means = np.array(params.emissions.means[state, d])
                    m_stds = np.array(params.emissions.stds[state, d])
                    pdf_vals = np.zeros_like(x_vals)
                    for m in range(len(w)):
                        pdf_vals += w[m] * stats.norm.pdf(x_vals, m_means[m], m_stds[m])
                    mu = float(np.sum(w * m_means))
                    var = float(np.sum(w * (m_stds ** 2 + m_means ** 2)) - mu ** 2)
                    sigma = np.sqrt(max(1e-6, var))
                else:
                    mu = float(params.emissions.means[state, d])
                    sigma = np.sqrt(np.clip(float(params.emissions.covariances[state, d, d]), a_min=1e-6, a_max=None))
                    pdf_vals = stats.norm.pdf(x_vals, mu, sigma)
                
                # Plot theoretical curve
                ax.plot(
                    x_vals, pdf_vals, color=color_config['line'], 
                    linewidth=2.2, label=f"{color_config['label']} PDF"
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
        output_dir = getattr(self.phlag, "output_file", None)
        if output_dir:
            output_dir = output_dir.parent
        else:
            if self.phlag.args.output_file:
                output_dir = pathlib.Path(self.phlag.args.output_file).parent
            else:
                output_dir = pathlib.Path.cwd() / "test" / self.input_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
            
        plot_file = output_dir / "em.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved EM distributions plot to: {plot_file}")


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
        "-e",
        dest="emission_parameterization",
        type=str.lower,
        default="attraction",
        choices=["free", "attraction", "anchor"],
        help="Parameterization of the emission probabilities of the default state (default: attraction)",
    )
    hmm_group.add_argument(
        "--emission-lambda",
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
        choices=["gaussian", "beta", "gmm"],
        help="Type of HMM emissions (gaussian, beta, or gmm. Default: gaussian)",
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

    args = parser.parse_args()

    from .utils import get_data_dir, get_repo_root, resolve_input_file, get_most_recent_file
    repo_root = get_repo_root()
    data_dir = get_data_dir()

    dist_type = getattr(args, "model_design", "gaussian")

    def resolve_model_scores(target_name=None):
        search_bases = [
            data_dir / "phlag" / dist_type,
        ]
        candidates = []
        for b in search_bases:
            if not b.exists():
                continue
            if target_name:
                # Look for target_name subdirectory
                target_dirs = list(b.glob(f"**/{target_name}/**/caster")) + list(b.glob(f"**/{target_name}"))
                for td in target_dirs:
                    if td.is_dir():
                        for sfile in td.rglob("scores.tsv"):
                            if sfile.parent.name == "caster":
                                candidates.append(sfile)
                        for sfile in td.rglob("*.tsv"):
                            if sfile.parent.name == "caster" and not any(sfile.name.startswith(p) for p in ["report_", "em_", "states_"]):
                                candidates.append(sfile)
            else:
                for sfile in b.rglob("scores.tsv"):
                    if sfile.parent.name == "caster":
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
        # Check model params directory under dist_type first (e.g. gaussian/<name>/.../caster/scores.tsv)
        model_scores = resolve_model_scores(target_name=args.caster_scores.name)
        if model_scores and model_scores.exists():
            args.caster_scores = model_scores
        else:
            resolved = resolve_input_file(args.caster_scores, default_subdirs=["scores", "msa", "store/scores", "store/phlag"], default_exts=[".tsv", ".txt"])
            if resolved.exists() and resolved.is_file():
                args.caster_scores = resolved
            else:
                sys.exit(f"Error: Score file not found for '{args.caster_scores.name}' under model output directories or relative paths.")

    # Check if --plot is supplied
    plot_supplied = any(arg == "--plot" or arg.startswith("--plot=") for arg in sys.argv)
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


def main():
    import copy
    organize_existing_test_files()
    args = parse_arguments()
    phlag = Phlag(args)
    
    # Ingest baseline setup properties before fitting
    initial_params = copy.deepcopy(phlag.params)
    
    phlag.run()
    phlag.save_output()
    
    if args.plot and "em" in args.plot:
        PhlagPlotter(phlag, initial_params)


if __name__ == "__main__":
    main()