import sys
import pathlib
import os
import re
import argparse
import subprocess
import shutil
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def format_val(val):
    """
    Abbreviate numbers using 'k' for thousands and 'm' for millions.
    0 remains '0'.
    """
    if val == 0:
        return "0"
    if val % 1000000 == 0:
        return f"{val // 1000000}m"
    elif val % 1000 == 0:
        return f"{val // 1000}k"
    return str(val)

def int_or_abbrev(val_str):
    val_str = str(val_str).strip().lower()
    if val_str.endswith('k'):
        return int(float(val_str[:-1]) * 1000)
    elif val_str.endswith('m'):
        return int(float(val_str[:-1]) * 1000000)
    return int(val_str)

def get_fasta_length(fasta_path):
    length = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                break
        for line in f:
            if line.startswith(">"):
                break
            length += len(line.strip())
    return length


class CasterPlotter:
    def __init__(self, scores_file, distribution='gaussian', data_dir=None, topologies=None, plot_scores=True, locus_pattern=None):
        self.scores_file = scores_file
        self.distribution = distribution
        self.data_dir = data_dir if data_dir is not None else str(pathlib.Path(scores_file).parent)
        self.topologies = topologies
        self.locus_pattern = locus_pattern

        os.makedirs(self.data_dir, exist_ok=True)

        base_name = pathlib.Path(scores_file).name
        name_part, _ = os.path.splitext(base_name)
        self.is_normalized_file = name_part.endswith('_n')

        from .utils import get_locus_description
        self.gene_name = get_locus_description(scores_file)

        self.load_data()

        if self.df is not None:
            # High-contrast, vibrant, and highly distinguishable color palette for the 3 topologies
            self.topo_colors = {
                'ABBA': '#1F77B4',    # Bold Royal Blue
                'BABA': '#D62728',    # Vivid Crimson Red
                'AABB': '#2CA02C',    # Vibrant Forest Green
            }

            if plot_scores:
                self.plot_topology_scatter()

    def load_data(self):
        """Parses the tab-separated value file into a Pandas DataFrame."""
        target_path = self.scores_file
        if not os.path.exists(target_path) and not os.path.isabs(target_path):
            target_path = os.path.join(self.data_dir, self.scores_file)

        try:
            self.df = pd.read_csv(target_path, sep='\t')
            print(f"Loaded {len(self.df)} windows for locus '{self.gene_name}' from: {target_path}")
            print("Detected columns:", self.df.columns.tolist())
            sns.set_theme(style="whitegrid")
        except Exception as e:
            print(f"Error reading dataset file {target_path}: {e}")
            self.df = None

    def _ground_truth_pattern(self):
        """
        The ground-truth locus pattern (e.g. '37-62' or 'n1a1n5...'), preferring
        the explicit locus_pattern the caller already knows (caster.py's own
        run passes it, since flat standalone output no longer encodes it in
        scores_file's path) and falling back to regex-parsing scores_file's
        path/filename otherwise (a bare scores.tsv invocation, or the
        canonical --bench tree, which still encodes it there).
        """
        if self.locus_pattern:
            return self.locus_pattern
        full_path_str = str(self.scores_file)
        m = re.search(r'((?:[an]\d+(?:-[an]?\d+)?(?:_)?)+|\d+-\d+(?:[;_,]\d+-\d+)*)', full_path_str)
        return m.group(1) if m else None

    def calculate_summary_statistics(self, series):
        """Calculates summary statistics, returns and sets self.params dict for scipy.stats."""
        if self.distribution:
            dist_name = self.distribution.lower()
            if dist_name in ['gaussian', 'normal']:
                dist_name = 'norm'

            import scipy.stats as stats_module
            dist_class = getattr(stats_module, dist_name)
            fit_vals = dist_class.fit(series)

            param_names = []
            if dist_class.shapes:
                param_names.extend([s.strip() for s in dist_class.shapes.split(',')])
            param_names.extend(['loc', 'scale'])
            self.params = dict(zip(param_names, fit_vals))
        else:
            self.params = {
                'loc': series.mean(),
                'scale': series.std()
            }
        return self.params

    def plot_topology_scatter(self):
        # caster-pair's chunk_scores.tsv carries both raw per-window quartet-support
        # sums (c*ABBA/c*BABA/c*AABB, unbounded, scale with window size) and their
        # normalized proportions (q1/q2/q3, in [0,1], positionally ABBA/BABA/AABB --
        # see caster-pair.cpp's scoreChunksForBranch). Prefer q1/q2/q3 when present
        # so the plotted "Score" is comparable in scale to dstar's own already-
        # normalized c*ABBA/c*BABA/c*AABB columns.
        pair_cols = ['q1', 'q2', 'q3']
        if all(c in self.df.columns for c in pair_cols):
            avg_cols = pair_cols
            rename_map = {'q1': 'ABBA', 'q2': 'BABA', 'q3': 'AABB'}
        else:
            avg_cols = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
            rename_map = {}
            for col in avg_cols:
                match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
                rename_map[col] = match.group(1).upper() if match else col

        if self.topologies is not None:
            filtered_cols = []
            for col in avg_cols:
                mapped_name = rename_map.get(col, col)
                for t in self.topologies:
                    if t.lower() in mapped_name.lower():
                        filtered_cols.append(col)
                        break
            avg_cols = filtered_cols

        if not avg_cols:
            print("No matching topology columns found to scatter plot.")
            return

        plt.figure(figsize=(12, 6))

        renamed_df = self.df.rename(columns=rename_map)
        clean_cols = [rename_map.get(c, c) for c in avg_cols]

        melted_df = renamed_df.melt(id_vars=['pos'], value_vars=clean_cols,
                                 var_name='Topology', value_name='Score')

        sns.scatterplot(data=melted_df, x='pos', y='Score', hue='Topology', palette=self.topo_colors, alpha=0.6, s=12)

        # Draw vertical split lines and shade alt regions if a ground truth pattern is known
        pattern_str = self._ground_truth_pattern()
        if pattern_str:
            from .utils import parse_pattern_string
            total_span = self.df['pos'].max() if ('pos' in self.df.columns and len(self.df) > 0) else None
            blocks, anomaly_intervals, _ = parse_pattern_string(pattern_str, block_size_bp=500000, total_span=total_span)

            if anomaly_intervals and not any(b[0] == 'n' for b in blocks):
                # Pure interval format (e.g. 45-55)
                alt_shaded = False
                for start_pos, end_pos in anomaly_intervals:
                    mid_pos = (start_pos + end_pos) / 2.0
                    lbl = 'Alt' if not alt_shaded else None
                    plt.axvspan(start_pos, end_pos, color='#E05638', alpha=0.12, label=lbl)
                    alt_shaded = True
                    plt.axvline(x=start_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)
                    plt.axvline(x=end_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)
            elif blocks:
                curr_pos_bp = 0
                alt_shaded = False
                for idx, (b_type, b_id, length_bp) in enumerate(blocks):
                    start_pos = curr_pos_bp
                    end_pos = curr_pos_bp + length_bp
                    mid_pos = (start_pos + end_pos) / 2.0
                    label_text = "null" if b_type == 'n' else "alt"

                    if b_type == 'a':
                        lbl = 'Alt' if not alt_shaded else None
                        plt.axvspan(start_pos, end_pos, color='#E05638', alpha=0.12, label=lbl)
                        alt_shaded = True

                    if start_pos > 0:
                        plt.axvline(x=start_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)

                    plt.text(mid_pos, 0.02, label_text, transform=plt.gca().get_xaxis_transform(),
                             ha='center', va='bottom', fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

                    curr_pos_bp = end_pos
                plt.axvline(x=curr_pos_bp, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)

        # Format names for cleaner legend and title
        plt.title(f'Genomic Topology Profile: {self.gene_name}', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('Genomic Position (pos)', fontsize=11, labelpad=8)
        plt.ylabel('Topology Score Value', fontsize=11, labelpad=8)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()

        output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_scatter = os.path.join(output_dir, 'scatter.png')
        plt.savefig(save_path_scatter, dpi=300)
        print(f"Saved empirical topology scatter plot to: {save_path_scatter}")
        plt.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Caster: Load scores and generate topology distribution and scatter plots."
    )

    parser.add_argument(
        "fasta_file",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help="Input FASTA file path"
    )
    
    # Recent flag & Left/Right indices
    parser.add_argument(
        "-r",
        "--recent",
        dest="recent",
        action="store_true",
        help="Use the most recently created FASTA file in store/msa/concat"
    )
    parser.add_argument(
        "-l",
        dest="left",
        type=int_or_abbrev,
        default=0,
        help="Left index of range (0-indexed, inclusive)"
    )
    parser.add_argument(
        "-R",
        "--right",
        dest="right",
        type=int_or_abbrev,
        required=False,
        default=None,
        help="Right index of range (0-indexed, exclusive)"
    )
    
    # Dstar parameters
    parser.add_argument(
        "-w",
        dest="window_size",
        type=int_or_abbrev,
        default=50000,
        help="Window size (default: 50000 / 50k)"
    )
    parser.add_argument(
        "-n",
        dest="normalize",
        action="store_true",
        help="Apply min-max normalization to D* scores"
    )
    parser.add_argument(
        "-s",
        dest="step_size",
        type=int_or_abbrev,
        default=1000,
        help="Step size (default: 1000 / 1k)"
    )

    parser.add_argument(
        "--shift-caster",
        dest="shift_caster",
        action="store_true",
        default=False,
        help="Shift each window's reported pos right by window_size/2 so it "
             "marks the window's center instead of its left edge (default: omitted)"
    )
    parser.add_argument(
        "-m",
        dest="mapping",
        type=pathlib.Path,
        default=None,
        help="Optional population mapping file path"
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=["scores"],
        default=["scores"],
        help="List of plots to generate (choices: scores. Default: scores)",
    )
    parser.add_argument(
        "-t",
        "--topologies",
        dest="topologies",
        nargs="+",
        default=None,
        help="List of topologies to plot (default: all)"
    )
    parser.add_argument(
        "-d",
        "--dist-type",
        dest="dist_type",
        default="gaussian",
        choices=["gaussian", "gmm"],
        help="Distribution type used for CasterPlotter's statistical fits (default: "
             "gaussian). No longer affects scores.tsv's output location -- that's "
             "shared across dist_types, see --bench."
    )
    parser.add_argument(
        "-o", dest="output_file", type=pathlib.Path, default=None,
        help="Path to save scores.tsv (its directory is also where scatter.png is "
             "saved, if plotting). Ignored when --bench is set (canonical tree "
             "always wins there)."
    )
    parser.add_argument(
        "--output-base",
        dest="output_base",
        default=None,
        help="Accepted for CLI compatibility with phlag/phlagster (which pass "
             "the same flag to both stages), but has no effect here: "
             "scores.tsv always lives in one canonical, --output-base/dist_type-"
             "independent location keyed only by w<W>_s<S>, shared across every "
             "phlag-side --output-base/--base/dist_type variant instead of being "
             "recomputed/duplicated per variant."
    )
    parser.add_argument(
        "--bench",
        dest="bench",
        action="store_true",
        default=False,
        help="Set by benchmark's run_all() for its own subprocess invocations -- "
             "not meant to be passed by hand. Accepted for CLI compatibility with "
             "phlag/phlagster; has no effect on scores.tsv's location when set "
             "(stays in the canonical shared tree, same as always: "
             "store/caster/w<W>_s<S>/...). When NOT set (standalone use, the "
             "default), scores go to <repo_root>/out/w<W>_s<S>/<node_name>/"
             "<node_name>.tsv instead of the shared canonical tree."
    )
    parser.add_argument(
        "--pair",
        dest="pair",
        action="store_true",
        default=False,
        help="Run ./caster/bin/caster-pair (auto-(re)compiling from "
             "caster/caster-pair.cpp if missing/stale) instead of dstar -- scores one "
             "fixed quartet branch's topology per genomic chunk, instead of D*/ABBA-"
             "BABA-AABB windows. The branch is the -m/--mapping population mapping "
             "file (auto-detected the same way as for dstar), which must assign every "
             "taxon to exactly one of 4 groups. See --chunk-scores."
    )
    parser.add_argument(
        "--chunk-scores",
        dest="chunk_scores",
        type=pathlib.Path,
        default=None,
        help="Output path for --pair's per-chunk quartet scores TSV (default: "
             "wherever --pair's scores file would normally go)."
    )
    parser.add_argument(
        "--chunk",
        dest="chunk_size",
        type=int_or_abbrev,
        default=None,
        help="Window size (in sites) for --pair's per-window quartet scores, rolled up "
             "from caster-pair's own fine-grained -s-sized chunks the same way dstar's "
             "window/step aggregation works below (default: -w's window size). See -s "
             "for the step/stride between windows -- when -s < --chunk, windows overlap."
    )
    return parser


def parse_arguments(argv=None):
    parser = build_parser()
    from .utils import apply_cli_config
    return apply_cli_config(parser, argv, "caster")


def run_caster_pair(args, repo_root, data_dir, final_output_path, locus_pattern):
    plot_data_dir = final_output_path.parent
    binary_name = "caster-pair.exe" if sys.platform == "win32" else "caster-pair"
    binary_candidates = [
        data_dir / "bin" / binary_name,
        repo_root / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "bin" / binary_name,
    ]
    which_path = shutil.which(binary_name)
    if which_path:
        binary_candidates.append(pathlib.Path(which_path))

    binary_path = None
    for candidate in binary_candidates:
        if candidate.exists():
            binary_path = candidate
            break

    if not binary_path:
        binary_path = repo_root / "caster" / "bin" / binary_name

    caster_pair_cpp = repo_root / "caster" / "caster-pair.cpp"
    if caster_pair_cpp.exists():
        if not binary_path.exists() or os.path.getmtime(caster_pair_cpp) > os.path.getmtime(binary_path):
            target_bin = repo_root / "caster" / "bin" / binary_name
            os.makedirs(target_bin.parent, exist_ok=True)
            print(f"Compiling 'caster-pair' binary from {caster_pair_cpp}...")
            includes_dir = repo_root / "caster" / "includes"
            compile_cmd = ["g++", "-std=gnu++17", "-O2", "-I", str(includes_dir), str(caster_pair_cpp), "-o", str(target_bin)]
            try:
                subprocess.run(compile_cmd, check=True)
                binary_path = target_bin
                print(f"Successfully compiled 'caster-pair' binary at {binary_path}")
            except Exception as e:
                print(f"Warning: Could not auto-compile 'caster-pair': {e}")

    if sys.platform != "win32" and binary_path.exists() and not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, os.stat(binary_path).st_mode | 0o755)
        except Exception as e:
            print(f"Warning: Failed to set executable permission on '{binary_path}': {e}")

    if not binary_path.exists():
        sys.exit(f"Error: 'caster-pair' binary not found and could not be compiled (looked for source at {caster_pair_cpp}).")

    if not args.mapping.exists():
        sys.exit(f"Error: Mapping file not found at '{args.mapping}'")

    chunk_scores_path = args.chunk_scores if args.chunk_scores else final_output_path
    chunk_scores_path.parent.mkdir(parents=True, exist_ok=True)
    window_size = args.chunk_size if args.chunk_size is not None else args.window_size
    pair_step = min(args.step_size, window_size)

    print(f"Running caster-pair on '{args.fasta_file}' with branch mapping '{args.mapping}', chunk(window)={window_size}, step={pair_step}...")
    cmd = [
        str(binary_path),
        "--branch-mapping", str(args.mapping.resolve()),
        "--chunk-scores", str(chunk_scores_path.resolve()),
        "--chunk", str(pair_step),
        str(args.fasta_file.resolve()),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error running 'caster-pair' binary:\nCommand: {e.cmd}\nExit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")

    # caster-pair itself only ever partitions a locus into non-overlapping
    # --chunk-sized blocks -- it has no native step/stride concept. To get an
    # overlapping window+step model (window_size != pair_step), we ran it above
    # at the fine pair_step granularity, and now roll consecutive per-locus rows
    # into window_size-wide sliding windows here -- the same two-stage
    # fine-grained-rows -> rolling-window trick used for dstar above.
    K = max(1, window_size // pair_step)
    if K > 1:
        raw_df = pd.read_csv(chunk_scores_path, sep="\t")
        agg_rows = []
        for source_file, locus_df in raw_df.groupby("file", sort=False):
            locus_df = locus_df.sort_values("pos").reset_index(drop=True)
            for i in range(len(locus_df) - K + 1):
                window = locus_df.iloc[i:i + K]
                s0 = window["c*ABBA"].sum()
                s1 = window["c*BABA"].sum()
                s2 = window["c*AABB"].sum()
                tot = s0 + s1 + s2
                pos_val = int(locus_df.iloc[i]["pos"])
                if args.shift_caster:
                    pos_val += window_size // 2
                agg_rows.append({
                    "file": source_file,
                    "pos": pos_val,
                    "c*ABBA": s0,
                    "c*BABA": s1,
                    "c*AABB": s2,
                    "q1": s0 / tot if tot > 0 else 1.0 / 3,
                    "q2": s1 / tot if tot > 0 else 1.0 / 3,
                    "q3": s2 / tot if tot > 0 else 1.0 / 3,
                })
        pd.DataFrame(agg_rows).to_csv(chunk_scores_path, sep="\t", index=False)

    print(f"Success: chunk scores written to: {chunk_scores_path}")

    if args.plot and "scores" in args.plot:
        # chunk_scores.tsv's columns (pos, c*ABBA, c*BABA, c*AABB) are written
        # by caster-pair.cpp to mirror dstar's scores.tsv exactly, so the same
        # CasterPlotter -- same palette, same ground-truth shading -- renders
        # it directly instead of a separate plotting path.
        CasterPlotter(
            scores_file=str(chunk_scores_path.resolve()),
            distribution=args.dist_type,
            data_dir=str(plot_data_dir.resolve()),
            topologies=args.topologies,
            plot_scores=True,
            locus_pattern=locus_pattern,
        )

    return chunk_scores_path


def main(argv=None):
    args = parse_arguments(argv)
    
    # Inject defaults for CLI flags if not provided
    if args.step_size is None:
        args.step_size = args.window_size

    if not args.bench:
        flags_str = " ".join(f"{k}={v}" for k, v in vars(args).items())
        print(f"[caster] Effective flags: {flags_str}")

    from .utils import get_data_dir, get_repo_root, get_most_recent_file, clean_locus_name
    repo_root = get_repo_root()
    data_dir = get_data_dir()
    
    # Resolve FASTA file fallback if not found or recent flag requested
    if args.recent or args.fasta_file == pathlib.Path("-r") or args.fasta_file is None:
        recent_fasta = get_most_recent_file(
            default_subdirs=["store/msa/concat", "msa/concat", "concat", "store/msa", "msa"],
            default_exts=[".fa", ".fasta", ".fa.gz"],
            target_dir_name="concat"
        )
        if recent_fasta is None or not recent_fasta.exists():
            sys.exit("Error: No input file found in store/msa/concat or candidate MSA directories.")
        args.fasta_file = recent_fasta

    # Ground-truth locus pattern (e.g. '37-62') for CasterPlotter's scatter.png shading.
    window_str = format_val(args.window_size)
    step_str = format_val(args.step_size)
    norm_suffix = "_n" if args.normalize else ""
    clean_stem = clean_locus_name(args.fasta_file.stem)
    left_str = format_val(args.left)
    right_str = format_val(args.right if args.right is not None else 0)

    from .utils import parse_filename_to_dir_structure, get_simulation_categories, get_short_sim_name
    parsed = parse_filename_to_dir_structure(clean_stem)
    locus_pattern = parsed["pattern"] if parsed else clean_stem

    parts = args.fasta_file.parts
    is_sim = "simulations" in parts
    cats = None
    short_sim = None
    if is_sim:
        sim_dir = args.fasta_file.parent
        if sim_dir.name in ["concat"] or sim_dir.name.startswith("concat_"):
            sim_dir = sim_dir.parent
        cats = get_simulation_categories(args.fasta_file)
        short_sim = get_short_sim_name(sim_dir.name)

    if args.bench:
        # --bench (set only by benchmark's own subprocess invocations) keeps
        # scores.tsv in the shared canonical tree, keyed only by window/step
        # -- caster's windowed dstar statistics don't depend on dist_type or
        # --output-base/--base at all (those only select a phlag-side
        # model/variant tree), so every dist_type/--base variant reads and
        # writes the same cached scores.tsv here instead of each getting its
        # own copy recomputed from scratch. Lives in its own store/caster/
        # tree, not nested under store/phlag/, since it isn't a phlag output.
        caster_root = data_dir / "caster" / f"w{window_str}_s{step_str}"
        if parsed:
            rel_dir = parsed["relative_dir_no_window"]
            final_output_path = caster_root / rel_dir / "scores.tsv"
        elif is_sim:
            pattern_stem = clean_stem
            if cats:
                final_output_path = caster_root / cats[0] / cats[1] / short_sim / pattern_stem / "scores.tsv"
            else:
                final_output_path = caster_root / short_sim / pattern_stem / "scores.tsv"
        else:
            pattern_stem = clean_stem
            final_output_name = f"{clean_stem}_{left_str}_{right_str}_w{window_str}_s{step_str}{norm_suffix}.tsv"
            final_output_path = caster_root / pattern_stem / final_output_name
    else:
        # Standalone use (the default): no shared/canonical tree, no
        # dist_type/category/pattern nesting -- everything for a node lands
        # under <repo_root>/out/w<W>_s<S>/<node_name>/<node_name>.tsv,
        # alongside phlag's report.tsv for the same node (see phlag.py).
        # Node name is the short simulation name for sim inputs, the alt
        # name for parsed null/alt filenames, or the cleaned stem otherwise.
        if parsed:
            node_name = get_short_sim_name(parsed["alt"])
        elif is_sim:
            node_name = short_sim
        else:
            node_name = clean_stem
        if args.pair:
            # --pair's chunk-rollup granularity is an independent axis from
            # dstar's window/step (it may not even be run for the same node),
            # so its output gets its own out/c<chunk>_s<step>/<node_name>/
            # prefix instead of colliding with dstar's out/w<W>_s<S>/<node_name>/.
            pair_chunk = args.chunk_size if args.chunk_size is not None else args.window_size
            pair_chunk_str = format_val(pair_chunk)
            pair_step_str = format_val(args.step_size)
            final_output_path = repo_root / "out" / f"c{pair_chunk_str}_s{pair_step_str}" / node_name / f"{node_name}.tsv"
        else:
            final_output_path = repo_root / "out" / f"w{window_str}_s{step_str}" / node_name / f"{node_name}.tsv"

    # -o override (standalone use only -- ignored under --bench, where the
    # canonical tree always wins): redirects where scores.tsv itself gets
    # written (and, via its parent, where CasterPlotter's scatter.png lands).
    if args.output_file and not args.bench:
        final_output_path = args.output_file
    plot_data_dir = final_output_path.parent

    if not args.fasta_file.exists():
        sys.exit(f"Error: FASTA file not found at '{args.fasta_file}'")
    if args.left < 0:
        sys.exit(f"Error: Left index must be >= 0, got {args.left}")
    if args.right is None:
        try:
            args.right = get_fasta_length(args.fasta_file)
        except Exception as e:
            sys.exit(f"Error reading FASTA file to compute right endpoint: {e}")
    if args.right <= args.left:
        sys.exit(f"Error: Right index must be greater than left index, got left={args.left}, right={args.right}")

    # Locate dstar binary, auto-(re)compiling from source if missing or stale
    binary_name = "dstar.exe" if sys.platform == "win32" else "dstar"
    binary_candidates = [
        data_dir / "bin" / binary_name,
        repo_root / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "caster" / "bin" / binary_name,
        pathlib.Path.cwd() / "bin" / binary_name,
    ]
    which_path = shutil.which(binary_name)
    if which_path:
        binary_candidates.append(pathlib.Path(which_path))

    binary_path = None
    for candidate in binary_candidates:
        if candidate.exists():
            binary_path = candidate
            break

    if not binary_path:
        binary_path = repo_root / "caster" / "bin" / binary_name

    dstar_cpp = repo_root / "caster" / "dstar.cpp"
    if dstar_cpp.exists():
        if not binary_path.exists() or os.path.getmtime(dstar_cpp) > os.path.getmtime(binary_path):
            target_bin = repo_root / "caster" / "bin" / binary_name
            os.makedirs(target_bin.parent, exist_ok=True)
            print(f"Compiling 'dstar' binary from {dstar_cpp}...")
            compile_cmd = ["g++", "-std=gnu++17", "-O2", str(dstar_cpp), "-o", str(target_bin)]
            try:
                subprocess.run(compile_cmd, check=True)
                binary_path = target_bin
                print(f"Successfully compiled 'dstar' binary at {binary_path}")
            except Exception as e:
                print(f"Warning: Could not auto-compile 'dstar': {e}")

    if sys.platform != "win32" and binary_path.exists() and not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, os.stat(binary_path).st_mode | 0o755)
        except Exception as e:
            print(f"Warning: Failed to set executable permission on '{binary_path}': {e}")

    # Auto-detect a population mapping file if none was given explicitly.
    # Simulation mapping files live alongside 'concat/' as neoaves_{node}_mapping.tsv,
    # where {node} may carry a ':clade' disambiguator suffix (e.g. 'Strigiformes:3').
    if args.mapping is None:
        sim_dir = args.fasta_file.parent
        if sim_dir.name == "concat" or sim_dir.name.startswith("concat_"):
            sim_dir = sim_dir.parent
        mapping_candidates = sorted(sim_dir.glob("neoaves_*_mapping.tsv")) or sorted(sim_dir.glob("*_mapping.tsv"))

        if len(mapping_candidates) == 1:
            args.mapping = mapping_candidates[0]
            print(f"Auto-detected mapping file: {args.mapping}")
        elif len(mapping_candidates) > 1:
            sys.exit(f"Error: Multiple candidate mapping files found in '{sim_dir}': {[str(p) for p in mapping_candidates]}. Please specify one explicitly with --mapping.")
        else:
            sys.exit(f"Error: No mapping file found in '{sim_dir}'. Please generate one or specify an existing one explicitly with --mapping.")

    if args.pair:
        return run_caster_pair(args, repo_root, data_dir, final_output_path, locus_pattern)

    # Run dstar binary on original fasta file directly, then aggregate its
    # fine-grained (step_size-spaced) rows into rolling window_size averages
    print(f"Running D* calculation on original file '{args.fasta_file}' with window={args.window_size}, step={args.step_size}...")
    cmd = [str(binary_path), str(args.fasta_file.resolve())]
    if args.mapping:
        if not args.mapping.exists():
            sys.exit(f"Error: Mapping file not found at '{args.mapping}'")
        cmd.append(str(args.mapping.resolve()))
    else:
        cmd.append("-")
    cmd.append(str(args.step_size))

    temp_dir = tempfile.mkdtemp()
    try:
        try:
            result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Error running 'dstar' binary:\nCommand: {e.cmd}\nExit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")

        lines = result.stdout.splitlines()
        if not lines:
            sys.exit(f"Error: D* output was empty. Stderr: {result.stderr}")

        header_idx = 0
        for idx, line in enumerate(lines):
            if "pos" in line:
                header_idx = idx
                break

        raw_rows = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            parts = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(parts) >= 7:
                raw_rows.append(parts)

        if not raw_rows:
            mapping_name = args.mapping.name if args.mapping else "-"
            sys.exit(f"Error: No window scores calculated for '{args.fasta_file.name}' using mapping '{mapping_name}'. Please check that sequence headers in the FASTA match species in the mapping file.")

        # Perform rolling window average with O(1) sliding window
        K = max(1, args.window_size // args.step_size)

        parsed_rows = [
            (
                row[0],             # file
                int(row[1]),        # pos
                float(row[2]),      # abba
                float(row[3]),      # baba
                float(row[4]),      # aabb
                float(row[6])       # qcnt
            )
            for row in raw_rows
        ]

        results = []
        if len(parsed_rows) >= K:
            run_abba = sum(r[2] for r in parsed_rows[:K])
            run_baba = sum(r[3] for r in parsed_rows[:K])
            run_aabb = sum(r[4] for r in parsed_rows[:K])
            run_qcnt = sum(r[5] for r in parsed_rows[:K])

            for i in range(len(parsed_rows) - K + 1):
                if i > 0:
                    outgoing = parsed_rows[i - 1]
                    incoming = parsed_rows[i + K - 1]
                    run_abba += incoming[2] - outgoing[2]
                    run_baba += incoming[3] - outgoing[3]
                    run_aabb += incoming[4] - outgoing[4]
                    run_qcnt += incoming[5] - outgoing[5]

                pos_val = parsed_rows[i][1]  # Position of the start of the window
                if args.shift_caster:
                    pos_val += args.window_size // 2

                avg_abba = run_abba / K
                avg_baba = run_baba / K
                avg_aabb = run_aabb / K
                avg_qcnt = run_qcnt / K

                # Recalculate D* for the combined window (denom ratio is invariant to K)
                denom = run_abba + run_baba + run_aabb
                dstar_val = (run_abba - run_baba) / denom if denom != 0 else 0.0

                if args.left <= pos_val < args.right:
                    file_val = parsed_rows[i][0]
                    results.append({
                        'file': file_val,
                        'pos': pos_val,
                        'abba': avg_abba,
                        'baba': avg_baba,
                        'aabb': avg_aabb,
                        'dstar': dstar_val,
                        'qcnt': avg_qcnt
                    })

        # If normalization is requested, apply min-max scaling to [0, 1] for each score column
        if args.normalize and len(results) > 0:
            for key in ['abba', 'baba', 'aabb', 'dstar']:
                vals = [r[key] for r in results]
                min_val = min(vals)
                max_val = max(vals)
                diff = max_val - min_val
                if diff == 0:
                    for r in results:
                        r[key] = 0.0
                else:
                    for r in results:
                        r[key] = (r[key] - min_val) / diff

        output_lines = ["file\tpos\tc*ABBA\tc*BABA\tc*AABB\tD*\tQuartetCnt\n"]
        for r in results:
            output_lines.append(f"{r['file']}\t{r['pos']}\t{r['abba']:.6g}\t{r['baba']:.6g}\t{r['aabb']:.6g}\t{r['dstar']:.6g}\t{r['qcnt']:.0f}\n")

        # final_output_path may be the shared, base-independent caster/
        # cache -- concurrent benchmark runs across different --base
        # variants can legitimately be computing the same scores.tsv at
        # once, so the write itself must be atomic (write-then-rename)
        # rather than an in-place open("w"): any reader checking
        # final_output_path.exists() must only ever see either nothing
        # or a complete file, never a partial one.
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(final_output_path.parent), prefix=".scores_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.writelines(output_lines)
            os.replace(tmp_path, final_output_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        print(f"Success: TSV output file generated at: {final_output_path}")
    finally:
        shutil.rmtree(temp_dir)

    print(f"Using scores file: {final_output_path}")

    if args.plot:
        plot_scores = "scores" in args.plot

        if plot_scores:
            CasterPlotter(
                scores_file=str(final_output_path.resolve()),
                distribution=args.dist_type,
                data_dir=str(plot_data_dir.resolve()),
                topologies=args.topologies,
                plot_scores=plot_scores,
                locus_pattern=locus_pattern,
            )

    return final_output_path

if __name__ == "__main__":
    main()
