import sys
import pathlib
import os
import re
import argparse
import subprocess
import shutil
import tempfile

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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Caster: Load scores and generate topology distribution and scatter plots."
    )
    
    # Input FASTA or scores.tsv file
    parser.add_argument(
        "fasta_file",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help="Input FASTA or scores.tsv file path"
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
        "-m",
        dest="mapping",
        type=pathlib.Path,
        default=None,
        help="Optional population mapping file path"
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=["scores", "dist", "hist"],
        default=["scores", "dist"],
        help="List of plots to generate (choices: scores, dist, hist. Default: scores, dist)",
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
        "--plot-dstar",
        dest="plot_dstar",
        action="store_true",
        help="Plot D* distribution (default: False)"
    )
    parser.add_argument(
        "-d",
        "--dist-type",
        dest="dist_type",
        default="gaussian",
        choices=["gaussian", "gmm"],
        help="Distribution type for output directory structure (default: gaussian)"
    )
    parser.add_argument(
        "--tree",
        dest="tree_file",
        type=pathlib.Path,
        default=None,
        help="Optional species tree file (default: store/63K.tre)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Inject defaults for CLI flags if not provided
    if args.step_size is None:
        args.step_size = args.window_size
        
    from .utils import get_data_dir, get_repo_root, resolve_input_file, get_most_recent_file, clean_locus_name
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

    input_str = str(args.fasta_file)
    if input_str.endswith('.tsv') or args.fasta_file.name == 'scores.tsv':
        final_output_path = args.fasta_file
    else:
        window_str = format_val(args.window_size)
        step_str = format_val(args.step_size)
        norm_suffix = "_n" if args.normalize else ""
        clean_stem = clean_locus_name(args.fasta_file.stem)
        left_str = format_val(args.left)
        right_str = format_val(args.right if args.right is not None else 0)

        # Resolve existing scores.tsv
        from .utils import parse_filename_to_dir_structure, get_phlag_output_base
        phlag_base = get_phlag_output_base(data_dir)
        parsed = parse_filename_to_dir_structure(clean_stem)
        if parsed:
            rel_dir = parsed["relative_dir"]
            final_output_path = phlag_base / args.dist_type / rel_dir / "caster" / "scores.tsv"
        else:
            parts = args.fasta_file.parts
            is_sim = False
            if "simulations" in parts:
                sim_dir = args.fasta_file.parent
                if sim_dir.name in ["concat"] or sim_dir.name.startswith("concat_"):
                    sim_dir = sim_dir.parent
                sim_name = sim_dir.name
                
                from .utils import get_simulation_categories, get_short_sim_name
                cats = get_simulation_categories(args.fasta_file)
                short_sim = get_short_sim_name(sim_name)
                pattern_stem = clean_stem
                if cats:
                    final_output_path = phlag_base / args.dist_type / f"w{window_str}_s{step_str}" / cats[0] / cats[1] / short_sim / pattern_stem / "caster" / "scores.tsv"
                else:
                    final_output_path = phlag_base / args.dist_type / f"w{window_str}_s{step_str}" / short_sim / pattern_stem / "caster" / "scores.tsv"
                is_sim = True
            
            if not is_sim:
                pattern_stem = clean_stem
                final_output_name = f"{clean_stem}_{left_str}_{right_str}_w{window_str}_s{step_str}{norm_suffix}.tsv"
                final_output_path = phlag_base / args.dist_type / f"w{window_str}_s{step_str}" / pattern_stem / "caster" / final_output_name

    is_fasta = not (input_str.endswith('.tsv') or args.fasta_file.name == 'scores.tsv')
    if is_fasta or not final_output_path.exists():
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
            repo_root / "caster" / "data" / "bin" / binary_name,
            pathlib.Path.cwd() / "caster" / "data" / "bin" / binary_name,
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
            binary_path = repo_root / "caster" / "data" / "bin" / binary_name

        dstar_cpp = repo_root / "caster" / "data" / "dstar.cpp"
        if dstar_cpp.exists():
            if not binary_path.exists() or os.path.getmtime(dstar_cpp) > os.path.getmtime(binary_path):
                target_bin = repo_root / "caster" / "data" / "bin" / binary_name
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

            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_output_path, "w") as f:
                f.writelines(output_lines)
            print(f"Success: TSV output file generated at: {final_output_path}")
        finally:
            shutil.rmtree(temp_dir)

    print(f"Using scores file: {final_output_path}")

    # Plotting support
    if args.plot:
        plot_scores = "scores" in args.plot
        plot_dist = "dist" in args.plot
        plot_hist = "hist" in args.plot
        
        if plot_scores or plot_dist or plot_hist:
            sys.path.append(str(repo_root / "caster" / "results"))
            from caster_plot import CasterPlotter
            
            CasterPlotter(
                scores_file=str(final_output_path.resolve()),
                distribution=args.dist_type,
                data_dir=str(final_output_path.parent.resolve()),
                topologies=args.topologies,
                plot_dstar=args.plot_dstar,
                plot_scores=plot_scores,
                plot_dist=plot_dist,
                plot_hist=plot_hist,
            )

if __name__ == "__main__":
    main()
