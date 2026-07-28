import sys
import pathlib
import os
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
        description="Caster: Compute D* statistic on a genomic range."
    )
    
    # Input FASTA file
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
        default=10000,
        help="Window size (default: 10000)"
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
        default=None,
        help="Step size (default: same as window size)"
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
        choices=["scores", "dist"],
        default=["scores", "dist"],
        help="List of plots to generate (choices: scores, dist. Default: both)",
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
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Inject defaults for CLI flags if not provided
    if args.step_size is None:
        args.step_size = args.window_size
        
    from .utils import get_data_dir, get_repo_root, resolve_input_file, get_most_recent_file
    repo_root = get_repo_root()
    data_dir = get_data_dir()
    
    # Resolve FASTA file fallback if not found or recent flag requested
    if args.recent or args.fasta_file == pathlib.Path("-r") or args.fasta_file is None:
        recent_fasta = get_most_recent_file(
            default_subdirs=["store/msa/concat", "msa/concat", "concat", "store/msa", "msa"],
            default_exts=[".fa", ".fasta", ".fa.gz"]
        )
        if recent_fasta is None or not recent_fasta.exists():
            sys.exit("Error: No FASTA file found in store/msa/concat or candidate MSA directories.")
        print(f"Using most recent FASTA file: {recent_fasta}")
        args.fasta_file = recent_fasta
    else:
        args.fasta_file = resolve_input_file(args.fasta_file, default_subdirs=["msa/concat", "msa", "concat"], default_exts=[".fa", ".fasta", ".fa.gz"])
                
    if args.mapping is None:
        # Auto-detect mapping file in repo root mapping directory or fasta_file parent
        stem = args.fasta_file.stem
        # 1. Exact match
        repo_map = repo_root / "mapping" / f"{stem}_mapping.tsv"
        default_map = args.fasta_file.parent / f"{stem}_mapping.tsv"
        
        # 2. Check clade name extraction if full path contains clade substring
        clade_map = None
        mapping_dir = repo_root / "mapping"
        if mapping_dir.exists():
            full_path_str = str(args.fasta_file.resolve())
            for mfile in mapping_dir.glob("*_mapping.tsv"):
                clade = mfile.stem.replace("_mapping", "").split("_")[-1]
                if clade and clade in full_path_str:
                    clade_map = mfile
                    break
        
        if repo_map.exists():
            args.mapping = repo_map
        elif default_map.exists():
            args.mapping = default_map
        elif clade_map and clade_map.exists():
            args.mapping = clade_map
        else:
            sys.exit(f"Error: No mapping file found for '{args.fasta_file.name}'. Please specify a mapping file using --mapping.")
    else:
        # Resolve Mapping file fallback if not found
        args.mapping = resolve_input_file(args.mapping, default_subdirs=["mapping"], default_exts=[".tsv", "_mapping.tsv", ".txt"])

    if not args.mapping or not args.mapping.exists():
        sys.exit(f"Error: No valid mapping file found for '{args.fasta_file.name}'. Please specify a mapping file using --mapping.")
    
    # 1. Validation
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
        
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 2. Locate dstar binary
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

        # Check if dstar binary needs compilation or re-compilation
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
            
        # 3. Run dstar binary on original fasta file directly (no splicing)
        print(f"Running D* calculation on original file '{args.fasta_file}' with window={args.window_size}, step={args.step_size}...")
        cmd = [str(binary_path), str(args.fasta_file.resolve())]
        if args.mapping:
            if not args.mapping.exists():
                sys.exit(f"Error: Mapping file not found at '{args.mapping}'")
            cmd.append(str(args.mapping.resolve()))
        else:
            cmd.append("-")
            
        cmd.append(str(args.step_size))
        try:
            result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Error running 'dstar' binary:\nCommand: {e.cmd}\nExit Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            
        # 4. Parse TSV lines from stdout and perform rolling window sum
        norm_suffix = "_n" if args.normalize else ""
        
        lines = result.stdout.splitlines()
        if not lines:
            sys.exit(f"Error: D* output was empty. Stderr: {result.stderr}")
            
        # Find index of header
        header_idx = 0
        for idx, line in enumerate(lines):
            if "pos" in line:
                header_idx = idx
                break
                
        raw_rows = []
        for line in lines[header_idx+1:]:
            if not line.strip():
                continue
            parts = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(parts) >= 7:
                raw_rows.append(parts)
                
        if not raw_rows:
            sys.exit(f"Error: No window scores calculated for '{args.fasta_file.name}' using mapping '{args.mapping.name}'. Please check that sequence headers in the FASTA match species in the mapping file.")
                
        # Perform rolling window sum
        K = max(1, args.window_size // args.step_size)
        
        results = []
        for i in range(len(raw_rows) - K + 1):
            window_rows = raw_rows[i : i + K]
            
            sum_abba = sum(float(row[2]) for row in window_rows)
            sum_baba = sum(float(row[3]) for row in window_rows)
            sum_aabb = sum(float(row[4]) for row in window_rows)
            sum_qcnt = sum(float(row[6]) for row in window_rows)
            
            pos_val = int(window_rows[0][1]) # Position of the start of the window
            
            # Recalculate D* for the combined window
            denom = sum_abba + sum_baba + sum_aabb
            dstar_val = (sum_abba - sum_baba) / denom if denom != 0 else 0.0
            
            if args.left <= pos_val < args.right:
                file_val = window_rows[0][0]
                results.append({
                    'file': file_val,
                    'pos': pos_val,
                    'abba': sum_abba,
                    'baba': sum_baba,
                    'aabb': sum_aabb,
                    'dstar': dstar_val,
                    'qcnt': sum_qcnt
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
                        
        output_lines = []
        header = "file\tpos\tc*ABBA\tc*BABA\tc*AABB\tD*\tQuartetCnt\n"
        output_lines.append(header)
        
        for r in results:
            output_lines.append(f"{r['file']}\t{r['pos']}\t{r['abba']:.6g}\t{r['baba']:.6g}\t{r['aabb']:.6g}\t{r['dstar']:.6g}\t{r['qcnt']:.0f}\n")
                    
        # 5. Write final TSV file to parsed directory structure
        clean_stem = args.fasta_file.stem
        
        from .utils import parse_filename_to_dir_structure
        parsed = parse_filename_to_dir_structure(clean_stem)
        
        if parsed:
            rel_dir = parsed["relative_dir"]
            final_output_path = data_dir / "phlag" / args.dist_type / rel_dir / "caster" / "scores.tsv"
        else:
            left_str = format_val(args.left)
            right_str = format_val(args.right)
            window_str = format_val(args.window_size)
            step_str = format_val(args.step_size)
            
            parts = args.fasta_file.parts
            is_sim = False
            if "simulations" in parts and "concat" in parts:
                sim_idx = parts.index("simulations")
                if sim_idx + 1 < len(parts):
                    sim_name = parts[sim_idx + 1]
                    final_output_path = data_dir / "phlag" / args.dist_type / sim_name / f"w{window_str}_s{step_str}" / clean_stem / "caster" / "scores.tsv"
                    is_sim = True
            
            if not is_sim:
                final_output_name = f"{clean_stem}_{left_str}_{right_str}_w{window_str}_s{step_str}{norm_suffix}.tsv"
                final_output_path = data_dir / "scores" / final_output_name
            
        os.makedirs(final_output_path.parent, exist_ok=True)
        
        with open(final_output_path, "w") as f:
            f.writelines(output_lines)
        print(f"Success: TSV output file generated at: {final_output_path}")
        
        # Inline plotting support
        if args.plot:
            plot_scores = "scores" in args.plot
            plot_dist = "dist" in args.plot
            
            if plot_scores or plot_dist:
                sys.path.append(str(repo_root / "caster" / "results"))
                from caster_plot import CasterPlotter
                
                CasterPlotter(
                    scores_file=str(final_output_path.resolve()),
                    distribution="gaussian",
                    data_dir=str(final_output_path.parent.resolve()),
                    topologies=args.topologies,
                    plot_dstar=args.plot_dstar,
                    plot_scores=plot_scores,
                    plot_dist=plot_dist,
                )
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
