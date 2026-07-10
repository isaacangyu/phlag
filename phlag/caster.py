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
        type=pathlib.Path,
        help="Input FASTA file path"
    )
    
    # Left and Right indices
    parser.add_argument(
        "-l",
        dest="left",
        type=int_or_abbrev,
        default=0,
        help="Left index of range (0-indexed, inclusive)"
    )
    parser.add_argument(
        "-r",
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
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Inject defaults for CLI flags if not provided
    if args.step_size is None:
        args.step_size = args.window_size
        
    repo_root = pathlib.Path(__file__).parent.parent.resolve()
    
    # Resolve FASTA file fallback if not found
    if not args.fasta_file.exists():
        fallback_fasta = repo_root / "caster" / "data" / args.fasta_file.name
        if fallback_fasta.exists():
            args.fasta_file = fallback_fasta
        else:
            fallback_fasta2 = repo_root / args.fasta_file
            if fallback_fasta2.exists():
                args.fasta_file = fallback_fasta2
                
    if args.mapping is None:
        # Auto-detect mapping file in same directory as fasta_file
        default_map = args.fasta_file.parent / f"{args.fasta_file.stem}_mapping.tsv"
        if default_map.exists():
            args.mapping = default_map
        else:
            fallback_map = args.fasta_file.parent / "ape_mapping.tsv"
            if fallback_map.exists():
                args.mapping = fallback_map
    else:
        # Resolve Mapping file fallback if not found
        if not args.mapping.exists():
            fallback_map = repo_root / "caster" / "data" / args.mapping.name
            if fallback_map.exists():
                args.mapping = fallback_map
            else:
                fallback_map2 = repo_root / args.mapping
                if fallback_map2.exists():
                    args.mapping = fallback_map2
    
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
        # Script is in phlag/caster.py, repo root is one level up
        binary_name = "dstar.exe" if sys.platform == "win32" else "dstar"
        binary_path = repo_root / "caster" / "data" / "bin" / binary_name
            
        if not binary_path.exists():
            sys.exit(f"Error: Could not locate compiled 'dstar' binary at '{binary_path}'")
            
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
                    
        # 5. Write final TSV file to current directory
        # Output filename should not have C, and includes left and right indices (formatted)
        clean_stem = args.fasta_file.stem.replace("C", "")
        left_str = format_val(args.left)
        right_str = format_val(args.right)
        window_str = format_val(args.window_size)
        step_str = format_val(args.step_size)
        
        final_output_name = f"{clean_stem}_{left_str}_{right_str}_w{window_str}_s{step_str}{norm_suffix}.tsv"
        final_output_path = repo_root / "caster" / "data" / final_output_name
        
        with open(final_output_path, "w") as f:
            f.writelines(output_lines)
        print(f"Success: TSV output file generated at: {final_output_path}")
        print(result.stdout)
        
        # Inline plotting support
        if args.plot:
            plot_scores = "scores" in args.plot
            plot_dist = "dist" in args.plot
            
            if plot_scores or plot_dist:
                sys.path.append(str(repo_root / "caster" / "results"))
                from caster_histogram import CasterPlotter
                
                CasterPlotter(
                    scores_file=str(final_output_path.resolve()),
                    distribution="gaussian",
                    data_dir=str(final_output_path.parent.resolve()),
                    plot_dstar=args.normalize,
                    plot_scores=plot_scores,
                    plot_dist=plot_dist,
                )
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
