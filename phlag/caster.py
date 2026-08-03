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
        
    from .utils import get_data_dir, get_repo_root, resolve_input_file, get_most_recent_file
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
            sys.exit("Error: No FASTA file found in store/msa/concat or candidate MSA directories.")
        print(f"Using most recent FASTA file: {recent_fasta}")
        args.fasta_file = recent_fasta
    else:
        # If passed a scores.tsv path (e.g. store/phlag/.../caster/scores.tsv), map to corresponding fasta in simulations
        input_str = str(args.fasta_file)
        if "scores.tsv" in input_str or input_str.endswith(".tsv") or "caster" in args.fasta_file.parts:
            # Extract simulation name and clean locus stem from path structure
            parts = args.fasta_file.parts
            sim_name = None
            locus_stem = None
            
            for idx, p in enumerate(parts):
                if "_" in p and ("admixture" in p or "recombination" in p or "down" in p or "up" in p):
                    sim_name = p
                    if idx + 1 < len(parts):
                        locus_stem = parts[idx + 1]
                    break
            
            if sim_name and locus_stem:
                from .utils import get_data_dir, get_simulation_categories, get_short_sim_name
                cats = get_simulation_categories(sim_name)
                short_sim = get_short_sim_name(sim_name)
                candidate_bases = []
                data_dir = get_data_dir()
                if cats:
                    for sname in set([short_sim, sim_name]):
                        candidate_bases.append(data_dir / "simulations" / cats[0] / cats[1] / sname)
                        candidate_bases.append(pathlib.Path("/drive2/iang/simulations") / cats[0] / cats[1] / sname)
                for sname in set([short_sim, sim_name]):
                    candidate_bases.append(data_dir / "simulations" / sname)
                    candidate_bases.append(pathlib.Path("/drive2/iang/simulations") / sname)
                
                cand_fasta = None
                for base in candidate_bases:
                    if (base / "concat" / f"{locus_stem}.fa").exists():
                        cand_fasta = base / "concat" / f"{locus_stem}.fa"
                        break
                    elif (base / f"{locus_stem}.fa").exists():
                        cand_fasta = base / f"{locus_stem}.fa"
                        break
                
                if cand_fasta:
                    args.fasta_file = cand_fasta
                else:
                    args.fasta_file = resolve_input_file(args.fasta_file, default_subdirs=["msa/concat", "msa", "concat", "concat_*", "simulations"], default_exts=[".fa", ".fasta", ".fa.gz"])
        else:
            args.fasta_file = resolve_input_file(args.fasta_file, default_subdirs=["msa/concat", "msa", "concat", "concat_*", "simulations"], default_exts=[".fa", ".fasta", ".fa.gz"])




                
    if args.mapping is None:
        # Auto-detect mapping file based on simulation parent directory node for quadripartition (using left node for admixture)
        stem = args.fasta_file.stem
        sim_dir = args.fasta_file.parent if args.fasta_file.parent.name not in ["concat"] and not args.fasta_file.parent.name.startswith("concat_") else args.fasta_file.parent.parent
        sim_dir_name = sim_dir.name
        
        # Check if sim_name was present in original input path parts (e.g. store/phlag/.../Strigiformes_N297_admixture_.../...)
        for p in pathlib.Path(input_str).parts if 'input_str' in locals() else args.fasta_file.parts:
            if "admixture" in p or "recombination" in p or "10X" in p:
                sim_dir_name = p
                break
        
        # Determine left node for quadripartition mapping
        if "_" in sim_dir_name:
            node_name = sim_dir_name.split("_")[0]  # Left node for quadripartition mapping
        else:
            node_name = sim_dir_name
            
        default_map = sim_dir / f"neoaves_{node_name}_mapping.tsv"
        
        if default_map.exists():
            args.mapping = default_map
        else:
            # Fallback 1: search for mapping file under simulations directory using sim_dir_name or node_name
            from .utils import get_simulation_categories, get_short_sim_name
            cats = get_simulation_categories(sim_dir_name)
            short_sim = get_short_sim_name(sim_dir_name)
            map_subdirs = []
            if cats:
                for sname in set([short_sim, sim_dir_name]):
                    map_subdirs.append(f"simulations/{cats[0]}/{cats[1]}/{sname}")
            map_subdirs.extend(["simulations/null", f"simulations/{sim_dir_name}", "simulations", "mapping"])
            sim_map = resolve_input_file(f"neoaves_{node_name}_mapping.tsv", default_subdirs=map_subdirs, default_exts=[".tsv"])
            if not sim_map:
                sim_map = resolve_input_file(f"avian_{node_name}_mapping.tsv", default_subdirs=map_subdirs, default_exts=[".tsv"])
            if sim_map and sim_map.exists():
                args.mapping = sim_map
            else:
                # Fallback 2: check if simulation directory or parent has a neoaves_*_mapping.tsv file
                neo_maps = list(sim_dir.glob("neoaves_*_mapping.tsv"))
                if not neo_maps:
                    neo_maps = list(args.fasta_file.parent.glob("neoaves_*_mapping.tsv"))
                if not neo_maps and args.fasta_file.parent.parent.exists():
                    neo_maps = list(args.fasta_file.parent.parent.glob("neoaves_*_mapping.tsv"))
                if neo_maps:
                    args.mapping = neo_maps[0]
                else:
                    # Fallback 3: try general resolve_input_file
                    resolved = resolve_input_file("neoaves_*_mapping.tsv", default_subdirs=["simulations", "mapping"], default_exts=[".tsv"])
                    if resolved and resolved.exists():
                        args.mapping = resolved
                    else:
                        sys.exit(f"Error: No mapping file found for '{args.fasta_file.name}'. Please specify a mapping file using --mapping.")




    else:
        # Resolve Mapping file fallback if not found
        args.mapping = resolve_input_file(args.mapping, default_subdirs=["mapping"], default_exts=[".tsv", "_mapping.tsv", ".txt"])

    if not args.mapping or not args.mapping.exists():
        sys.exit(f"Error: No valid mapping file found for '{args.fasta_file.name}'. Please specify a mapping file using --mapping.")
        
    print(f"Using mapping file: {args.mapping}")
    
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
        
        # Display mapping clade information for the simulation/source node if present
        sim_dir_name = args.fasta_file.parent.name if args.fasta_file.parent.name != "concat" else args.fasta_file.parent.parent.name
        
        # Display source node information for admixture simulations (taking the right taxon node e.g. N297 from Strigiformes_N297_admixture_...)
        target_node = None
        if "admixture" in sim_dir_name.lower():
            parts_dir = sim_dir_name.split("_")
            if len(parts_dir) >= 2:
                target_node = parts_dir[1]  # Right taxon node
                print(f"Admixture source node: {target_node}")
        else:
            if "_" in sim_dir_name:
                target_node = sim_dir_name.split("_")[0]
            else:
                target_node = sim_dir_name

        # Resolve tree file (defaulting to store/63K.tre)
        tree_path = args.tree_file
        if tree_path is None:
            tree_candidates = [
                repo_root / "store" / "63K.tre",
                pathlib.Path.cwd() / "store" / "63K.tre",
                pathlib.Path.cwd() / "63K.tre",
                data_dir / "store" / "63K.tre",
                repo_root / "63K.tre"
            ]
            for cand in tree_candidates:
                if cand.exists():
                    tree_path = cand
                    break
        else:
            tree_path = resolve_input_file(tree_path, default_subdirs=["store", "tree"], default_exts=[".tre", ".tree", ".nwk"])

        if tree_path and tree_path.exists() and target_node:
            try:
                with open(tree_path, "r") as tf:
                    tree_str = tf.read().strip()
                
                # 1. Direct leaf or named internal node match: NodeName:length or NodeName)length or )NodeName:length
                pattern = re.escape(target_node) + r'(?:[^\):]*):([0-9.]+)'
                match = re.search(pattern, tree_str)
                if not match:
                    # 2. Support-value prefix before branch length e.g. )1:0.797287 or )Nyctibiidae:0.5
                    pattern = r'\)[^:\)]*' + re.escape(target_node) + r'[^:\)]*:([0-9.]+)'
                    match = re.search(pattern, tree_str)
                
                if match:
                    branch_len = float(match.group(1))
                    print(f"Internal node '{target_node}' CU branch length: {branch_len:.6f}")
                else:
                    # 3. Read species/taxa from mapping file if available
                    mapping_taxa = []
                    if args.mapping and args.mapping.exists():
                        with open(args.mapping, "r") as mf:
                            for mline in mf:
                                mparts = mline.strip().split()
                                if len(mparts) >= 2:
                                    mapping_taxa.append(mparts[0])
                    
                    found_len = None
                    for taxon in mapping_taxa:
                        t_match = re.search(re.escape(taxon) + r'[^:]*:([0-9.]+)', tree_str)
                        if t_match:
                            found_len = float(t_match.group(1))
                            print(f"Internal node '{target_node}' (via taxon {taxon}) CU branch length: {found_len:.6f}")
                            break
                    if not found_len:
                        # 4. Partial taxon substring match
                        taxon_match = re.search(r'([A-Za-z0-9_]*' + re.escape(target_node) + r'[A-Za-z0-9_]*):([0-9.]+)', tree_str, re.IGNORECASE)
                        if taxon_match:
                            print(f"Internal node '{target_node}' (matching {taxon_match.group(1)}) CU branch length: {float(taxon_match.group(2)):.6f}")
                        else:
                            print(f"Tree loaded ({tree_path.name}), target node '{target_node}' branch length not found.")
            except Exception as ex:
                pass




        
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
                
        # Perform rolling window average with O(1) sliding window
        K = max(1, args.window_size // args.step_size)
        
        # Pre-parse numeric columns for all rows
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
            # Initialize running sum for the first window
            run_abba = sum(r[2] for r in parsed_rows[:K])
            run_baba = sum(r[3] for r in parsed_rows[:K])
            run_aabb = sum(r[4] for r in parsed_rows[:K])
            run_qcnt = sum(r[5] for r in parsed_rows[:K])
            
            for i in range(len(parsed_rows) - K + 1):
                if i > 0:
                    # Subtract outgoing element (i - 1) and add incoming element (i + K - 1)
                    outgoing = parsed_rows[i - 1]
                    incoming = parsed_rows[i + K - 1]
                    run_abba += incoming[2] - outgoing[2]
                    run_baba += incoming[3] - outgoing[3]
                    run_aabb += incoming[4] - outgoing[4]
                    run_qcnt += incoming[5] - outgoing[5]
                
                pos_val = parsed_rows[i][1]  # Position of the start of the window
                
                # Average scores by dividing by K sub-windows
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
            if "simulations" in parts:
                sim_dir = args.fasta_file.parent
                if sim_dir.name in ["concat"] or sim_dir.name.startswith("concat_"):
                    sim_dir = sim_dir.parent
                sim_name = sim_dir.name
                
                from .utils import get_simulation_categories, get_short_sim_name
                cats = get_simulation_categories(sim_name)
                short_sim = get_short_sim_name(sim_name)
                if cats:
                    final_output_path = data_dir / "phlag" / args.dist_type / f"w{window_str}_s{step_str}" / cats[0] / cats[1] / short_sim / clean_stem / "caster" / "scores.tsv"
                else:
                    final_output_path = data_dir / "phlag" / args.dist_type / f"w{window_str}_s{step_str}" / short_sim / clean_stem / "caster" / "scores.tsv"
                is_sim = True
            
            if not is_sim:
                final_output_name = f"{clean_stem}_{left_str}_{right_str}_w{window_str}_s{step_str}{norm_suffix}.tsv"
                final_output_path = data_dir / "phlag" / args.dist_type / f"w{window_str}_s{step_str}" / clean_stem / "caster" / final_output_name

            
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
