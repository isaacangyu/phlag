import time
import argparse

from functools import wraps

def count_lines(filepath):
    try:
        with open(filepath, "r") as f:
            line_count = sum(1 for line in f)
        return line_count
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

def parse_pattern_string(pattern_str, block_size_bp=500000, total_span=None):
    """
    Parses pattern strings representing locus blocks, percentage intervals, or coordinate intervals.
    
    Supported formats:
      1. Standard token sequence: e.g., 'n1n2n3n4n5n6n8a1a2a3n10n11' (12-chunk pattern)
      2. Range-token sequence: e.g., 'n1-n6,a1-a3,n10-n11' or 'n1-n6_a1-a3_n10-n11'
      3. Percentage interval lists: e.g. '70-80,85-100' or '40-45' (marked as state 1 / anomaly)
      4. Prepended locus pattern + interval suffix: e.g., 'n4n6n3n8n9a3a1a2n1n10n12n2_40-65'
      5. Coordinate intervals: e.g. '0-3000000,3000000-4500000,4500000-6000000'

    Returns:
      (blocks, anomaly_intervals, total_length_bp)
    """
    import re
    blocks = []
    anomaly_intervals = []
    curr_pos = 0

    # 0. Check if there is a trailing/standalone explicit interval range suffix (e.g. '_40-65', ';40-65', '_70-80,85-100', '70-80;85-100')
    range_suffix_match = re.search(r'(?:^|[;_,])(\d+-\d+(?:[;_,]\d+-\d+)*)$', pattern_str)
    if range_suffix_match:
        suffix_str = range_suffix_match.group(1)
        coord_matches = re.findall(r'(\d+)-(\d+)', suffix_str)
        if coord_matches:
            all_vals = [int(v) for pair in coord_matches for v in pair]
            max_val = max(all_vals) if all_vals else 0
            first_start = int(coord_matches[0][0])
            
            # Case A: Percentage intervals (0-100) (e.g. 40-65 or 70-80,85-100)
            if max_val == 100 or (first_start >= 15 and max_val <= 100):
                span = total_span if total_span is not None and total_span > 0 else 6000000
                for s_str, e_str in coord_matches:
                    s_pct, e_pct = int(s_str) / 100.0, int(e_str) / 100.0
                    start_bp = s_pct * span
                    end_bp = e_pct * span
                    anomaly_intervals.append((start_bp, end_bp))
                    blocks.append(('a', f"{s_str}%-{e_str}%", end_bp - start_bp))
                return blocks, anomaly_intervals, span

            # Case B: Contiguous base-pair coordinate partition starting at 0
            elif coord_matches[0][0] == '0' and max_val > 100:
                for idx, (s_str, e_str) in enumerate(coord_matches):
                    s_bp, e_bp = int(s_str), int(e_str)
                    b_type = 'a' if (len(coord_matches) == 3 and idx == 1) or (len(coord_matches) > 3 and 0.25 <= (idx / len(coord_matches)) <= 0.75) else ('n' if idx % 2 == 0 else 'a')
                    length_bp = max(0, e_bp - s_bp)
                    blocks.append((b_type, f"{s_bp}-{e_bp}", length_bp))
                    if b_type == 'a':
                        anomaly_intervals.append((s_bp, e_bp))
                    curr_pos = max(curr_pos, e_bp)
                return blocks, anomaly_intervals, curr_pos

    # 1. Try range tokens with explicit 'n' or 'a' prefixes (e.g. n1-n6,a1-a3 or n1n2n3a1a2...)
    if re.search(r'[an]\d+', pattern_str, re.IGNORECASE):
        range_matches = re.findall(r'([an])(\d+)-(?:([an])?(\d+))', pattern_str, re.IGNORECASE)
        if range_matches:
            for b_type, start_str, _, end_str in range_matches:
                b_type = b_type.lower()
                start_id = int(start_str)
                end_id = int(end_str)
                
                if end_id >= start_id and (end_id - start_id) < 1000:
                    for chunk_id in range(start_id, end_id + 1):
                        blocks.append((b_type, str(chunk_id), block_size_bp))
                        if b_type == 'a':
                            anomaly_intervals.append((curr_pos, curr_pos + block_size_bp))
                        curr_pos += block_size_bp
                else:
                    length_bp = max(0, end_id - start_id)
                    blocks.append((b_type, f"{start_id}-{end_id}", length_bp))
                    if b_type == 'a':
                        anomaly_intervals.append((start_id, end_id))
                    curr_pos = max(curr_pos, end_id)
        else:
            # Fallback to individual chunk tokens like n1n2n3a1a2...
            token_matches = re.findall(r'([an])(\d+)', pattern_str, re.IGNORECASE)
            for b_type, b_id in token_matches:
                b_type = b_type.lower()
                blocks.append((b_type, b_id, block_size_bp))
                if b_type == 'a':
                    anomaly_intervals.append((curr_pos, curr_pos + block_size_bp))
                curr_pos += block_size_bp

    # 2. Chunk index ranges without 'a'/'n' prefix (e.g. 1-6,7-9,10-12)
    if not blocks:
        coord_matches = re.findall(r'(\d+)-(\d+)', pattern_str)
        if coord_matches:
            for idx, (s_str, e_str) in enumerate(coord_matches):
                s_val, e_val = int(s_str), int(e_str)
                b_type = 'a' if (len(coord_matches) == 3 and idx == 1) or (len(coord_matches) > 3 and 0.25 <= (idx / len(coord_matches)) <= 0.75) else ('n' if idx % 2 == 0 else 'a')
                for chunk_id in range(s_val, e_val + 1):
                    blocks.append((b_type, str(chunk_id), block_size_bp))
                    if b_type == 'a':
                        anomaly_intervals.append((curr_pos, curr_pos + block_size_bp))
                    curr_pos += block_size_bp

    return blocks, anomaly_intervals, curr_pos

def get_short_sim_name(sim_name):
    """
    Strips redundant category suffixes/infixes from simulation names to match shortened folder names on disk.
    e.g. 'Nyctibiidae_10X_up' -> 'Nyctibiidae'
         'N109_10X_down' -> 'N109'
         'N252_recombination_down' -> 'N252'
         'N482_N477_admixture_rate090-time2599554' -> 'N482_N477_rate090-time2599554'
    """
    s = str(sim_name)
    s = s.replace("_10X_up", "").replace("_10X_down", "")
    s = s.replace("_recombination_up", "").replace("_recombination_down", "")
    s = s.replace("_admixture_", "_")
    return s

def get_simulation_categories(sim_name):
    """
    Categorizes a simulation name or path into two subdirectory levels.
    Returns (level1, level2) or () if not a recognized 10X/recombination/admixture simulation.
    
    10X:
      up: '10X_up', '10X/up', 'Nyctibiidae', 'N497', 'N544', 'N554', 'N716' -> ('10X', 'up')
      down: '10X_down', '10X/down', 'N109', 'N498', 'N717' -> ('10X', 'down')
      
    recombination:
      up: 'recombination_up', 'recombination/up' -> ('recombination', 'up')
      down: 'recombination_down', 'recombination/down', 'N252' -> ('recombination', 'down')
      
    admixture:
      'admixture', or ('rate' and 'time') -> ('admixture', 'low' if time < 4.0 else 'high')
      Time is parsed from 'time<NUM>' in sim_name (if > 1000, divided by 1e6).
    """
    import re
    s = str(sim_name)
    if "10X_down" in s or "10X/down" in s:
        return ("10X", "down")
    elif "10X_up" in s or "10X/up" in s:
        return ("10X", "up")
    elif "recombination_down" in s or "recombination/down" in s:
        return ("recombination", "down")
    elif "recombination_up" in s or "recombination/up" in s:
        return ("recombination", "up")
    elif "admixture" in s or ("rate" in s and "time" in s):
        m = re.search(r'time(\d+(?:\.\d+)?)', s)
        if m:
            val = float(m.group(1))
            t_val = val / 1000000.0 if val > 1000 else val
            sub2 = "low" if t_val < 4.0 else "high"
        else:
            sub2 = "low"
        return ("admixture", sub2)
    elif "10X" in s:
        if "down" in s or any(k in s for k in ["N109", "N498", "N717"]):
            return ("10X", "down")
        if "up" in s or any(k in s for k in ["Nyctibiidae", "N497", "N544", "N554", "N716"]):
            return ("10X", "up")
    elif "recombination" in s:
        if "down" in s or "N252" in s:
            return ("recombination", "down")
        if "up" in s:
            return ("recombination", "up")
    # Check node stubs
    if any(k in s for k in ["Nyctibiidae", "N497", "N544", "N554", "N716"]):
        return ("10X", "up")
    if any(k in s for k in ["N109", "N498", "N717"]):
        return ("10X", "down")
    if "N252" in s:
        return ("recombination", "down")
    return ()

def parse_filename_to_dir_structure(filename):
    """
    Parses a filename like 'null-neoaves_alt-Nyctibiidae_10X_up_n1n8a1n5_0_2m_w50k_s1k'
    or 'null-neoaves_alt-Nyctibiidae_10X_up_1-6,7-9,10-11_w50k_s1k'
    Returns a dict with 'null', 'alt', 'pattern', 'locus', 'window_step'.
    """
    import re
    pattern_regex = r'((?:[a-zA-Z0-9;_-]+[;_,])?(?:[an]?\d+-[an]?\d+[;_,]?)+|(?:[an]\d+)+|\d+-\d+(?:[;_,]\d+-\d+)*)'
    # Fallback structure
    m = re.search(r'null-(.*?)_alt-(.*?)_' + pattern_regex + r'_(\d+_\d+m)_(w\w+_s\w+)', filename)
    if m:
        alt_name = m.group(2)
        short_alt = get_short_sim_name(alt_name)
        cats = get_simulation_categories(alt_name)
        cat_prefix = f"{cats[0]}/{cats[1]}/" if cats else ""
        return {
            "null": m.group(1),
            "alt": alt_name,
            "pattern": m.group(3),
            "locus": m.group(4),
            "window_step": m.group(5),
            "relative_dir": f"{m.group(5)}/{cat_prefix}{short_alt}/{m.group(3)}"
        }
    # Attempt parsing without locus chunk
    m2 = re.search(r'null-(.*?)_alt-(.*?)_' + pattern_regex + r'_(w\w+_s\w+)', filename)
    if m2:
        alt_name = m2.group(2)
        short_alt = get_short_sim_name(alt_name)
        cats = get_simulation_categories(alt_name)
        cat_prefix = f"{cats[0]}/{cats[1]}/" if cats else ""
        return {
            "null": m2.group(1),
            "alt": alt_name,
            "pattern": m2.group(3),
            "window_step": m2.group(4),
            "relative_dir": f"{m2.group(4)}/{cat_prefix}{short_alt}/{m2.group(3)}"
        }
    return None

def integer_pair(arg_str):
    try:
        parts = arg_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                "Argument must contain exactly two integers separated by a comma."
            )
        int1 = int(parts[0].strip())
        int2 = int(parts[1].strip())
        return [int1, int2]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer pair format: {arg_str}. {e}")


def is_float(val):
    try:
        return float(val) == float(val)
    except (ValueError, TypeError):
        return False


def limited_float(min_val, max_val):
    def check_range(value):
        fvalue = float(value)
        if not (min_val <= fvalue <= max_val):
            raise ValueError(
                f"Argument must be between {min_val} and {max_val} (inclusive)."
            )
        return fvalue

    return check_range


def limited_int(min_val, max_val):
    def check_range(value):
        ivalue = int(value)
        if not (min_val <= ivalue <= max_val):
            raise ValueError(
                f"Argument must be between {min_val} and {max_val} (inclusive)."
            )
        return ivalue

    return check_range


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper


def get_repo_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent


def get_data_dir():
    """
    Returns the user data output directory where scores and HMM state files should be stored.
    Checks CONNECTION_DIR environment variable first, then defaults to /drive2/iang/phlag if present,
    else repository 'caster/data' directory.
    """
    import os
    import pathlib
    conn_env = os.environ.get("CONNECTION_DIR")
    if conn_env:
        return pathlib.Path(conn_env)
    
    drive2_dir = pathlib.Path("/drive2/iang/phlag")
    if drive2_dir.exists():
        return drive2_dir
        
    return get_repo_root() / "caster" / "data"


def resolve_input_file(path_input, default_subdirs=None, default_exts=None):
    """
    Resolves a user-provided file or directory path flexibly.
    Supports absolute paths, relative paths, filenames without directory,
    stem names without extension, directory names containing fasta files,
    and automatic searches within repo store/msa/simulations directories.
    """
    import pathlib

    if path_input is None:
        return None

    path_obj = pathlib.Path(path_input)

    # 1. Direct path check (file exists)
    if path_obj.exists() and path_obj.is_file():
        return path_obj.resolve()

    # 2. Build candidate filenames
    candidates = []
    if path_obj.suffix:
        candidates.append(path_obj)
    else:
        if default_exts:
            for ext in default_exts:
                if not ext.startswith("."):
                    ext = "." + ext
                candidates.append(pathlib.Path(str(path_obj) + ext))
        else:
            candidates.extend([
                pathlib.Path(str(path_obj) + ".fa"),
                pathlib.Path(str(path_obj) + ".fasta"),
                pathlib.Path(str(path_obj) + ".tsv"),
                pathlib.Path(str(path_obj) + ".txt"),
            ])

    # Direct candidate checks relative to CWD
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    # 3. Build search directories
    repo_root = get_repo_root()
    data_dir = get_data_dir()

    search_bases = [
        pathlib.Path.cwd(),
        data_dir,
        repo_root,
        repo_root / "connection_dir",
        repo_root / "connection_dir" / "simulations",
        repo_root / "store",
        repo_root / "test",
        pathlib.Path.cwd() / "test",
        pathlib.Path("/drive2/iang"),
    ]

    if default_subdirs is None:
        default_subdirs = [
            "store/msa/concat", "store/msa", "store", 
            "msa/concat", "msa", "concat", "simulations"
        ]

    search_dirs = []
    for base in search_bases:
        if base.exists():
            search_dirs.append(base)
            for s in default_subdirs:
                sub_path = base / s
                if sub_path.exists():
                    search_dirs.append(sub_path)
            try:
                for sim_sub in list(base.glob("simulations/*/*/*/concat")) + list(base.glob("simulations/*/concat")):
                    if sim_sub.is_dir():
                        search_dirs.append(sim_sub)
            except Exception:
                pass

    # Deduplicate search_dirs
    seen_dirs = set()
    unique_dirs = []
    for d in search_dirs:
        try:
            resolved_d = d.resolve()
            if resolved_d not in seen_dirs:
                seen_dirs.add(resolved_d)
                unique_dirs.append(d)
        except Exception:
            pass

    for cand in candidates:
        cand_name = cand.name
        for sdir in unique_dirs:
            check_path = sdir / cand_name
            if check_path.exists() and check_path.is_file():
                return check_path.resolve()

    # 5. Directory resolution
    clean_name = path_obj.name
    if clean_name.startswith("alt-"):
        clean_name = clean_name[4:]
    parts = clean_name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        clean_name = parts[0]

    cats = get_simulation_categories(clean_name) or get_simulation_categories(path_obj.name)
    short_clean = get_short_sim_name(clean_name)
    short_path_name = get_short_sim_name(path_obj.name)
    names_to_try = []
    for n in [short_clean, short_path_name, clean_name, path_obj.name]:
        if n and n not in names_to_try:
            names_to_try.append(n)

    dir_candidates = [path_obj]
    if cats:
        cat_rel = pathlib.Path(cats[0]) / cats[1]
        for sname in names_to_try:
            dir_candidates.extend([
                repo_root / "connection_dir" / "simulations" / cat_rel / sname,
                data_dir / "simulations" / cat_rel / sname,
                repo_root / "simulations" / cat_rel / sname,
                pathlib.Path("/drive2/iang/simulations") / cat_rel / sname,
            ])
    for sname in names_to_try:
        dir_candidates.extend([
            repo_root / "connection_dir" / "simulations" / sname,
            data_dir / "simulations" / sname,
            data_dir / sname,
            repo_root / "simulations" / sname,
            pathlib.Path("/drive2/iang/simulations") / sname,
        ])
    for dcand in dir_candidates:
        if dcand.exists() and dcand.is_dir():
            concat_dir = dcand / "concat"
            target_dirs = [concat_dir] if (concat_dir.exists() and concat_dir.is_dir()) else [dcand]
            for tdir in target_dirs:
                if tdir.name == "simulated" or tdir.parent.name == "simulated":
                    continue
                files = list(tdir.glob("*.fa")) + list(tdir.glob("*.fasta")) + list(tdir.glob("*.fa.gz"))
                # Exclude any files under a simulated subfolder
                files = [f for f in files if "simulated" not in f.parts]
                if files:
                    # Pull most recent FASTA file in concat/ or directory
                    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    return files[0].resolve()

    return path_obj


def get_most_recent_file(default_subdirs=None, default_exts=None, exclude_prefixes=None, target_dir_name=None):
    """
    Finds the most recently created or modified file in store/scores, store/msa/concat,
    or specified subdirectories, ignoring report output files.
    """
    import os
    import pathlib

    repo_root = get_repo_root()
    data_dir = get_data_dir()

    if exclude_prefixes is None:
        exclude_prefixes = ["report_", "gaussian_", "scores_", "em_", "walkthrough", "implementation_plan"]

    if default_subdirs is None:
        default_subdirs = ["store/scores", "scores", "store/msa/concat", "msa/concat"]

    search_dirs = []
    bases_to_search = [repo_root, pathlib.Path.cwd(), data_dir, pathlib.Path("/drive2/iang")]

    if target_dir_name:
        for base in bases_to_search:
            if base.exists():
                try:
                    for d in base.rglob(target_dir_name):
                        if d.is_dir():
                            search_dirs.append(d)
                except Exception:
                    pass
    
    for s in default_subdirs:
        for base in bases_to_search:
            sp = base / s
            if sp.exists():
                search_dirs.append(sp)

    # Also search recursively in model output directories (store/phlag, phlag, data_dir) for scores files
    for base in [repo_root / "store" / "phlag", repo_root / "phlag", data_dir / "phlag", data_dir]:
        if base.exists():
            try:
                for score_file in base.rglob("scores.tsv"):
                    if score_file.is_file():
                        search_dirs.append(score_file.parent)
                for score_file in base.rglob("*.tsv"):
                    if score_file.is_file() and not any(score_file.name.startswith(p) for p in exclude_prefixes):
                        search_dirs.append(score_file.parent)
            except Exception:
                pass

    seen_dirs = set()
    newest_file = None
    newest_mtime = -1.0

    for d in search_dirs:
        try:
            resolved_d = d.resolve()
            if target_dir_name and resolved_d.name != target_dir_name:
                continue
            if resolved_d in seen_dirs:
                continue
            seen_dirs.add(resolved_d)

            for item in d.iterdir():
                if item.is_file():
                    fname = item.name
                    if exclude_prefixes and any(fname.startswith(p) for p in exclude_prefixes):
                        continue
                    if default_exts and not any(fname.endswith(ext) for ext in default_exts):
                        continue
                    mtime = item.stat().st_mtime
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest_file = item.resolve()
        except Exception:
            pass

    return newest_file



