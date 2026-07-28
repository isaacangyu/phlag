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

def parse_filename_to_dir_structure(filename):
    """
    Parses a filename like 'null-neoaves_alt-Nyctibiidae_10X_up_n1n8a1n5_0_2m_w50k_s1k'
    Returns a dict with 'null', 'alt', 'pattern', 'locus', 'window_step'.
    """
    import re
    # Fallback structure
    m = re.search(r'null-(.*?)_alt-(.*?)_((?:[an]\d+)+)_(\d+_\d+m)_(w\w+_s\w+)', filename)
    if m:
        return {
            "null": m.group(1),
            "alt": m.group(2),
            "pattern": m.group(3),
            "locus": m.group(4),
            "window_step": m.group(5),
            "relative_dir": f"{m.group(5)}/{m.group(3)}/{m.group(2)}"
        }
    # Attempt parsing without locus chunk
    m2 = re.search(r'null-(.*?)_alt-(.*?)_((?:[an]\d+)+)_(w\w+_s\w+)', filename)
    if m2:
        return {
            "null": m2.group(1),
            "alt": m2.group(2),
            "pattern": m2.group(3),
            "window_step": m2.group(4),
            "relative_dir": f"{m2.group(4)}/{m2.group(3)}/{m2.group(2)}"
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
        fvalue = int(value)
        if not (min_val <= fvalue <= max_val):
            raise ValueError(
                f"Argument must be an integer between {min_val} and {max_val} (inclusive)."
            )
        return fvalue

    return check_range


def timeit(func):
    """
    A decorator that measures the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result

    return wrapper


def get_repo_root():
    """
    Resolves the repository root directory.
    Checks:
    1. PHLAG_REPO_ROOT environment variable
    2. ~/phlag (pathlib.Path.home() / "phlag") if it exists
    3. pathlib.Path.cwd() if it contains a 'caster' or 'phlag' directory
    4. Fallback to pathlib.Path(__file__).parent.parent.resolve()
    """
    import os
    import pathlib
    env_root = os.environ.get("PHLAG_REPO_ROOT")
    if env_root:
        return pathlib.Path(env_root).resolve()
        
    home_phlag = pathlib.Path.home() / "phlag"
    if home_phlag.exists():
        return home_phlag.resolve()
        
    cwd = pathlib.Path.cwd()
    if (cwd / "caster").exists() or (cwd / "phlag").exists():
        return cwd.resolve()
        
    return pathlib.Path(__file__).parent.parent.resolve()


def get_data_dir():
    """
    Resolves the data directory to use. Looks up the PHLAG_DIR or LARGE_DIR environment variable
    or reads it from a .env file. Defaults to repo_root / "caster" / "data" if not found.
    """
    import os
    import pathlib
    target_dir = os.environ.get("PHLAG_DIR") or os.environ.get("LARGE_DIR")
    if not target_dir:
        repo_root = get_repo_root()
        # Search for .env file at repo root or cwd
        for base_dir in [repo_root, pathlib.Path.cwd()]:
            env_path = base_dir / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, val = line.split("=", 1)
                                key_str = key.strip()
                                if key_str in ("PHLAG_DIR", "LARGE_DIR"):
                                    target_dir = val.strip().strip("'").strip('"')
                                    break
                except Exception:
                    pass
            if target_dir:
                break
    if target_dir:
        return pathlib.Path(target_dir)
    else:
        repo_root = get_repo_root()
        return repo_root / "caster" / "data"


def resolve_input_file(path_input, default_subdirs=None, default_exts=None):
    """
    Resolves an input path that may be a full path, relative path, filename with extension,
    or base filename without path or extension.
    """
    import os
    import pathlib

    if path_input is None:
        return None

    path_obj = pathlib.Path(path_input)

    # 1. Direct check
    if path_obj.exists():
        return path_obj.resolve()

    if default_exts is None:
        default_exts = [".fa", ".fasta", ".tsv", ".txt", ".fa.gz"]

    if default_subdirs is None:
        default_subdirs = [
            "msa/concat", "msa", "concat", 
            "store/msa/concat", "store/msa", "store", 
            "scores", "mapping"
        ]
    else:
        expanded = []
        for s in default_subdirs:
            expanded.append(s)
            if "msa" in s:
                expanded.extend(["msa/concat", "concat", "store/msa/concat", "store/msa", "store"])
        default_subdirs = expanded

    # Generate extensions to try
    exts_to_try = [""]
    if not path_obj.suffix:
        exts_to_try.extend(default_exts)

    candidates = []
    for ext in exts_to_try:
        if ext:
            candidates.append(path_obj.with_suffix(ext))
        else:
            candidates.append(path_obj)

    # 2. Check candidates directly
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    # 3. Search in candidate directories
    repo_root = get_repo_root()
    data_dir = get_data_dir()

    search_bases = [
        pathlib.Path.cwd(),
        data_dir,
        repo_root,
        repo_root / "store",
        repo_root / "test",
        pathlib.Path.cwd() / "test",
        pathlib.Path("/drive2/iang"),
    ]

    search_dirs = []
    for base in search_bases:
        if base.exists():
            search_dirs.append(base)
            for s in default_subdirs:
                sub_path = base / s
                if sub_path.exists():
                    search_dirs.append(sub_path)

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
            if check_path.exists():
                return check_path.resolve()

    # 4. Fuzzy match by stem in search directories
    stem = path_obj.stem
    for sdir in unique_dirs:
        if sdir.exists():
            try:
                for f in sdir.iterdir():
                    if f.is_file() and f.stem == stem:
                        return f.resolve()
            except Exception:
                pass

    return path_obj


def get_most_recent_file(default_subdirs=None, default_exts=None, exclude_prefixes=None):
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
    for s in default_subdirs:
        for base in [repo_root, pathlib.Path.cwd(), data_dir, pathlib.Path("/drive2/iang")]:
            sp = base / s
            if sp.exists():
                search_dirs.append(sp)

    # Add simulations/*/concat from data_dir (LARGE_DIR)
    if data_dir and data_dir.exists():
        try:
            for concat_dir in data_dir.glob("simulations/*/concat"):
                if concat_dir.is_dir():
                    search_dirs.append(concat_dir)
        except Exception:
            pass

    seen_dirs = set()
    newest_file = None
    newest_mtime = -1.0

    for d in search_dirs:
        try:
            resolved_d = d.resolve()
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



