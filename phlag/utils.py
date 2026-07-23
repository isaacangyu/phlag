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


def get_data_dir():
    """
    Resolves the data directory to use. Looks up the PHLAG_DIR environment variable
    or reads it from a .env file. Defaults to repo_root / "caster" / "data" if not found.
    """
    import os
    import pathlib
    phlag_dir = os.environ.get("PHLAG_DIR")
    if not phlag_dir:
        # Search for .env file at repo root
        repo_root = pathlib.Path(__file__).parent.parent.resolve()
        # Also try current directory just in case
        for base_dir in [repo_root, pathlib.Path.cwd()]:
            env_path = base_dir / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, val = line.split("=", 1)
                                if key.strip() == "PHLAG_DIR":
                                    phlag_dir = val.strip().strip("'").strip('"')
                                    break
                except Exception:
                    pass
            if phlag_dir:
                break
    if phlag_dir:
        return pathlib.Path(phlag_dir)
    else:
        repo_root = pathlib.Path(__file__).parent.parent.resolve()
        return repo_root / "caster" / "data"

