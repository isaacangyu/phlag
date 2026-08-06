import sys
import pathlib
import argparse


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Phlagster: Run caster then phlag end-to-end from a single FASTA or scores.tsv input."
    )
    parser.add_argument(
        "input_file",
        type=pathlib.Path,
        help="Input FASTA or scores.tsv file path"
    )
    parser.add_argument(
        "-d",
        "--dist-type",
        dest="dist_type",
        default="gaussian",
        choices=["gaussian", "gmm"],
        help="Distribution type, threaded through to both caster and phlag (default: gaussian)"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    from . import caster
    from . import phlag as phlag_main

    print(f"[phlagster] Running caster on '{args.input_file}' (-d {args.dist_type})...")
    scores_path = caster.main([str(args.input_file), "-d", args.dist_type])
    if scores_path is None:
        sys.exit("Error: caster did not produce a scores file.")

    print(f"[phlagster] Running phlag on '{scores_path}' (-d {args.dist_type})...")
    phlag_main.main([str(scores_path), "-d", args.dist_type])


if __name__ == "__main__":
    main()
