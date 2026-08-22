import sys
import pathlib
import argparse


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Phlagster: Run caster then phlag end-to-end from a single FASTA input."
    )
    parser.add_argument(
        "input_file",
        type=pathlib.Path,
        help="Input FASTA file path"
    )
    parser.add_argument(
        "-d",
        "--dist-type",
        dest="dist_type",
        default="gaussian",
        choices=["gaussian", "gmm"],
        help="Distribution type, threaded through to both caster and phlag (default: gaussian)"
    )
    parser.add_argument(
        "--output-base",
        dest="output_base",
        default=None,
        help="Forwarded to both caster's and phlag's --output-base (default: unset, "
             "uses the normal '<dist-type>/w<W>_s<S>' output-path prefix). Caster "
             "accepts but ignores it -- scores.tsv always lives in one canonical, "
             "--output-base-independent location; only phlag's report.tsv honors it.",
    )
    parser.add_argument(
        "--bench",
        dest="bench",
        action="store_true",
        default=False,
        help="Forwarded to both caster's and phlag's --bench (default: unset). "
             "Set by benchmark's run_all() for its own subprocess invocations -- "
             "not meant to be passed by hand.",
    )
    parser.add_argument(
        "--no-plots",
        dest="no_plots",
        action="store_true",
        help="Skip every diagnostic plot in both stages (caster's scatter.png, phlag's em.png/states.png/"
             "correlations.png) -- only scores.tsv and report.tsv are produced.",
    )
    parser.add_argument(
        "--np",
        dest="null_emission_parameterization",
        type=str.lower,
        default=None,
        choices=["free", "repulsion"],
        help="Forwarded to phlag's --np (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--ap",
        dest="alt_emission_parameterization",
        type=str.lower,
        default=None,
        choices=["free", "repulsion"],
        help="Forwarded to phlag's --ap (default: whatever phlag's own default is).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    from phlag import caster
    from phlag import phlag as phlag_main

    output_base_args = ["--output-base", args.output_base] if args.output_base else []
    bench_args = ["--bench"] if args.bench else []

    caster_plot_args = ["--plot"] if args.no_plots else ["--plot", "scores"]
    print(f"[phlagster] Running caster on '{args.input_file}' (-d {args.dist_type})...")
    scores_path = caster.main([str(args.input_file), "-d", args.dist_type] + output_base_args + bench_args + caster_plot_args)
    if scores_path is None:
        sys.exit("Error: caster did not produce a scores file.")

    # phlag's CLI requires -s/--step-size whenever --plot is passed at all, even
    # with no plot names -- unused by the report itself, so any placeholder
    # value satisfies it.
    phlag_plot_args = ["--plot", "-s", "1000"] if args.no_plots else []
    emission_param_args = []
    if args.null_emission_parameterization is not None:
        emission_param_args += ["--np", args.null_emission_parameterization]
    if args.alt_emission_parameterization is not None:
        emission_param_args += ["--ap", args.alt_emission_parameterization]
    # phlag has no -d/--dist-type flag -- model_design is inferred from the
    # scores.tsv/report.tsv filename, which already carries the dist_type
    # (see phlag.Phlag.extract_distribution_type_from_filename).
    print(f"[phlagster] Running phlag on '{scores_path}'...")
    phlag_main.main([str(scores_path)] + output_base_args + bench_args + emission_param_args + phlag_plot_args)


if __name__ == "__main__":
    main()
