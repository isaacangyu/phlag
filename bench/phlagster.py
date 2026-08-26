import sys
import pathlib
import argparse

from phlag.caster import int_or_abbrev


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
        "-w",
        dest="window_size",
        type=int_or_abbrev,
        default=None,
        help="Forwarded to caster's -w (default: whatever caster's own default is).",
    )
    parser.add_argument(
        "-s",
        "--step-size",
        dest="step_size",
        type=int_or_abbrev,
        default=None,
        help="Forwarded to both caster's -s and phlag's -s/--step-size (default: "
             "whatever caster's own default is).",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Forwarded to caster's -n.",
    )
    parser.add_argument(
        "--shift-caster",
        dest="shift_caster",
        action="store_true",
        help="Forwarded to caster's --shift-caster.",
    )
    parser.add_argument(
        "--pair",
        dest="pair",
        action="store_true",
        help="Forwarded to caster's --pair (caster-pair/quartet-scoring mode instead of dstar).",
    )
    parser.add_argument(
        "--chunk",
        dest="chunk_size",
        type=int_or_abbrev,
        default=None,
        help="Forwarded to caster's --chunk (default: whatever caster's own default is).",
    )
    parser.add_argument(
        "--chunk-scores",
        dest="chunk_scores",
        type=pathlib.Path,
        default=None,
        help="Forwarded to caster's --chunk-scores (default: whatever caster's own default is).",
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
    parser.add_argument(
        "-L",
        "--n-iters",
        dest="n_iters",
        type=int_or_abbrev,
        default=None,
        help="Forwarded to phlag's -L/--n-iters (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--lam",
        dest="emission_lambda",
        type=float,
        default=None,
        help="Forwarded to phlag's --lam (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--double-variance-init",
        dest="double_variance_init",
        action="store_true",
        help="Forwarded to phlag's --double-variance-init.",
    )
    parser.add_argument(
        "--repulsion-optimizer",
        dest="repulsion_optimizer",
        type=str.lower,
        default=None,
        choices=["lm", "gd"],
        help="Forwarded to phlag's --repulsion-optimizer (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--annealing",
        dest="annealing",
        action="store_true",
        help="Forwarded to phlag's --annealing.",
    )
    parser.add_argument(
        "--mu",
        dest="lm_damping",
        type=float,
        default=None,
        help="Forwarded to phlag's --mu (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "-t",
        "--silhouette-threshold",
        dest="silhouette_threshold",
        type=float,
        default=None,
        help="Forwarded to phlag's -t/--silhouette-threshold (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "-p",
        "--best-paths",
        dest="best_paths",
        type=int,
        default=None,
        help="Forwarded to phlag's -p/--best-paths (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--correct-transition",
        dest="correct_transition",
        nargs="?",
        const="auto",
        default=None,
        help="Forwarded to phlag's --correct-transition.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    from phlag import caster
    from phlag import phlag as phlag_main

    output_base_args = ["--output-base", args.output_base] if args.output_base else []
    bench_args = ["--bench"] if args.bench else []

    caster_extra_args = []
    if args.window_size is not None:
        caster_extra_args += ["-w", str(args.window_size)]
    if args.step_size is not None:
        caster_extra_args += ["-s", str(args.step_size)]
    if args.normalize:
        caster_extra_args += ["-n"]
    if args.shift_caster:
        caster_extra_args += ["--shift-caster"]
    if args.pair:
        caster_extra_args += ["--pair"]
    if args.chunk_size is not None:
        caster_extra_args += ["--chunk", str(args.chunk_size)]
    if args.chunk_scores is not None:
        caster_extra_args += ["--chunk-scores", str(args.chunk_scores)]

    caster_plot_args = ["--plot"] if args.no_plots else ["--plot", "scores"]
    print(f"[phlagster] Running caster on '{args.input_file}' (-d {args.dist_type})...")
    scores_path = caster.main(
        [str(args.input_file), "-d", args.dist_type]
        + caster_extra_args + output_base_args + bench_args + caster_plot_args
    )
    if scores_path is None:
        sys.exit("Error: caster did not produce a scores file.")

    # phlag's CLI requires -s/--step-size whenever --plot is passed at all, even
    # with no plot names -- unused by the report itself, so any placeholder
    # value satisfies it.
    phlag_step_size = args.step_size if args.step_size is not None else 1000
    phlag_plot_args = ["--plot", "-s", str(phlag_step_size)] if args.no_plots else []
    phlag_extra_args = []
    if args.null_emission_parameterization is not None:
        phlag_extra_args += ["--np", args.null_emission_parameterization]
    if args.alt_emission_parameterization is not None:
        phlag_extra_args += ["--ap", args.alt_emission_parameterization]
    if args.n_iters is not None:
        phlag_extra_args += ["-L", str(args.n_iters)]
    if args.emission_lambda is not None:
        phlag_extra_args += ["--lam", str(args.emission_lambda)]
    if args.double_variance_init:
        phlag_extra_args += ["--double-variance-init"]
    if args.repulsion_optimizer is not None:
        phlag_extra_args += ["--repulsion-optimizer", args.repulsion_optimizer]
    if args.annealing:
        phlag_extra_args += ["--annealing"]
    if args.lm_damping is not None:
        phlag_extra_args += ["--mu", str(args.lm_damping)]
    if args.silhouette_threshold is not None:
        phlag_extra_args += ["-t", str(args.silhouette_threshold)]
    if args.best_paths is not None:
        phlag_extra_args += ["-p", str(args.best_paths)]
    if args.correct_transition is not None:
        phlag_extra_args += ["--correct-transition", args.correct_transition]
    # phlag has no -d/--dist-type flag -- model_design is inferred from the
    # scores.tsv/report.tsv filename, which already carries the dist_type
    # (see phlag.Phlag.extract_distribution_type_from_filename).
    print(f"[phlagster] Running phlag on '{scores_path}'...")
    phlag_main.main(
        [str(scores_path)] + output_base_args + bench_args
        + phlag_extra_args + phlag_plot_args
    )


if __name__ == "__main__":
    main()
