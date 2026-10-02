import sys
import pathlib
import argparse

from phlag.caster import int_or_abbrev, step_size_or_fraction

CASTER_PLOTS = ["scatter", "dist", "correlation", "topology_pairs", "quartet_counts", "sums", "ecdf", "qq", "qq-gaussian", "qq-exp", "qq-dexp"]
PHLAG_PLOTS = ["em", "states", "relerr", "correlations", "topologies_3d"]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Phlagster: Run caster then phlag end-to-end from a FASTA or an existing caster scores.tsv."
    )
    parser.add_argument(
        "input_file",
        type=pathlib.Path,
        nargs="+",
        help="Input FASTA(s), locus spec(s) (e.g. 10X/down/N109/37-62), or existing caster "
             "scores.tsv(s); multiple values run phlagster once per value with the same flags. "
             "For a scores.tsv, its source FASTA is recovered "
             "from the 'file' column). With a scores.tsv and no -w/-s, caster is skipped "
             "and phlag runs on that file as-is; with -w/-s, caster recomputes from the "
             "source FASTA at each new window/step."
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
        nargs="+",
        default=None,
        help="Forwarded to caster's -w (default: whatever caster's own default is). "
             "Multiple values run caster then phlag once per (-w, -s) combination "
             "(cartesian product). Must come after the input file.",
    )
    parser.add_argument(
        "-s",
        "--step-size",
        dest="step_size",
        type=step_size_or_fraction,
        nargs="+",
        default=None,
        help="Forwarded to caster's -s (default: whatever caster's own default is). "
             "A value with a decimal point is a ratio of -w (1.0 = non-overlapping); "
             "a whole value is a literal step. Multiple values combine with -w as "
             "a cartesian product.",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Forwarded to caster's -n.",
    )
    parser.add_argument(
        "--exp-minus",
        dest="exp_minus",
        action="store_true",
        help="Forwarded to caster's --exp-minus.",
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
        "--site",
        dest="site",
        action="store_true",
        help="Forwarded to caster's --site (caster-site quartet-scoring mode instead of dstar).",
    )
    parser.add_argument(
        "-z",
        "--zscale",
        dest="zscale",
        action="store_true",
        help="Forwarded to caster's -z/--zscale.",
    )
    parser.add_argument(
        "-i",
        "--ilr",
        dest="ilr",
        action="store_true",
        help="Forwarded to caster's -i/--ilr.",
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
        "--recompute",
        dest="recompute",
        action="store_true",
        help="Recompute scores.tsv/report.tsv even if they already exist for these "
             "flags (default: reuse them and only draw missing plots; always "
             "recomputes under --bench).",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        nargs="*",
        choices=CASTER_PLOTS + PHLAG_PLOTS,
        default=None,
        help="Plot names for either stage, routed to whichever one owns them "
             f"(caster: {', '.join(CASTER_PLOTS)}; phlag: {', '.join(PHLAG_PLOTS)}). "
             "A stage with none of its names listed makes no plots. Bare --plot "
             "requests every plot from both. Omitted: caster 'scatter' and phlag's "
             "own default, or nothing with --no-plots.",
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
        "-k",
        "--n-clusters",
        dest="n_clusters",
        type=int,
        default=None,
        help="Forwarded to phlag's -k/--n-clusters (default: whatever phlag's own default is).",
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
        "--dirichlet-mean",
        dest="dirichlet_mean",
        action="store_true",
        help="Forwarded to phlag's --dirichlet-mean.",
    )
    parser.add_argument(
        "--emission-param",
        dest="emission_param",
        type=str.lower,
        default=None,
        choices=["free", "zero-alt-anchor"],
        help="Forwarded to phlag's --emission-param (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--prior-init-probs",
        dest="prior_init_probs",
        action="store_true",
        help="Forwarded to phlag's --prior-init-probs.",
    )
    parser.add_argument(
        "--correct-transition",
        dest="correct_transition",
        nargs="?",
        const="auto",
        default=None,
        help="Forwarded to phlag's --correct-transition.",
    )
    parser.add_argument(
        "--rho",
        dest="rho",
        type=float,
        nargs="+",
        default=None,
        help="Forwarded to phlag's --rho; multiple values run the cartesian product (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--beta",
        dest="beta",
        type=float,
        nargs="+",
        default=None,
        help="Forwarded to phlag's --beta (default: whatever phlag's own default is).",
    )
    parser.add_argument(
        "--beta-prime",
        dest="beta_prime",
        type=float,
        nargs="+",
        default=None,
        help="Forwarded to phlag's --beta-prime (default: whatever phlag's own default is).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_arguments(raw_argv)
    from phlag.utils import expand_node_specs
    inputs = expand_node_specs(args.input_file)
    if len(inputs) > 1 or [str(p) for p in args.input_file] != inputs:
        rest = [t for t in raw_argv if pathlib.Path(t) not in set(args.input_file)]
        for input_file in inputs:
            print(f"[phlagster] multi-input -- running {input_file}...")
            main([str(input_file)] + rest)
        return
    args.input_file = args.input_file[0]

    from phlag import caster
    from phlag import phlag as phlag_main

    output_base_args = ["--output-base", args.output_base] if args.output_base else []
    bench_args = ["--bench"] if args.bench else []
    if not args.bench and not args.recompute:
        bench_args = ["--skip-existing"]

    caster_extra_args = []
    if args.window_size is not None:
        caster_extra_args += ["-w"] + [str(w) for w in args.window_size]
    if args.step_size is not None:
        caster_extra_args += ["-s"] + [str(s) for s in args.step_size]
    if args.normalize:
        caster_extra_args += ["-n"]
    if args.shift_caster:
        caster_extra_args += ["--shift-caster"]
    if args.pair:
        caster_extra_args += ["--pair"]
    if args.site:
        caster_extra_args += ["--site"]
    if args.zscale:
        caster_extra_args += ["-z"]
    if args.ilr:
        caster_extra_args += ["-i"]
    if args.exp_minus:
        caster_extra_args += ["--exp-minus"]
    if args.chunk_size is not None:
        caster_extra_args += ["--chunk", str(args.chunk_size)]
    if args.chunk_scores is not None:
        caster_extra_args += ["--chunk-scores", str(args.chunk_scores)]

    if args.plot == []:
        caster_plot_args, phlag_plot_args = ["--plot"], ["--plot"]
    elif args.plot is not None:
        caster_plot_args = ["--plot"] + ([p for p in args.plot if p in CASTER_PLOTS] or ["none"])
        phlag_plot_args = ["--plot"] + ([p for p in args.plot if p in PHLAG_PLOTS] or ["none"])
    elif args.no_plots:
        caster_plot_args, phlag_plot_args = ["--plot", "none"], ["--plot", "none"]
    else:
        caster_plot_args, phlag_plot_args = ["--plot", "scatter"], []
    from_scores = args.input_file.suffix == ".tsv"
    if from_scores and args.window_size is None and args.step_size is None:
        if not args.input_file.exists():
            sys.exit(f"Error: scores file '{args.input_file}' does not exist.")
        scores_paths = [args.input_file]
    else:
        print(f"[phlagster] Running caster on '{args.input_file}' (-d {args.dist_type})...")
        result = caster.main(
            [str(args.input_file), "-d", args.dist_type]
            + caster_extra_args + output_base_args + bench_args + caster_plot_args
        )
        scores_paths = result if isinstance(result, list) else [result]
        if not scores_paths or any(p is None for p in scores_paths):
            sys.exit("Error: caster did not produce a scores file.")

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
    if args.n_clusters is not None:
        phlag_extra_args += ["-k", str(args.n_clusters)]
    if args.best_paths is not None:
        phlag_extra_args += ["-p", str(args.best_paths)]
    if args.correct_transition is not None:
        phlag_extra_args += ["--correct-transition", args.correct_transition]
    if args.dirichlet_mean:
        phlag_extra_args += ["--dirichlet-mean"]
    if args.emission_param is not None:
        phlag_extra_args += ["--emission-param", args.emission_param]
    if args.prior_init_probs:
        phlag_extra_args += ["--prior-init-probs"]
    if args.rho is not None:
        phlag_extra_args += ["--rho"] + [str(v) for v in args.rho]
    if args.beta is not None:
        phlag_extra_args += ["--beta"] + [str(v) for v in args.beta]
    if args.beta_prime is not None:
        phlag_extra_args += ["--beta-prime"] + [str(v) for v in args.beta_prime]
    phlag_extra_args += ["-d", args.dist_type]
    for scores_path in scores_paths:
        print(f"[phlagster] Running phlag on '{scores_path}'...")
        phlag_main.main(
            [str(scores_path)] + output_base_args + bench_args
            + phlag_extra_args + phlag_plot_args
        )


if __name__ == "__main__":
    main()
