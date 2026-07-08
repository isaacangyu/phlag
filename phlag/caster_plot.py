import sys
import pathlib
import argparse

# Resolve repo root and append caster/results to path
repo_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(repo_root / "caster" / "results"))

from caster_histogram import CasterPlotter

def main():
    parser = argparse.ArgumentParser(description="Empirical topology and parametric fit plotter.")
    parser.add_argument("scores_file", type=str, help="Path to CASTER scores TSV file.")
    parser.add_argument("-d", dest="distribution", type=str, default=None, help="Optional parametric distribution to fit.")
    parser.add_argument("-t", dest="topologies", type=str, nargs="+", default=None, help="List of topologies to plot.")
    
    args = parser.parse_args()
    CasterPlotter(scores_file=args.scores_file, distribution=args.distribution, topologies=args.topologies)

if __name__ == "__main__":
    main()
