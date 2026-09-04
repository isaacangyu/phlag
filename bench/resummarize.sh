# Regenerates runs.tsv/analysis.tsv for finished gaussian, non-var2x, non-ilr
# store/phlag trees at standard window sizes (1,10,100,1k,2k,5k,10k,25k,50k) with
# step==window (non-overlapping), whose null_mean_*/alt_mean_*/pooled_mean_*/
# *_norm columns were wrong -- see project_gt_stats_window_bug memory /
# bench/benchmark.py's main()/BenchmarkStats.__init__/_build_record fix. Each
# line is --create against an ALREADY-FINISHED tree, so run_all() skips
# caster/phlag and this only re-runs summarize(). Flags are read back from that
# tree's own args.json.
#
# ILR trees excluded entirely: write_ground_truth_stats (phlag/caster.py) never
# writes gt_stats.txt for an ILR-variant scores.tsv (it only has c*ILR1/c*ILR2,
# not the raw ABBA/BABA/AABB columns the ground-truth split needs), so
# null_mean_*/alt_mean_*/pooled_mean_* are structurally None for every ILR run --
# resummarizing them fixes nothing.
#
# Regenerated against a `ps aux` snapshot -- any tree with a live --create/--copy
# process at generation time was excluded. RE-CHECK `ps aux` before running this
# again if time has passed -- this list is a snapshot, not a live guarantee.
benchmark --create store/phlag/gaussian/w100_s100/normalize -w 100 -s 100 -d gaussian -n --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w100_s100/repulsion/annealing/lam1.5 -w 100 -s 100 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w100_s100/repulsion -w 100 -s 100 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w100_s100/rho0.9_beta4.0 -w 100 -s 100 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w100_s100/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 100 -s 100 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w100_s100/rho0.9_beta4.0/repulsion/lam1.5 -w 100 -s 100 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10_s10/repulsion/annealing/lam1.5 -w 10 -s 10 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w10_s10/repulsion -w 10 -s 10 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w10_s10/rho0.9_beta4.0 -w 10 -s 10 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10_s10/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 10 -s 10 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10_s10/rho0.9_beta4.0/repulsion/lam1.5 -w 10 -s 10 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10k_s10k/normalize -w 10000 -s 10000 -d gaussian -n --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10k_s10k/repulsion/annealing/lam1.5 -w 10000 -s 10000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w10k_s10k/repulsion -w 10000 -s 10000 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w10k_s10k/rho0.9_beta4.0 -w 10000 -s 10000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10k_s10k/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 10000 -s 10000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w10k_s10k/rho0.9_beta4.0/repulsion/lam1.5 -w 10000 -s 10000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w1k_s1k/repulsion/annealing/lam1.5 -w 1000 -s 1000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w1k_s1k/repulsion -w 1000 -s 1000 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w1k_s1k/rho0.9_beta4.0 -w 1000 -s 1000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w1k_s1k/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 1000 -s 1000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w1k_s1k/rho0.9_beta4.0/repulsion/lam1.5 -w 1000 -s 1000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w25k_s25k/normalize -w 25000 -s 25000 -d gaussian -n --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w25k_s25k/repulsion/annealing/lam1.5 -w 25000 -s 25000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w25k_s25k/repulsion -w 25000 -s 25000 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w25k_s25k/rho0.9_beta4.0 -w 25000 -s 25000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w25k_s25k/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 25000 -s 25000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w25k_s25k/rho0.9_beta4.0/repulsion/lam1.5 -w 25000 -s 25000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w2k_s2k/normalize -w 2000 -s 2000 -d gaussian -n --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w2k_s2k/repulsion/annealing/lam1.5 -w 2000 -s 2000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w2k_s2k/repulsion -w 2000 -s 2000 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w2k_s2k/rho0.9_beta4.0 -w 2000 -s 2000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w2k_s2k/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 2000 -s 2000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w2k_s2k/rho0.9_beta4.0/repulsion/lam1.5 -w 2000 -s 2000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w50k_s50k/normalize -w 50000 -s 50000 -d gaussian -n --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w50k_s50k/repulsion/annealing/lam1.5 -w 50000 -s 50000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w50k_s50k/repulsion -w 50000 -s 50000 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w50k_s50k/rho0.9_beta4.0 -w 50000 -s 50000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w50k_s50k/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 50000 -s 50000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w50k_s50k/rho0.9_beta4.0/repulsion/lam1.5 -w 50000 -s 50000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w5k_s5k/normalize -w 5000 -s 5000 -d gaussian -n --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w5k_s5k/repulsion/annealing/lam1.5 -w 5000 -s 5000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w5k_s5k/repulsion -w 5000 -s 5000 -d gaussian --np free --ap repulsion -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w5k_s5k/rho0.9_beta4.0 -w 5000 -s 5000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w5k_s5k/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 5000 -s 5000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --annealing --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/w5k_s5k/rho0.9_beta4.0/repulsion/lam1.5 -w 5000 -s 5000 -d gaussian --np free --ap repulsion -L 10 --lam 1.5 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1 --rho 0.9 --beta 4.0;
