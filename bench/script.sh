# Rerun phlag + resummarize the plain (transition prior free/free -- no
# --rho/--beta, no repulsion) baseline trees at the standard non-overlapping
# window sizes (1,10,100,1k,2k,5k,10k,25k,50k), now that store/caster's
# scores.tsv/gt_stats.txt have been backfilled to the per-site dstar average
# fix. Each tree's reports/ was deleted beforehand (scores.tsv left alone),
# so --create finds scores.tsv already present (caster skipped) and only
# reruns phlag, then resummarizes analysis.tsv/runs.tsv from the fresh
# reports.
benchmark --create store/phlag/gaussian/w1_s1 -w 1 -s 1 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w10_s10 -w 10 -s 10 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w100_s100 -w 100 -s 100 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w1k_s1k -w 1000 -s 1000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w2k_s2k -w 2000 -s 2000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w5k_s5k -w 5000 -s 5000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w10k_s10k -w 10000 -s 10000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w25k_s25k -w 25000 -s 25000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;
benchmark --create store/phlag/gaussian/w50k_s50k -w 50000 -s 50000 -d gaussian --np free --ap free -L 10 --lam 1.0 --repulsion-optimizer lm --mu 1.0 -t 0.5 -k 2 -p 1;