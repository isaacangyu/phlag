# rm -r logs/*;
# grep '^benchmark' bench/script.sh | sed 's/;$//' | \
#   xargs -P 35 -I CMD env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c '
#     cmd="CMD"
#     name=$(echo "$cmd" | grep -oE "gaussian/[^ ]+" | tail -1 | sed "s#gaussian/##" | tr "/" "_")
#     eval "$cmd" > "logs/${name}.log" 2>&1
#     echo "done: $name (exit $?)"
#   '

benchmark --create store/phlag/gaussian/c25k_s25k/normalize/rho0.9_beta4.0/repulsion/annealing/lam1.5 --pair --normalize -c 25k -s 25k --ap repulsion --lam 1.5 --annealing --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/c25k_s25k/zscale/rho0.9_beta4.0/repulsion/annealing/lam1.5 --pair --zscale -c 25k -s 25k --ap repulsion --lam 1.5 --annealing --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/c25k_s25k/site/normalize/rho0.9_beta4.0/repulsion/annealing/lam1.5 --site --normalize -c 25k -s 25k --ap repulsion --lam 1.5 --annealing --rho 0.9 --beta 4.0;
benchmark --create store/phlag/gaussian/c25k_s25k/site/zscale/rho0.9_beta4.0/repulsion/annealing/lam1.5 --site --zscale -c 25k -s 25k --ap repulsion --lam 1.5 --annealing --rho 0.9 --beta 4.0;
