# benchmark -w 100-25k --no-floor-fix --rho 0.9 --beta 4 --ap repulsion
# benchmark -w 100-25k --no-floor-fix --rho 0.9 --beta 4 --ap repulsion --annealing
# benchmark -w 100-25k --rho 0.9 --beta 4 --topologies snr-avg

# changes that may be affected by null randomization
# benchmark -w 100-25k
# benchmark -w 100-25k --rho 0.9 --beta 4
# benchmark -w 100-25k --ap repulsion
# benchmark -w 100-25k --topologies AABB
# benchmark -w 100-25k --np null-alt-equal
# benchmark -w 100-25k --all-null -b --rho 0.9 --beta 4

benchmark -w 100-25k --pareto 
benchmark -w 100-25k --pareto --all-null