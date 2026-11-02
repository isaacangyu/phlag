# keeping floor fix since regularizer still better
benchmark -w 50-50k --topologies AABB --floor-fix --rho 0.9 --beta 4
benchmark -w 50-50k --topologies ABBA AABB --floor-fix --rho 0.9 --beta 4
benchmark -w 50-50k --topologies ABBA BABA --floor-fix --rho 0.9 --beta 4
benchmark -w 100-25k --snr --floor-fix 
benchmark -w 100-25k --snr --rho 0.9 --beta 4 --floor-fix 
# done per site but same either way
benchmark -w 100-25k --signal-avg --floor-fix 
benchmark -w 100-25k --signal-avg --rho 0.9 --beta 4 --floor-fix 
# done per site
benchmark -w 100-25k --signal-max --floor-fix 
benchmark -w 100-25k --signal-max --rho 0.9 --beta 4 --floor-fix 