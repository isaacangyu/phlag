benchmark --create store/phlag/gaussian/w25k_s25k/rho0.9_beta4.0/repulsion/lam1.5 -w 25k -s 25k --ap repulsion --lam 1.5 --rho 0.9 --beta 4.0 --skip caster;
benchmark --create store/phlag/gaussian/w50k_s50k/rho0.9_beta4.0/repulsion/lam1.5 -w 50k -s 50k --ap repulsion --lam 1.5 --rho 0.9 --beta 4.0 --skip caster;
benchmark --create store/phlag/gaussian/w10_s10/rho0.9_beta4.0/ -w 10 -s 10 --ilr --rho 0.9 --beta 4.0 --skip caster;
benchmark --create store/phlag/gaussian/w10_s10/rho0.9_beta4.0/repulsion/lam1.5 -w 10 -s 10 --ap repulsion --lam 1.5 --rho 0.9 --beta 4.0 --skip caster;
benchmark --create store/phlag/gaussian/w10_s10/rho0.9_beta4.0/repulsion/annealing/lam1.5 -w 10 -s 10 --ap repulsion --annealing --lam 1.5 --rho 0.9 --beta 4.0 --skip caster;