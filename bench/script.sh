benchmark --create store/phlag/gaussian/w250k_s1k --config bench/config.json; 
benchmark --create store/phlag/gaussian/w250k_s1k/repulsion --skip caster --config bench/config1.json; 
benchmark --create store/phlag/gaussian/w250k_s1k/repulsion/annealing --skip caster --config bench/config2.json; 
benchmark --create store/phlag/gaussian/w250k_s1k/repulsion/annealing/lam1.5 --skip caster --config bench/config3.json; 
benchmark --create store/phlag/gaussian/w250k_s1k/var2x/repulsion --skip caster --config bench/config4.json; 
benchmark --create store/phlag/gaussian/w250k_s1k/var2x/repulsion/annealing --skip caster --config bench/config5.json; 
benchmark --create store/phlag/gaussian/w250k_s1k/var2x/repulsion/annealing/lam1.5 --skip caster --config bench/config6.json