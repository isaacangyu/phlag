# Resumes store/phlag/ runs after killing the earlier batch mid-flight.
# Every one of these already has its own recorded args.json from before the
# kill (confirmed: all 42 have partial reports/), so passing -w/-s/--ap/etc
# again would conflict with resume ("can't also pass ... directly") even
# though the values match -- omit them and let --create replay the recorded
# args.json instead. --skip caster is not a mirrored flag, safe to keep.
#
# grep '^benchmark' bench/script.sh | sed 's/;$//' | \
#   xargs -P 42 -I CMD bash -c '
#     cmd="CMD"
#     name=$(echo "$cmd" | grep -oE "gaussian/[^ ]+" | sed "s#gaussian/##" | tr "/" "_")
#     eval "$cmd" > "logs/${name}.log" 2>&1
#     echo "done: $name (exit $?)"
#   '

# --- w50k_s1k ---
benchmark --create store/phlag/gaussian/w50k_s1k --skip caster;
benchmark --create store/phlag/gaussian/w50k_s1k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w50k_s1k/repulsion/annealing/lam1.5 --skip caster;

# --- w50k_s40k ---
benchmark --create store/phlag/gaussian/w50k_s40k --skip caster;
benchmark --create store/phlag/gaussian/w50k_s40k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w50k_s40k/repulsion/annealing/lam1.5 --skip caster;

# --- w100k_s1k ---
benchmark --create store/phlag/gaussian/w100k_s1k --skip caster;
benchmark --create store/phlag/gaussian/w100k_s1k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w100k_s1k/repulsion/annealing/lam1.5 --skip caster;

# --- w100k_s80k ---
benchmark --create store/phlag/gaussian/w100k_s80k --skip caster;
benchmark --create store/phlag/gaussian/w100k_s80k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w100k_s80k/repulsion/annealing/lam1.5 --skip caster;

# --- w250k_s1k ---
benchmark --create store/phlag/gaussian/w250k_s1k --skip caster;
benchmark --create store/phlag/gaussian/w250k_s1k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w250k_s1k/repulsion/annealing/lam1.5 --skip caster;

# --- w250k_s200k ---
benchmark --create store/phlag/gaussian/w250k_s200k --skip caster;
benchmark --create store/phlag/gaussian/w250k_s200k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w250k_s200k/repulsion/annealing/lam1.5 --skip caster;

# --- w500k_s1k ---
benchmark --create store/phlag/gaussian/w500k_s1k --skip caster;
benchmark --create store/phlag/gaussian/w500k_s1k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w500k_s1k/repulsion/annealing/lam1.5 --skip caster;

# --- w500k_s400k ---
benchmark --create store/phlag/gaussian/w500k_s400k --skip caster;
benchmark --create store/phlag/gaussian/w500k_s400k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w500k_s400k/repulsion/annealing/lam1.5 --skip caster;

# --- w50k_s50k ---
benchmark --create store/phlag/gaussian/w50k_s50k --skip caster;
benchmark --create store/phlag/gaussian/w50k_s50k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w50k_s50k/repulsion/annealing/lam1.5 --skip caster;

# --- w100k_s100k ---
benchmark --create store/phlag/gaussian/w100k_s100k --skip caster;
benchmark --create store/phlag/gaussian/w100k_s100k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w100k_s100k/repulsion/annealing/lam1.5 --skip caster;

# --- w250k_s250k ---
benchmark --create store/phlag/gaussian/w250k_s250k --skip caster;
benchmark --create store/phlag/gaussian/w250k_s250k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w250k_s250k/repulsion/annealing/lam1.5 --skip caster;

# --- w500k_s500k ---
benchmark --create store/phlag/gaussian/w500k_s500k --skip caster;
benchmark --create store/phlag/gaussian/w500k_s500k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w500k_s500k/repulsion/annealing/lam1.5 --skip caster;

# --- w10k_s1k ---
benchmark --create store/phlag/gaussian/w10k_s1k --skip caster;
benchmark --create store/phlag/gaussian/w10k_s1k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w10k_s1k/repulsion/annealing/lam1.5 --skip caster;

# --- w10k_s10k ---
benchmark --create store/phlag/gaussian/w10k_s10k --skip caster;
benchmark --create store/phlag/gaussian/w10k_s10k/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w10k_s10k/repulsion/annealing/lam1.5 --skip caster;

# === Additions: var2x (14 existing bases) + w1m (3 new bases, all 6 non-pair variants) ===
# var2x for existing bases: dstar is cached, --skip caster is safe/fast.
# w1m bases: dstar NOT cached (new window value) -- omit --skip caster so it
# generates fresh scores (~19s/file combined caster+phlag, confirmed by test).

# --- w50k_s1k var2x ---
benchmark --create store/phlag/gaussian/w50k_s1k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w50k_s1k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w50k_s1k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w50k_s40k var2x ---
benchmark --create store/phlag/gaussian/w50k_s40k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w50k_s40k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w50k_s40k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w100k_s1k var2x ---
benchmark --create store/phlag/gaussian/w100k_s1k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w100k_s1k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w100k_s1k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w100k_s80k var2x ---
benchmark --create store/phlag/gaussian/w100k_s80k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w100k_s80k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w100k_s80k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w250k_s1k var2x ---
benchmark --create store/phlag/gaussian/w250k_s1k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w250k_s1k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w250k_s1k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w250k_s200k var2x ---
benchmark --create store/phlag/gaussian/w250k_s200k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w250k_s200k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w250k_s200k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w500k_s1k var2x ---
benchmark --create store/phlag/gaussian/w500k_s1k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w500k_s1k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w500k_s1k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w500k_s400k var2x ---
benchmark --create store/phlag/gaussian/w500k_s400k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w500k_s400k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w500k_s400k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w50k_s50k var2x ---
benchmark --create store/phlag/gaussian/w50k_s50k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w50k_s50k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w50k_s50k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w100k_s100k var2x ---
benchmark --create store/phlag/gaussian/w100k_s100k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w100k_s100k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w100k_s100k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w250k_s250k var2x ---
benchmark --create store/phlag/gaussian/w250k_s250k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w250k_s250k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w250k_s250k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w500k_s500k var2x ---
benchmark --create store/phlag/gaussian/w500k_s500k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w500k_s500k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w500k_s500k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w10k_s1k var2x ---
benchmark --create store/phlag/gaussian/w10k_s1k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w10k_s1k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w10k_s1k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w10k_s10k var2x ---
benchmark --create store/phlag/gaussian/w10k_s10k/var2x --skip caster;
benchmark --create store/phlag/gaussian/w10k_s10k/var2x/repulsion --skip caster;
benchmark --create store/phlag/gaussian/w10k_s10k/var2x/repulsion/annealing/lam1.5 --skip caster;

# --- w1m_s1k (all variants, dstar uncached) ---
benchmark --create store/phlag/gaussian/w1m_s1k -w 1m -s 1k;
benchmark --create store/phlag/gaussian/w1m_s1k/repulsion -w 1m -s 1k --ap repulsion;
benchmark --create store/phlag/gaussian/w1m_s1k/repulsion/annealing/lam1.5 -w 1m -s 1k --ap repulsion --annealing --lam 1.5;
benchmark --create store/phlag/gaussian/w1m_s1k/var2x -w 1m -s 1k --double-variance-init;
benchmark --create store/phlag/gaussian/w1m_s1k/var2x/repulsion -w 1m -s 1k --double-variance-init --ap repulsion;
benchmark --create store/phlag/gaussian/w1m_s1k/var2x/repulsion/annealing/lam1.5 -w 1m -s 1k --double-variance-init --ap repulsion --annealing --lam 1.5;

# --- w1m_s800k (all variants, dstar uncached) ---
benchmark --create store/phlag/gaussian/w1m_s800k -w 1m -s 800k;
benchmark --create store/phlag/gaussian/w1m_s800k/repulsion -w 1m -s 800k --ap repulsion;
benchmark --create store/phlag/gaussian/w1m_s800k/repulsion/annealing/lam1.5 -w 1m -s 800k --ap repulsion --annealing --lam 1.5;
benchmark --create store/phlag/gaussian/w1m_s800k/var2x -w 1m -s 800k --double-variance-init;
benchmark --create store/phlag/gaussian/w1m_s800k/var2x/repulsion -w 1m -s 800k --double-variance-init --ap repulsion;
benchmark --create store/phlag/gaussian/w1m_s800k/var2x/repulsion/annealing/lam1.5 -w 1m -s 800k --double-variance-init --ap repulsion --annealing --lam 1.5;

# --- w1m_s1m (all variants, dstar uncached) ---
benchmark --create store/phlag/gaussian/w1m_s1m -w 1m -s 1m;
benchmark --create store/phlag/gaussian/w1m_s1m/repulsion -w 1m -s 1m --ap repulsion;
benchmark --create store/phlag/gaussian/w1m_s1m/repulsion/annealing/lam1.5 -w 1m -s 1m --ap repulsion --annealing --lam 1.5;
benchmark --create store/phlag/gaussian/w1m_s1m/var2x -w 1m -s 1m --double-variance-init;
benchmark --create store/phlag/gaussian/w1m_s1m/var2x/repulsion -w 1m -s 1m --double-variance-init --ap repulsion;
benchmark --create store/phlag/gaussian/w1m_s1m/var2x/repulsion/annealing/lam1.5 -w 1m -s 1m --double-variance-init --ap repulsion --annealing --lam 1.5;
