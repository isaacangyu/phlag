# Session Sync

Shared status board for concurrent Claude sessions working in this repo. Each session keeps one section below, keyed by its own identifier, and updates it on every substantial turn.

---

## session-20260926-rho-beta-multi

**Status:** done
**Task:** `--rho`/`--beta`/`--beta-prime` accept multiple values, cartesian product with each other and `-w`/`-s` (phlag/phlag.py `main()`).
**Change:** nargs="+"; `--beta` set drops `--beta-prime` from product (would duplicate). Batch + `-o` → `<o.parent>/[w.._s..]/[rho.._beta[prime]..]/<o.name>`. bench/benchmark.py, phlagster.py untouched (still scalar).
**Also:** FASTA/locus-spec sibling resolution picked `quartet_counts.tsv` (not excluded by `is_score_candidate`); fixed. Still picks newest variant (e.g. exp-minus) -- pass explicit scores path to choose.

---

## session-20260926-transition-prior-low-rho

**Status:** done
**Task:** extend transition-prior sweep with low rho {0.001,0.01,0.1} x beta-prime {1e-6..1e-2}; artifact 7BULTFo9cDHw1bVHuJQrwC v2.
**Change:** none to repo; 480 scratchpad runs.
**Finding:** low rho beats best high-rho in 0/40 cells; fits don't mirror, prior forces ~50/50 occupancy (N113: p01~p10~1e-4); hurts weak-emission cells (Accipitriformes w10k 0.43->0). 184/480 NaN (n*rho-beta<1).

---

## session-20260926-gtrees-perf-check

**Status:** done
**Task:** Why are gtrees (phlag_orig) F1s ~0 on the 5 KDE branches (45-55)?
**Finding:** No bug in the report wrapper; the truth labels line up (the recombination Alt region's QQS is exactly [1,0,0]). Viterbi is all-Null on all 5 branches. Focal-edge QQS by region: recomb Alt is *less* discordant than Null (Null already covers it; AUC 0.03-0.24 = inverted); N109 is [1,0,0] everywhere (no signal); Accipitriformes differs only weakly (AUC 0.50); admixture has real signal (0.66 -> 0.45, AUC 0.75) but the posterior of Alt stays ≤0.001.
**Change:** none.
**Follow-up:** gene-tree provenance: concat/*.gtrees = pre-made Dryad trees, 125 per 500 kb chunk (1 per 4 kb); admixture N340 used genetrees/ (IQ-TREE, 500 bp every 8.5 kb, 706 trees). Only those 5 sims have trees. Picked high-signal replacements (pop-info CU): 10X/up N281 (2.62), 10X/down N473 (0.281), recomb/up N282 (3.17), recomb/down N682 (3.16). They need `estimate_gene_trees.py <exp> 45-55`, which the user runs.

---

## session-20260926-transition-prior-sweep

**Status:** done
**Task:** sweep --rho/--beta-prime on 5 KDE branches (45-55) across w10..w50k, oracle-tune per branch, artifact https://claude.ai/artifact/7BULTFo9cDHw1bVHuJQrwC
**Change:** none to repo; 2151 phlag runs via -o into scratchpad.
**Finding:** prior useless at w<=1k (emission AUC~0.5; even --correct-transition F1<=0.21); helps w5k-w10k on 10X (N109 w10k 0.51->0.88). Any psi entry <1 makes Dirichlet mode's -1 a negative pseudo-count -> NaN EM (0/878 NaN with all psi>=1, 496/1092 with beta<1); true beta=1 sits on that edge. Admixture F1=0 at >=1k under every config.

---

## session-20260925-phlag-multi-input

**Status:** done
**Task:** phlag.py and bench/phlagster.py take multiple input/locus-spec positionals.
**Change:** positional `nargs="*"`; `main()` re-invokes itself once per spec with the other argv tokens (composes with -w/-s batch); `parse_arguments` unwraps to a single path. phlagster: `input_file` `nargs="+"`, same per-input re-invoke in `main()`.
**Verified:** dry-run (stubbed `_run_single`) on N113/N115/N340 45-55, with and without `-w 1k 10k`.
**Follow-up:** standalone caster/phlag/phlagster root is now `out/msa` (`phlag.utils.get_out_root`, `get_out_relative_parts` strips `out[/msa]` for titles). Moved the 5 45-55 w10k trees out/ -> out/msa. `PHLAG_SOURCE_ROOTS`: gtrees -> `out/gtrees`, msa -> absolute `get_out_root()` (was cwd-relative, broke from bench/). Verified `_collect_run_df` returns both sources for all 5 branches.

---

## session-20260925-branches-all-rho-beta

**Status:** done (awaiting reports)
**Task:** caster.ipynb: cra.plot over the 5 KDE branches with every rho/beta config (BASELINE deepcopy minus --rho/--beta).
**Change:** new markdown+code cell after the gtrees-vs-msa branch cell: `config = deepcopy(BASELINE)`, del rho/beta, `phlag: [gtrees, msa]`, F1 + tpr_fpr.
**Verified:** same config on out/msa N281 45-55 and N276 37-62: all rho_beta/betaprime dirs come through as separate configs. The 5 branches still have no reports in either root.

---

## session-20260925-phlag-source-branches

**Status:** done (awaiting reports)
**Task:** cra.plot run-mode `"phlag"` axis: gtrees -> `out/phlag_orig`, msa -> `out/msa` (`PHLAG_SOURCE_ROOTS`); F1 + TPR/FPR for the 5 KDE-grid branches at w10k, branch length in titles.
**Change:** bench/utils.py: `_collect_run_df` (pops "phlag", reads `run` under each root, config prefixed by source), `run` may be `{label: path}` -> `_plot_branches` (cols=branches, rows=metrics, "tpr_fpr" = FPR/TPR scatter), `_branch_length_label` (report header, else population-info; admixture shows divergence time), `branch_length` column in collect_out_runs. caster.ipynb: new cell after KDE grid.
gtrees is in `WINDOWLESS_PHLAG_SOURCES`: reports have no w<W>_s<S> dir, so `collect_out_runs(windowless=True)` keeps them (window None, no gt_stats), axes don't filter them, and `_plot_branches` draws them at every window slot.
**Verified:** scratchpad copies of out/msa w10k reports as both sources. The 5 branches (45-55) have no reports in either root yet.

---

## session-20260925-exp-minus

**Status:** done
**Task:** `--exp-minus`: topology cols x -> exp(-x), to test whether windows look more Gaussian.
**Change:** caster.py (flag, `apply_exp_minus_to_scores_file`, `strip_exp_minus`; applied last after -n/-i/-z by recursing main() without the flag then transforming; nested `exp-minus` segment in adhoc + --bench paths; regen recognizes it), phlag.py/phlag utils.py/bench utils.py (strip `exp-minus` variant segment), bench/benchmark.py (CASTER_ARG_SPECS, flag, get_expected_caster_sim_dir/scores_path, BenchmarkStats), bench/phlagster.py (forward).
**Verified:** N281 45-55 w500 with -o scratchpad. Raw dstar is ~1e-2 scale, so exp(-x)≈1-x: skew just flips sign (2.42 -> -2.34), kurtosis unchanged.

---

## session-20260925-beta-prime

**Status:** done
**Task:** `--beta-prime X`: beta = X * n_windows, only when `--beta` omitted (still needs `--rho`); default unset.
**Change:** phlag.py (flag, psi, report.tsv path recovery), phlag/utils.py (`rho<X>_betaprime<Y>` segment + node-name regex), bench/benchmark.py (PHLAG_ARG_SPECS, flag, `rho_betaprime` SEGMENT_CONVENTION), bench/phlagster.py (forward), bench/utils.py (`_OUT_PARAM_SEGMENT`/recorded `beta_prime`).
**Verified:** N281 w500 (12000 windows): `--beta-prime 0.01` → beta 120; with `--beta 4` → 4.

---

## session-20260925-category-kde-grid

**Status:** done
**Task:** caster.ipynb: one branch per merged_category, 2 rows (w500_s500, w5k_s5k), each panel 6 ECDFs (3 topologies x Null/Alt) + per-topology Null/Alt means box (lower-right). Switched KDE->ECDF 2026-09-25; x-range now 0.5pct..max so ECDF tails reach 1.
**Change:** new markdown+code cells after the aggregate within-window-variance cells (before "# One Run"). `pick_branch` = first sorted sim under `store/caster/<w>/<cat>` with `37-62` scores.tsv in both windows (admixture uses `admixture/low`); `load_labeled_scores` reuses `CasterPlotter.resolve_topology_columns` + the notebook's existing parse_pattern_string Null/Alt split. Topology colors = CasterPlotter.topo_colors, Null dashed / Alt solid (em.png's convention), x clipped to 0.5-99.5 pct.
**Verified:** ran the cell headless against the real store; 2x5 grid renders.

---

## session-20260923-scatter-prediction-overlay

**Status:** done
**Task:** User: if `scatter.png` exists in the (non-`--bench`) experiment dir, overlay phlag's own predictions shaded in yellow, same style as the existing ground-truth shading.
**Change:** `caster.py` `CasterPlotter` gains `predicted_intervals` (list of `(start_bp, end_bp)`) + `_shade_predicted_intervals` (yellow `#F4D03F` axvspan, mirrors `_shade_locus_pattern`'s red), called from `plot_topology_scatter` right after the ground-truth shading — overlap reads as blended color, disagreement as pure red/yellow. `phlag.py` `Phlag._predicted_alt_intervals` (static) merges contiguous Viterbi-Alt windows into bp spans (step recovered from consecutive `pos` values, not stored elsewhere on Phlag); `compute_output`, right after the states.png block, checks `not args.bench` and `<caster_scores dir>/scatter.png` exists, then re-instantiates `CasterPlotter` on the same scores.tsv with `predicted_intervals` set (only `plot_scores` runs) to overwrite it in place. Independent of `--plot`.
**Verified:** real N276 `10X/down/37-62/w1k_s1k` scores.tsv copied to scratchpad, ran phlag end-to-end (`~/.local/share/mamba/envs/phlag/bin/python`) — scatter.png's md5/mtime changed, rendered image confirmed yellow "Predicted Alt" shading + legend entry alongside the existing red "Alt" band, blended in the overlap region. Scratchpad cleaned up after.

---

## session-20260923-report-tsv-replay

**Status:** done
**Task:** User: if phlag's positional arg is a relative path to an existing `report.tsv` (not scores.tsv), rerun phlag using that report's own recorded config, with this invocation's own flags taking precedence.
**Change:** `phlag.py` `parse_arguments` — report.tsv's first line already records the exact `sys.argv` that produced it (`initialize_output`). New check right after the initial `parser.parse_args`: if `caster_scores` is relative, named `report.tsv`, and exists, `shlex.split` that first line (minus program name) as recovered tokens, strip the report.tsv positional out of this run's own raw argv as override tokens, and recurse via `parse_arguments(recovered_tokens + override_tokens)` — argparse's left-to-right store semantics (no `action="append"` flags in this parser) make later (override) tokens win over earlier (recovered) ones automatically. Absolute report.tsv paths untouched (existing behavior).
**Verified:** real `out/10X/up/N635/37-62/w500_s500/gaussian/report.tsv` via `~/.local/share/mamba/envs/phlag/bin/python` — relative path replays the recorded `10X/up/N635/37-62 --plot -w 100 500 1k 2k 5k` invocation and resolves through the normal FASTA->sibling-scores path; new `--rho`/`--beta` passed this run land on top; absolute path to the same file falls through unchanged; a real scores.tsv path unaffected.

---

## session-20260922-within-window-variance

**Status:** active (code done, production backfill not started)
**Task:** Add within-window mean/variance (site-level spread inside one window, distinct from gt_stats' existing between-window covariance) to gt_stats.txt/runs.tsv/report.tsv; new aggregate cra.plot cell in caster.ipynb mirroring the existing per-topology-means-by-window-size figure; confirm per-window-average vs pooled-average equivalence.
**Change:** `caster.py` `write_ground_truth_stats` buckets the sibling non-overlapping w1_s1 (per-site) scores.tsv by `pos // window` (mirrors the notebook's existing `topology_variance_within_window`) into `<Label>WithinMean`/`<Label>WithinVariance` (only for `w<...>_s<...>` with step==window). Round-tripped via `write_gt_stats_file`/`read_gt_stats_file`/`collect_gt_stats` (phlag/utils.py), `_out_gt_stats_columns` (bench/utils.py, out/ tree), and `RunRecord`/`RUN_COLUMNS`/`TOPO_WITHIN_COLUMNS` (bench/benchmark.py, store tree). `phlag.py` report.tsv gets new within-window mean/variance lines per label (Overall = "total stats by averaging").
**Verified:** real N635 10X/up 37-62 data (scratchpad output only) — mean of per-bucket means exactly equals the pooled raw-site mean (float precision); mean of per-bucket variances is close to but NOT exactly the pooled raw-site variance (law of total variance: pooled = within + between-bucket variance of means; between term ~1000x smaller here, so they nearly agree for this dataset but aren't identically equal in general). New notebook cells (10-11 aggregate, 20-21 per-run) execute cleanly against the real store tree but render empty — existing gt_stats.txt/runs.tsv predate this change and need backfill.
**Next:** user to decide whether/how to backfill gt_stats.txt (rerun `write_ground_truth_stats` per existing scores.tsv, no caster/phlag recompute) + resummarize runs.tsv, before the new plots show real data.

---

## session-20260921-locus-spec

**Status:** done
**Task:** caster/phlag/phlagster accept `<category>/<sub>/<node>/<pattern>` (e.g. `recombination/down/N564/37-62`).
**Change:** `utils.resolve_locus_spec` -> `<sims root>/<cat>/<sub>/<node dir>/concat/<pattern>.fa` (node = full dir name, short name, or leading token). Hooked into `caster.main` (before regen check) and phlag `main` (before `resolve_input_file`, then existing FASTA->sibling-scores path); phlagster inherits via caster.
**Verified:** all three at `-w 2k` on N564/37-62 (dir removed after); a live user phlagster run was writing other windows there, untouched.

---

## session-20260921-phlagster-ws

**Status:** done
**Task:** phlagster accepts scores.tsv + multi `-w`/`-s`.
**Change:** `bench/phlagster.py`: `-w`/`-s` now `nargs="+"` (`-s` uses `step_size_or_fraction`), forwarded to caster's cartesian loop; phlag then runs on each returned scores path. `.tsv` input with no `-w`/`-s` skips caster, runs phlag as-is; with `-w`/`-s`, caster regen recomputes from the source FASTA.
**Verified:** stubbed caster/phlag dispatch only (no real run, to avoid writing to `out/`).

---

## session-20260921-em-kde

**Status:** done
**Task:** em.png: KDE (smooth curve) instead of histograms.
**Change:** `phlag.py` `PhlagPlotter._plot_kde` (seaborn `kdeplot`, clipped to plot range, skipped if <2 pts/zero spread) replaces both rows' `histplot`s; legend now "Null/Alt KDE"; dead `_compute_bin_edges` and `bin_edges` params removed.
**Update:** KDE lines dashed, Null light blue `#7CC4F2` / Alt orange `#FF9F1C` (`colors[..]["kde"]`); fits keep steel blue/coral.
**Update:** em.png "After EM" row now honors the Viterbi/Hamming flip: `Phlag.flipped_for_eval` (set in `compute_output`) -> `_plot_em_row` swaps state indices for KDE split + fitted curves. Checked on N276 w250k_s200k (flipped=True).
**Verified:** N276 w1k_s1k, `-o` scratchpad, `--plot em`.

---

## session-20260921-squash-for-push

**Status:** done, push pending (no credentials in this shell)
**Task:** Push out/ PNGs and new work but no scores.tsv/quartet_counts.tsv, without the old commits.
**Change:** squashed the 6 local commits since `52c1697` into `82fe225` (tree identical to old HEAD, 0 scores/quartet tsv). Old history kept on local branch `backup-all-commits` (contains the tsv blobs; don't push it).

---

## session-20260920-partial-push

**Status:** blocked on credentials
**Task:** Push unpushed commits except `13efc4a` (second-last).
**State:** built `c019ff5` (b8e4d61 rebased onto 7c01e37, session.md conflict resolved) on local branch `push-skip-13efc4a`. `git push origin push-skip-13efc4a:master` failed: no GitHub credentials in this shell. master untouched.

---

## session-20260920-gitignore-cleanup

**Status:** done
**Task:** Trim `.gitignore` to relevant entries.
**Change:** 219 -> 26 lines; dropped unused Django/Flask/Scrapy/Sphinx/poetry/pdm/etc. template blocks. Kept python, notebook, env, data/log (`store`, `connection_dir/`, `logs/`, `results*.txt`), Claude entries. Ignored set in `git status --ignored` unchanged.

---

## session-20260919-cra-plot-run

**Status:** done
**Task:** `cra.plot(<path>, metrics)` run-specific plot in `bench/caster.ipynb`, with gt/fitted transitions.
**Change:** `bench/utils.py` `collect_out_runs` (runs.tsv-shaped table from `out/**/report.tsv`) + `CrossRunAnalysis._plot_run`; signature now `plot(axes, run=None, metrics=..., logy=)` -- `axes` still the config dict, filters the run's reports via path-derived flags (`_recorded_from_report_path`/`_filter_out_runs`; underivable flags like `--np` skipped, named in title). `benchmark.parse_report` now parses `(Null|Alt row)` lines into `gt_/viterbi_transition_*` (+ `transition_*`). Backfilled 34 `out/` reports with GT/Viterbi rows (2 single-line ones converted).
**Update:** `collect_out_runs` also merges gt_stats.txt (`null_/alt_/pooled_` mean/var/cov, store names) so the notebook's window-means and variance/covariance sections are now `cra.plot(RUN_AXES, RUN, ...)` calls; within-window site variance (w1_s1 per-site) left as is.
**Update 2026-09-20:** `_plot_run` x-axis now log-scaled by window magnitude (was evenly spaced by rank); hollow markers = step != window. Rendered on `out/10X/down/N276/37-62` (scratchpad).
**Update 2026-09-20 (titles):** em.png/correlations.png/topologies_3d.png/states.png titles now show the output dir after `out/` (`get_title_locus` in phlag.py; falls back to `get_locus_description` if not under `out/`); `_plot_run` suptitle strips `out/` too.
**Next:** none open.

---

## session-20260919-10x-n276-transitions

**Status:** done (investigation only, no code changes)
**Task:** Why do transitions "explode" in `out/10X/down/N276/37-62`, at which EM step, and does it change with window size.
**Finding:** Per-step replay (scratchpad, `-o` redirected) shows the jump is at EM steps ~4-15 (outer iter 1-2), fixed point by ~30. Onset (p01>0.1) moves later with window: w50/w100 step 5, w500 6, w1k 9, w5k 14, w10k_s8k 26; none at non-overlapping >=10k. Final p01: 0.54 (w50) -> 0.49 (w500) -> 0.22 (w1k) -> 0.29 (w5k) -> 0.003 (w10k). Also bad at overlapping windows and at <=30 windows.
**Cause:** emissions split first (state 0 -> high mean/var outlier state), transitions follow ~2 steps later. GT-labeled solution has *lower* LL than the fitted one for every size <10k (w500: -207298 vs -202040); at w10k they match. So it's Gaussian misspecification on heavy-tailed windows, not an optimizer bug.
**Next:** none open.

---

## session-20260919-out-layout

**Status:** done
**Task:** Restructure ad-hoc `out/` to mirror store: category/subcategory first, then caster params, then phlag params; migrate existing files.
**Code:** `adhoc_scores_path` ([phlag/caster.py](phlag/caster.py)); `get_default_out_dir` + `get_phlag_param_segments`/`SIM_CATEGORY_PAIRS` ([phlag/phlag.py](phlag/phlag.py), [phlag/utils.py](phlag/utils.py)). Variants nest like store/caster; `-z` stays `_z`.
**Migration:** 416 files moved (category/pattern read from scores.tsv's source-FASTA column; phlag params parsed from each report.tsv's command line). `N109_z` -> `c25k_s25k_z/N109`; `N109_p` kept as its own node (differing gt_stats). Filenames unchanged (`<node>.tsv` legacy names kept). 3 stray top-level PNGs left.
**Update:** user asked node+pattern above the window/step dir; builders and tree re-migrated (419 files intact): `out/<cat>/<sub>/<node>/<pattern>/w<W>_s<S>[variant]/[<phlag segs>/]`. Size-dir regex in phlag.py now digit-strict.
**Next:** none open.

---

## session-20260919-bare-plot-flag

**Status:** done
**Task:** User: "--plot with no args plots everything."
**Root cause:** Both `phlag/caster.py` and `phlag/phlag.py`'s `--plot` (`nargs="*"`, `default=None`/`["em","states"]`) treat bare `--plot` (argparse gives `[]`, distinct from the flag being omitted, which gives `None`/the list default) as an empty selection -- every downstream `if args.plot and ...`/`"x" in args.plot` check is then falsy, so bare `--plot` silently plotted **nothing**, the opposite of the intended "no args = all" shorthand.
**Fix:** [phlag/caster.py:1993-1999](phlag/caster.py#L1993-L1999) and [phlag/phlag.py:2146-2153](phlag/phlag.py#L2146-L2153) -- added an `elif args.plot == []: args.plot = ALL_PLOTS` branch alongside the existing `is None` (omitted-flag) branch in each. Help text updated in both to document it. Left caster.py's separate omitted-flag bench-default (`[]`, i.e. no plots) untouched -- only the explicit bare-flag case changed.
**Verified for real** via `~/micromamba/envs/phlag/bin/python`, real `build_parser()`/`parse_arguments()` from both modules (no jax needed for this): confirmed argparse itself gives `None` (omitted) vs `[]` (bare `--plot`) vs the explicit list, and that the added normalization resolves bare `--plot` to every choice in both modules, in all `--bench` combinations for caster.py.
**Next:** none open. (Unrelated: `phlag/caster.py` has other uncommitted WIP in the working tree from a different session -- replacing the manual `--plot log` modifier with auto-detected log-scaling -- not touched here.)

---

## session-20260913-caster-multi-ws

**Status:** done
**Task:** User asked: when `-w`/`-s` get multiple space-separated values, run one caster command per combination instead of erroring.
**Implementation:** [phlag/caster.py](phlag/caster.py) `-w`/`-s` now `nargs="+"` (default `[50000]`/`[1000]`, `int_or_abbrev`/`step_size_or_fraction` applied per-value as before). `main()` unwraps to scalars for the existing single-run body unchanged; when either list has >1 value, loops the cartesian product (user's explicit choice over paired/zip), stripping `-w`/`-s` out of the original argv (new `_strip_ws_flags`, mirrors argparse's own nargs='+' consumption rule) and recursing into `main()` once per `(w, s)` pair with a single value each. Each combo already gets its own output path for free via the existing `w<...>_s<...>` path derivation, so no collisions in the normal (no `-o`) case.
**Caveat (not fixed, out of scope):** `-w`/`-s` must come before any positional FASTA path on the command line, or `nargs='+'` will swallow the positional as an extra window/step value (confirmed: `caster file.fa -w 10 100 -s 5` works, `caster -w 10 100 -s 5 file.fa` errors). Matches this codebase's existing convention (`bench/phlagster.py`'s own caster invocation always puts the positional first); `--plot` already has the same nargs='*' gotcha, unaddressed the same way. Also: passing an explicit single `-o PATH` together with multiple `-w`/`-s` values makes every combo overwrite the same file (last one wins) -- not guarded against, since `-o` is meant for single ad-hoc runs.
**Verified for real** via `~/micromamba/envs/phlag/bin/python`, tiny synthetic 4-seq FASTA + hand-written mapping in scratchpad: `-w 10 20 -s 10` ran both combos to completion (exit 0), confirmed via `--help` usage string too. Caught and fixed my own mistake mid-verification: the run (no `-o` passed) landed in the real `out/w10_s10/tiny/` and `out/w20_s10/` (should have redirected via `-o`, see [[feedback_test_output_scratchpad]]) -- cleaned up immediately (`out/w10_s10/N276` untouched, `out/w20_s10` fully removed since it didn't exist before).
**Next:** none open.

**Update 2026-09-13 (found via real usage, then corrected by user):** User's own command `-s 1.0` hit `caster: error: argument -s: invalid step_size_or_fraction value: '1.0'`. First fix attempt (integral-float special-case) was wrong -- user clarified the actual intended rule: ratio-vs-literal should be decided **lexically** (does the string have a decimal point?), not by numeric range. Re-fixed [phlag/caster.py:36-49](phlag/caster.py#L36-L49): `step_size_or_fraction` now returns a `float` (ratio of -w, multiplied out later in `main()`'s existing `isinstance(..., float)` branch, unrestricted to `(0,1)` -- `1.0` -> non-overlapping, `2.0` -> window-sized gap) whenever the string contains `.`, and an `int`/`int_or_abbrev` (literal step) otherwise, regardless of `k`/`m` suffix. Verified via `~/micromamba/envs/phlag/bin/python`: unit-level (`1.0`->`1.0` float, `1000`->`1000` int, `1k`->`1000` int) and end-to-end (`-w 20 -s 1.0` against a synthetic FASTA in scratchpad, `-o` redirected -- correctly ran D* with `window=20, step=20`). User's original repro command now parses and resolves as intended.

---

## session-20260913-phlag-multi-ws

**Status:** done
**Task:** User asked to add `-w`/`-s` to `phlag/phlag.py` (which currently has no `-w` at all, and `-s` only as an otherwise-unused-downstream plotting placeholder -- window/step is normally baked into the input `caster_scores` path, not passed to phlag directly): given batch `-w`/`-s` values (same cartesian nargs='+' machinery as [[project_caster_multi_ws]]), substitute them into `caster_scores`' `w<...>_s<...>`/`c<...>_s<...>` path segment to locate sibling scores.tsv files (already produced by caster for other window/step combos, same node/pattern) and run phlag once per combo.
**Implementation:** New `substitute_ws_in_path(path, new_val, new_step)` in [phlag/caster.py](phlag/caster.py) (right after `parse_ws_from_path`, its inverse) -- regex-replaces just the two numeric values in the first `w`/`c`-prefixed size segment (via `format_val`, so it names directories exactly like caster.py would), leaving the mode letter and any trailing suffix (`_site`/`_z`/`_i`/`_n`/`_norm-eps`, or nested `site`/`ilr`/`normalize` dirs) untouched. `phlag/phlag.py`'s `-w` (new, `nargs="+"`, default `None`) and `-s` (now also `nargs="+"`) both plain `int_or_abbrev` (no ratio-fraction semantics like caster's `-s` -- these must match an existing directory's literal number). `main()` split into `_run_single(args)` (the original body, now returns `phlag.output_file`) and a new batch wrapper: if `-w` and/or `-s` was passed, calls `parse_ws_from_path` on the resolved `caster_scores`, defaults whichever of window/step wasn't explicitly given to its value from that same path (so passing only `-w` keeps the original step, and vice versa), loops `itertools.product`, `copy.copy(args)` per combo (mutating `.caster_scores`/`.window_size`/`.step_size` on the copy -- no argv re-parsing needed, unlike caster.py's recursion, since `Phlag.__init__` derives everything else from `args.caster_scores` fresh), and `sys.exit`s if a substituted path doesn't exist. Same `-o`-collides-across-combos caveat as caster.py's own feature (not fixed, by design).
**Verified for real** via `~/micromamba/envs/phlag/bin/python`: `substitute_ws_in_path` against three real path shapes (flat `w10_s10`, canonical nested `.../w5k_s5k/ilr/...`, flat `--site` `c25k_s25k_site`) -- all substituted correctly. End-to-end: two synthetic 6-row scores.tsv siblings in scratchpad (`w10_s10/testnode`, `w20_s10/testnode`) via `phlag <path> -w 10 20 -s 10 --plot` -- both combos ran to completion (exit 0), each `Saved PHLAG output report to` its own correctly-derived path. Caught and cleaned up my own mistake again mid-verification (same lesson as [[feedback_test_output_scratchpad]], sharper this time): phlag's ad-hoc (`--bench`-less) output path is *always* rebuilt against the real repo `out/` tree from the node-name+window/step segments -- it does NOT mirror wherever the input scores.tsv itself lives, so even scratchpad-only inputs land real output in `out/w10_s10/testnode/` and `out/w20_s10/` (a `-o` cannot fix this for a multi-combo batch either, since one `-o` path would collide across combos). Used a `testnode` name that can't collide with real sim nodes, then fully removed both afterward (`out/w10_s10/N276` untouched, `out/w20_s10` -- which didn't exist before -- removed entirely).
**Next:** none open.

**Update 2026-09-13 (found via user's real usage, corrected):** Two bugs surfaced running `phlag out/w10_s10/N276/N276.tsv -w 10k 1k 100 10 -s 1.0`: (1) `-s 1.0` errored (`invalid int_or_abbrev value`) since `-s`'s type was plain `int_or_abbrev`, not ratio-aware; (2) `-w 10k 1k 100 10` alone (no `-s`) defaulted every combo's step to N276's *original literal* step (10), producing nonexistent paths like `w10k_s10` (real sweeps use step==window per size, e.g. `w10k_s10k`). User confirmed the fix: reuse caster.py's `step_size_or_fraction` for `-s` (decimal point -> ratio of *that combo's* `-w`, resolved per-combo, e.g. `1.0` -> non-overlapping) instead of `int_or_abbrev`, and change the omitted-`-s` default from "reuse `caster_scores`' own literal step" to ratio `1.0` (non-overlapping) for every `-w` value. `phlag/phlag.py`'s `-s` now `type=step_size_or_fraction` (imported top-level from `.caster`); `main()`'s batch block computes `step = s if isinstance(s, int) else max(1, round(s * w))` per combo (mirrors caster.py's own resolution) before calling `substitute_ws_in_path`. Verified via `~/micromamba/envs/phlag/bin/python`: `-s 1.0` now parses; replayed the exact batch loop (without running the heavy HMM, to avoid overwriting N276's real reports) confirming all 4 of the user's `-w` values now resolve to real existing files (`w10k_s10k`, `w1k_s1k`, `w100_s100`, `w10_s10`) instead of erroring.

**Status:** done
**Task:** User noticed `bench/aggregate.ipynb` violin plots of `em_hd`/`em_gt_hd` (pinned to a fixed [0,1] y-axis per `_BOUNDED_METRICS`) looked cut off near the boundary; asked to verify squared Hellinger distances are actually bounded to [0,1].
**Root cause found:** `phlag/hmm.py`'s `gaussian_hellinger2` (JAX, used for both `em_hd` via `PhlagHMMEmissions.em_divergence` and the `em_gt_hd` fallback path in `phlag/phlag.py:1078`) computed `1.0 - bc` with no floor — unlike its numpy sibling `gaussian_hellinger2_nd` in `phlag/utils.py`, which clamps via `max(1.0 - bc, 0.0)`. Since `bc = exp(log_coef - 0.125*quad)` can numerically exceed 1 for near-singular/near-identical covariances, this let squared Hellinger go slightly negative — the docstring's "bounded to [0, 1]" claim wasn't actually enforced.
**Confirmed in production data:** scanned all 330 `runs.tsv` under `store/phlag` (420k `em_hd`/`em_gt_hd` values) — 0 values above 1, but 2 negative `em_gt_hd` values (min -0.112) in `gaussian/w1k_s1k/normalize/runs.tsv`, consistent with the known [[project_gt_stats_normalize_blowup]] near-zero-divisor covariance blowup at small windows.
**Fix applied:** [phlag/hmm.py:180-182](phlag/hmm.py#L180-L182), `gaussian_hellinger2` now clamps via `jnp.maximum(1.0 - bc, 0.0)`, matching `gaussian_hellinger2_nd`. Verified with `~/micromamba/envs/phlag/bin/python` (bc=1.0 exact case -> h2=0.0, not negative; well-separated case still -> h2=1.0).
**Not done (out of scope/needs a run):** the 2 already-written bad rows in `w1k_s1k/normalize` are stale report values, not re-derivable without re-running phlag on that run — left for the user to resummarize if wanted. The violin plots' visual "cut off at 1.0" for small-window `em_hd` (well-separated states clustering near 1.0) is separately just standard KDE overshoot against the fixed axis (`_draw_stat`'s violin body's own docstring already covers this) — not a bug.
**Next:** none open.

---

## session-20260908-sums-plot

**Status:** done
**Task:** New `sums.png` (`plot_sums`/`--plot sums` in `phlag/caster.py`): single log-y histogram of real per-bin counts (`bins='auto'`) of row_sum=ABBA+BABA+AABB, each bar split into 3 stacked segments by that bin's topology composition (negative bin-means clamped to 0 for the split). `dist.png` (`plot_distribution`) got the matching treatment: log-y + x_grid/xlim clipped to each column's [1,99] percentile (not full min/max), plus a 5-decade y-floor below the tallest bar so a well-separated Null/Alt (one fit curve underflowing under the other's peak) can't blow out the axis. Both now read straight-line-vs-curved decay on log-y as exponential-vs-Gaussian shape, robust to variance/outliers.
**Fixed along the way:** dstar.cpp's window/step path computed per-site-average `c*ABBA` etc. (the separate "dstar window average site-count fix") but never wrote the `SiteCnt` total it divided by — added to `results`/output header. `plot_sums` no longer needs `SiteCnt` at all after the redesign above (dropped the earlier two-panel sites/windows split per user feedback).
**Verified for real** (not just syntax) via `~/micromamba/envs/phlag/bin/python`, scratchpad-only output, against w50k_s50k (clean), w50_s50 (120k rows, heavy-tailed — now shows a legible sharp exponential-ish peak instead of blank), and a real Null/Alt file (`store/caster/w50k_s40k/.../37-62/scores.tsv`, H²=0.93) for dist.png.
**Follow-up bug (user caught by eye):** `--dist-type exp`'s fit line looked flat/non-decaying even after the log-y+percentile fix. Root cause: `shift = vals.min()` (and the joint-H² version, `shift_all = Y_all.min(axis=0)`) used the column's true min, not a percentile — one extreme outlier (verified on `out/w10_s10/N276/N276.tsv`: std≈303 but one row at -10173) inflates `scale` to ~30x the visible window, so the curve is technically decaying but imperceptibly slowly on screen. Fixed both to use the same 1st-percentile floor `x_grid` already clips to (`p_lo`), clipping any point still below it to 0 so `expon.pdf`'s x>=shift domain stays valid. Re-verified on the same file with `-locus-pattern 37-62` (recovered from the source FASTA path baked into the tsv's `file` column, since N276 itself has none) — fit now shows a real visible log-linear decay matching the histogram's right tail; also newly visible: the data is peaked/two-sided, not truly one-sided-exponential, which a Q-Q plot (discussed, not yet built) would diagnose better than this histogram+fit approach.
**Made log-y opt-in:** added `log` to `--plot` (choices list + `plot_choices`/`is_plot_only_argv`, all 6 `CasterPlotter(plot_log=...)` call sites) as a pure modifier -- not in the OR-gate checks or the all-of-the-above default, so it only does something when combined with `dist`/`sums`, and is linear (original behavior) unless explicitly requested. `CasterPlotter.__init__` stores `self.log_scale`; `plot_distribution`/`_stacked_sum_hist` gate `set_yscale('log')` (+dist.png's 5-decade y-floor) behind it, while the percentile-clipped x-view stays unconditional in both. Verified both `plot_log=True` and default (linear) render correctly on the same file.
**Added `dexp` (double-exponential/Laplace) as a third `-d`/`--dist-type`:** new `laplace_hellinger2_nd` in `phlag/utils.py` -- closed-form per-dimension Bhattacharyya coefficient between two Laplace(loc,scale), derived by hand and verified against `scipy.integrate.quad` to ~1e-8 or better (incl. same-scale degenerate case, multi-dim product form, zero-scale NaN guard). Wired into `plot_distribution` (per-topology fit via `scipy.stats.laplace.fit`, no shift/clip needed since Laplace's support is all of R, unlike `exp`'s one-sided cutoff) and `write_ground_truth_stats` (gt_stats.txt's Hellinger2). Verified on `out/w10_s10/N276/N276.tsv`: dexp's "tent" shape (two straight log-y lines meeting at the peak) matches this data's actual peaked-two-sided histogram far better than `exp` ever could, H²=0.011 vs exp/gaussian's ~0.00002.
**Also fixed while touching this code:** `write_ground_truth_stats`'s `exp` branch had the *same* raw-min outlier-fragility bug as the per-topology plot fix from earlier this session (`Y.min(axis=0)` instead of a percentile floor) -- fixed identically (1st-percentile + clip). gt_stats.txt's exp Hellinger2 for `w10_s10/N276` went from a bogus 1.6e-7 to a real 2.4e-5, consistent with the plot-side fix.
**Next:** none open; Q-Q plot alternative offered but not requested yet.

---

## session-20260830-heatmap-overlap

**Status:** active
**Task:** Fix overlapping labels in CrossRunAnalysis heatmap plots (`bench/utils.py`, `_plot_heatmap`-family methods around line 1400-1550).
**Progress:**
- Root cause found: every subplot in the row×col grid was calling `ax.set_yticklabels(y_vals, ...)` unconditionally, so panel N+1's y-axis bin-range labels (e.g. `(0.833,1.000]`) rendered on top of panel N's rightmost data cells once subplots were packed tightly (small `wspace`).
- Fix applied: `bench/utils.py:1493-1502` now only sets real y-tick labels when `ci == 0` (leftmost column in each row), matching the existing `ci == 0` gate already used for the row `ylabel`. Other columns get `ax.set_yticklabels([])`.
**Blocker:** none.
**Next:** user to re-render the cross-run heatmap and confirm the overlap is gone.

**Update 2026-08-31:** Fixed `_plot_tpr_fpr_grouped`'s (single-panel TPR/FPR comparison) legend overlap, twice:
1. First pass reserved figure height dynamically instead of a fixed `bbox_to_anchor` y=0.97 — fixed vertical clipping into the suptitle, but user reported still overlapping.
2. Root cause was actually horizontal: the two legends sit in opposite corners (upper-left/upper-right), which only works on a wide multi-column figure — this plot is single-panel (~3.3in wide), so any real legend text collides in the middle. Reworked to stack both legends in the same corner (`bench/utils.py:806-855`), placed via real `get_window_extent` measurement (same technique as this file's heatmap colorbar) rather than an estimated offset, so each legend and the axes below land exactly where the previous one's rendered edge ends.
Could not render-test locally (no matplotlib in this shell, same limitation as the jax gap noted elsewhere) — syntax-checked only. Awaiting user's re-render to confirm.

**Incident:** While chasing a separate `json.dump`-reformatting mistake on `bench/cross-run.ipynb`, ran `git checkout -- bench/cross-run.ipynb` without checking `git status` first. That file already had uncommitted user edits (rho/beta being iterated on live in VSCode's Jupyter editor, autosaving) — the checkout discarded them, reverting to last commit (`b0c28fa`). Not recoverable via git (never committed/stashed); no filesystem history recovery on this remote. User's only path back is VSCode's own local-history/undo buffer. User was informed; memory note saved (`feedback_check_status_before_checkout`) to always check status/diff before any file-targeted checkout/reset, especially IDE-open/autosaving files.

---

## session-20260831-report-path-bug

**Status:** active (paused for live jobs to finish)
**Task:** Root-caused benchmark failures logged as `[fail] ... phlagster reported success but report file not found` in `logs/w1k_s1k_ilr_rho0.9_beta4.0.log`.
**Root cause:** `phlag/phlag.py`'s `get_default_out_dir()` (`--bench` branch) copied the caster scores path's `site`/`ilr`/`normalize` variant-nesting segment straight into the report output dir, duplicating it on top of an `--output-base` that already encodes that variant (e.g. `.../ilr/rho0.9_beta4.0/reports`). Every `--bench` run using `-i`/`--site`/`-n` wrote reports one level too deep (`reports/ilr/...` instead of `reports/...`), so `benchmark.py`'s post-hoc existence check (`get_expected_report_path`, which never expected the extra segment) always reported them as missing even though phlag succeeded.
**Fix applied:** [phlag/phlag.py:604-622](phlag/phlag.py#L604-L622) now strips a leading `site`/`ilr`/`normalize` marker from the segments pulled out of the scores path before building the report dir. Verified the corrected path logic in isolation (no jax in this shell to run the module directly).
**Migration:** moved all pre-existing misplaced report trees (30 dirs, ~17k files) under `/drive2/iang/phlag/**/reports/{ilr,site,normalize}/` up one level; no collisions.
**Blocker:** 26 live `bench/benchmark.sh` processes are running from a frozen pre-fix source snapshot (`create_source_snapshot()`) and are actively re-creating small `reports/ilr`/`reports/site` marker dirs as they write new reports — confirmed in `ps aux`, e.g. jobs targeting `gaussian/c25k_s25k/{pair,site}/{ilr,zscale}/rho0.9_beta4.0(/repulsion/annealing/lam1.5)?`. User chose to let them finish rather than kill/restart.
**Next:** once those jobs finish, re-run the same migration sweep (`find ... -path "*/reports/ilr" -o -path "*/reports/site" -o -path "*/reports/normalize"`, move contents up one level, rmdir) to flatten whatever they wrote under the old buggy paths. `phlag/phlag.py` fix is uncommitted in the working tree.

---

## session-20260831-caster-score-analysis

**Status:** done
**Task:** Add a "Caster scores analysis" section to `bench/cross-run.ipynb` — given a run root dir (e.g. `store/caster/w5k_s5k`, which directly holds `10X`/`admixture`/`recombination` category dirs), report mean/std of the raw caster topology score columns across every `scores.tsv` under it.
**Implementation:** New markdown+code cells inserted after the existing "## CASTER" runs.tsv cell (now cells 3-4). `caster_score_stats(root_dir)` walks `root_dir.rglob("scores.tsv")`, reuses `phlag.caster.CasterPlotter.resolve_topology_columns` (same column-detection logic phlag's own plotting uses) to pick the score columns (`c*ABBA/c*BABA/c*AABB` or ILR `c*ILR1/c*ILR2`) regardless of variant, concats, and reports `.agg(["mean","std"])`.
**Gotcha caught before landing:** the canonical store tree nests `site`/`ilr`/`normalize` as *sibling* dirs of the category dirs at the same level (confirmed via `phlag/caster.py:78-96`'s `parse_ws_from_path` and `phlag/phlag.py:620`), not underneath them — so an unfiltered rglob from a base window/step dir silently mixes raw and ILR-transformed columns via pandas' NaN-padded concat. Fixed by excluding any relative path containing a `site`/`ilr`/`normalize` component (unless that's the root itself), so a caller pointed at a variant dir still gets that variant's own consistent columns.
**Verified:** ran standalone (outside the notebook, using the `phlag` mamba env — this shell has no pandas) against `store/caster/w5k_s5k` (672 files/806400 rows, raw ABBA/BABA/AABB only) and `store/caster/w5k_s5k/ilr` (672 files/806400 rows, ILR1/ILR2 only); ~5s runtime. Notebook JSON validated with `nbformat.validate`.
**Next:** none — done pending user's own re-render/scan of other directories.

---

## session-20260919-transition-matrix-report

**Status:** done
**Task:** Report empirical transition matrices next to the fitted one.
**Change:** `phlag.py` writes `Ground truth transition matrix` and `Viterbi path transition matrix` (2x2, rows divided by each state's own window count, Null/Alt order) above the Initial/Final matrix lines in report.tsv. `caster.py` writes `TransitionMatrix` to gt_stats.txt; `utils.py` read/write handle it.
**Verified:** synthetic gt_stats round-trip; phlag on `out/10X/up/N635/37-62/w1k_s1k/N635.tsv` (scratch output).
**Note:** `out/` was reorganized to `out/10X/up/<node>/<pattern>/w<..>_s<..>/`; not wired into `bench/benchmark.py` parser.

---

## session-20260919-transition-rows-5sig

**Status:** done
**Task:** Transition-matrix report lines: Null and Alt rows on separate lines, 5 significant digits.
**Change:** `phlag.py` report headers (Ground truth / Viterbi path / Initial / Final) now one `(Null row)` and one `(Alt row)` line each via `_append_transition_rows` (`:.5g`). `gt_stats.txt` TransitionMatrix left full-precision (parsed back).
**Verified:** phlag on `out/10X/down/N276/37-62/w1k_s1k/scores.tsv` with `-o` in scratchpad.

---

## session-20260919-drop-histogram

**Status:** done
**Task:** User: don't print flagged-position buckets in terminal regardless of `--agg`.
**Change:** `phlag.py` removed the `generate_histogram` call, method, and the now-unused `--agg` flag (no other references in repo).

---

## session-20260919-agg-histogram-plot

**Status:** done
**Task:** `--agg` pools flagged windows into a histogram panel in states.png, nothing in terminal.
**Change:** `phlag.py` `--agg` restored (default None = no panel; N = bucket of N windows). With `--plot states --agg N`, states.png gets a second panel: Path 1 Alt-window counts per bucket, ground truth overlaid. Title moved to `ax1.set_title`.
**Verified:** `out/10X/down/N276/37-62/w1k_s1k/scores.tsv --agg 20` with `-o` in scratchpad.

---

## session-20260919-rename-scores

**Status:** done
**Task:** Rename legacy `out/**/<node>.tsv` caster scores files to `scores.tsv`.
**Change:** 44 renamed (header + stem==node + source-FASTA column checked). Skipped: `out/10X/down/N109_p/37-62/c25k_s25k/N109.tsv` (stem N109 != node N109_p), `out/10X/down/N276/37-62/w10k_s10k/N276.tsv` (scores.tsv exists, cmp identical, left for user to delete). No code changes needed: `get_default_out_dir` derives from path parts, stem only a fallback.
**Verified:** `get_default_out_dir` identical before/after for all 46 paths; phlag on renamed w50k_s50k with `-o` scratchpad ran, ROC-AUC 0.999. Pre-existing: report `Clade: N/A` (`get_simulation_node_name` uses parts[-3] = pattern in new out/ layout).

---

## session-20260920-viterbi-region-matrices

**Status:** done
**Task:** Null/Alt-region + pooled transition matrices in report.tsv.
**Change:** `phlag.py` adds Viterbi-path matrices counted only over window pairs inside GT Null / Alt region; existing Viterbi line relabeled `(pooled)`. GT-label matrices unchanged (region-split would be trivial).
**Verified:** N635 w1k_s1k, `-o` scratchpad.

---

## session-20260921-transition-matrix-points

**Status:** done
**Task:** One transition matrix per report line; states.png as points.
**Change:** `phlag.py` `_append_transition_rows` writes `label: [[a, b], [c, d]]` (was separate Null/Alt row lines). states.png: Viterbi paths plotted as dots, ground truth as x (under paths), no step interpolation. `bench/benchmark.py` `_RE_TRANSITION_MATRIX` now parses GT/Viterbi (pooled)/Final matrix lines (old row-form regex kept for archived reports).
**Verified:** w5k_s5k, `-o` scratchpad; parse_report yields gt_/viterbi_/transition_ keys.

---

## session-20260921-em-window-counts

**Status:** done
**Task:** em.png: print Alt/Null window counts left of legend.
**Change:** `phlag.py` `_finalize_legend` returns the legend; new `PhlagPlotter._annotate_state_counts` writes `$N_a$ (Alt)` / `$N_b$ (Null)` for ground truth (if available) and After EM (flip-aware), right-aligned at the legend's left edge.
**Verified:** w5k_s5k admixture, `-o` scratchpad.

---

## session-20260922-fisher-skewness

**Status:** done
**Task:** Compute Fisher's skewness metric, output in gt_stats and report.
**Change:** `caster.py` `write_ground_truth_stats` computes per-topology (ABBA/BABA/AABB) Fisher-Pearson skewness (`scipy.stats.skew`, `bias=True`, g1=m3/m2^1.5) for Null/Alt/Overall, stored as `"<label>Skewness"` in the stats dict. `utils.py`: `write_gt_stats_file` writes `"<label> skewness: [...]"` lines after each label's covariance norm; `read_gt_stats_file` parses them back as `result["<label>Skewness"]`; `collect_gt_stats` adds `<region>_skew_<topo>` columns. `phlag.py`'s `compute_output` reads gt_stats.txt's skewness (same "prefer caster's file" convention as em_gt_hd) and appends `"<label> skewness (ABBA, BABA, AABB): [...]"` lines to report.tsv, gated on `self.ground_truth_fits` (same block as the other gt_stats-derived lines).
**Verified:** real run on `out/10X/down/N276/37-62/w1k_s1k/scores.tsv` copied to scratchpad (`-o` scratchpad, not touching `out/`) — gt_stats.txt round-trips through `read_gt_stats_file`, `collect_gt_stats` produces the 9 skew columns, report.tsv shows matching skewness lines.
**Next:** not wired into `bench/benchmark.py`'s `RunRecord`/runs.tsv (only gt_stats.txt + report.tsv were asked for) — flag if runs.tsv-level skewness columns are wanted later.

---

## session-20260921-em-rows-swap

**Status:** done
**Task:** em.png rows: top = KDEs, bottom = fits; GT dashed/light, EM solid/dark.
**Change:** `phlag.py` `_plot_kde_row` (GT KDE `colors[..]['kde']` dashed + HMM-assigned KDE `['line']` solid) and `_plot_fit_row`/`_draw_fit` (GT or GMM-seed fit dashed/light, EM fit solid/dark; GT labels at lower y). Old `_plot_ground_truth_row`/`_plot_gmm_init_row`/`_plot_em_row` removed. Legend 2 columns in em.png. Fit-row μ/σ labels and mean/±σ lines removed.
**Verified:** w5k_s5k gaussian, `-o` scratchpad; gmm path not run.

---

## session-20260922-em-autolog

**Status:** done
**Task:** em.png: auto log scale, same heuristic as scatter.png.
**Change:** `phlag.py` new `PhlagPlotter._apply_log_scales`, called after both rows are drawn in `plot_distributions`. Reuses `CasterPlotter._needs_log_scale` (imported from `.caster`) independently per axis per topology column: x off raw `Y[:,d]` scores (shared by both rows), y off the actual drawn line y-data (KDE + fit curves) in that column. Symlog (linthresh=1e-9, matching scatter's NOISE_FLOOR) when negatives present, else plain log.
**Verified:** N564 recombination/down w1k_s1k gaussian, `-o` scratchpad, `~/micromamba/envs/phlag/bin/python`; em.png now shows y in log (density spans 1e-5 to 1e2) and x in symlog (scores straddle 0).
**Update:** `LOG_SCALE_RATIO_THRESHOLD=1000.0` (10x `_needs_log_scale`'s 100x default) for em.png's x/y calls only, other callers untouched -- default was flagging nearly every column, cluttering symlog ticks. New `_separate_overlapping_lines` (called per-axis after both rows draw, before log scaling): when a dashed (GT/seed) line coincides with its solid (EM) counterpart within 0.1% relative, multiplicatively nudges the dashed one's y by 2% so both colors stay visible (additive would've been invisible at the peak or swamped the tails on log-y).
**Verified:** N564 w1k_s1k (linear x now, threshold fix confirmed) and N635 10X/up w5k_s5k (ROC-AUC=1, near-perfect GT/EM overlap) -- cropped AABB Fit panel shows light-blue/orange dashed GT lines as distinct thin traces beside the solid EM fits instead of fully hidden.
**Bugfix (user report: "blue dashed line not showing up" on real out/10X/down/N276/37-62/w1k_s1k/gaussian/em.png):** two bugs. (1) `_separate_overlapping_lines` only compared dashed-vs-solid pairs -- missed Null GT Fit (light blue dashed) fully hidden under Alt GT Fit (orange dashed, drawn second) when both ground-truth fits land on near-identical mu/sigma for a topology; now compares every line pair and nudges whichever was drawn earlier (lower zorder). (2) coincidence was judged by linear relative difference (`rel_tol=1e-3` of peak) and the offset was `value * (1+0.02)` -- both wrong for a 50+-decade log axis: debug-verified real reldiff for the hidden pair was 1-3% (10-30x over tol, so it never even triggered), and even forcing it, a 2% *value*-relative shift is ~0.01 decades, sub-pixel against a ~56-decade span. Fixed: coincidence on a log/symlog axis is now judged by max decade gap (`decade_tol=1.0`, i.e. curves within 1 order of magnitude everywhere -- linear rel_tol kept only for non-log axes), and the offset is sized as `offset_frac` (2%) of the axis' own visible decade span (or linear span), so it's the same few-pixel gap regardless of how many decades the panel covers. Order fixed too: `_separate_overlapping_lines` must run after `_apply_log_scales` now (needs the final scale to size the offset).
**Verified:** debug-printed real reldiff/decdiff for all 6 line pairs across N276 w1k_s1k's 3 Fit panels before fixing (confirmed the miss: ABBA/BABA Null-vs-Alt GT pairs at decdiff 0.2-0.4 decades, everything else 2.9-55 decades); after the fix, re-ran against the same real input (`-o` scratchpad) -- Null GT Fit now visibly separated from Alt GT Fit in all 3 Fit panels, and the N635 AABB re-check still shows clean separation (no regression).

---

## session-20260924-report-tsv-path-config

**Status:** done
**Task:** User: phlag given a report.tsv should take its config from the path, not the recorded first line.
**Change:** `phlag.py` `parse_arguments`'s relative-report.tsv branch (supersedes session-20260923-report-tsv-replay's first-line replay): walks back to the last `gaussian`/`gmm` segment → positional `<prefix>/scores.tsv` + `-d <dist>`, then maps trailing segments `rho<X>_beta<Y>`/`var2x`/`repulsion`/`annealing`/`lam<X>` → `--rho/--beta`/`--double-variance-init`/`--ap repulsion`/`--annealing`/`--lam`; unknown segment → `parser.error`. This run's own flags still appended last and win.
**Verified:** parse-only on `out/10X/up/N281/45-55/w500_s500/gaussian/rho0.75_beta0.01/report.tsv` — resolves to that dir's scores.tsv, rho 0.75/beta 0.01, `get_default_out_dir` lands back on the same report dir; with `--rho 0.9 --beta 4.0` override lands on `rho0.9_beta4.0`. No full phlag run.

---

## session-20260924-em-symlog-linthresh

**Status:** done
**Task:** User: em.png ABBA/BABA x-axis wastes width on the flat central density plateau; make the tails (Alt signal) visible.
**Change:** `phlag.py` `_apply_log_scales`: x-axis symlog `linthresh` now data-driven — 90th percentile of |x| (`X_LINTHRESH_PERCENTILE`), rounded up to a power of 10, floor 1e-9 — instead of fixed 1e-9 (~56% of ABBA/BABA values were within 1e-6 of 0, so ~14 decades showed a flat KDE). Y-axis unchanged.
**Verified:** N281 45-55 w500_s500 scores.tsv copied to scratchpad and rendered with `--plot em`; center collapsed to one linear band (linthresh 1e-3), tails fill most of the width, and the ticks are clean.

---

## session-20260926-cra-plot-run-list

**Status:** done
**Task:** `cra.plot(run=[...])` accepts a list of run dirs (label = dir minus trailing node/pattern); aggregate.ipynb Transition Prior cell uses a hardcoded list of the 5 KDE branches (was referencing caster.ipynb's undefined `kde_branches`).
**Verified:** headless render of that cell's config over the 5 branches.
**Follow-up:** relative run paths that do not exist under the repo root now resolve under `out/msa` (`_resolve_run_path`). The 5 branches only have 45-55 w10k reports in out/msa; N109 37-62 only has old c25k dirs.
