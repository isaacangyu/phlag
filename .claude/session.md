# Session Sync

Shared status board for concurrent Claude sessions working in this repo. Each session keeps one section below, keyed by its own identifier, and updates it on every substantial turn.

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
