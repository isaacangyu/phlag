# session.md

## 2026-10-10
- `bench/utils.py` `_plot_beta_line`: lines now split by `--dirichlet-mean` (read from each config's args.json); dirichlet-mean = darker shade of the hue color, others lighter; legend gains grey "dirichlet-mean"/"dirichlet-mode" entries. Only active when both states are present in the plot. Uncommitted.
- `~/.bash_aliases` `check`: cores now from per-PID utime+stime deltas (PIDs in both samples only, no more negatives); prints total/idle plus cores/processes/busy for phlag-caster-benchmark tree, my other, other users. `_cpu_ticks` replaced by `_cpu_snap`.
- `bench/utils.py`: `'phlag'` axis accepts a path relative to `store/phlag_orig` (e.g. `'expand-edges/beta5.0'`), registered as a windowless source via `_register_phlag_sources`; `_title_suffix` appends the phlag_orig-relative path(s) of non-paper configs. Default `phlag` agg sources fixed to gtrees+msa. Uncommitted.

## 2026-10-11
- Ad-hoc tree moved off `/`: `out/` -> `store/out/` (`/drive2/iang/out`, 8.0 GB, 21139 files, checksum-verified, old dir removed). `utils.get_out_root()` now `<repo>/store/out/msa`; help text in `phlag.py`/`caster.py` + CLAUDE.md updated. No `out` symlink left in the repo. 1118 report.tsv header lines still record the old `/home/iang/phlag/out/...` command path (provenance only, not parsed). Uncommitted.
