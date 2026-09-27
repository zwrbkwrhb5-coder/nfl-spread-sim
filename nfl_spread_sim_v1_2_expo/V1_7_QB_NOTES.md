# v1.7 QB Layer Notes

Current spread baseline:
- diff_off_epa
- diff_def_epa
- diff_off_success
- diff_def_success

Current totals baseline:
- totals-level offensive / defensive efficiency
- scoring levels
- opponent-adjusted EPA
- explosive pass rate
- turnover rate

QB features are added on top of these baselines and evaluated walk-forward.

College priors and first-game-back injury adjustments are intentionally NOT
included in this release. They should be separate ablation layers after the
NFL-only QB signal is measured.
