# Archived SOP experiment outputs

Historical runs kept unchanged for traceability. They were produced by earlier versions of the
routing contract and are not comparable with runs of the corrected release.

- `a38bf4b0/` — committed results at commit a38bf4b0: per-segment ten-feature cost, ANTROUTE early stop
  after 10 non-improving iterations, Cheng fixed 60 iterations, Shapiro-Wilk then t-test or Wilcoxon.
- `session-conv20/` — re-run on 2026-10-20: both algorithms stop after 20 non-improving iterations
  (N_conv), two-sided Wilcoxon on d = baseline − ANTROUTE with Holm (m = 4). Same per-segment cost
  contract as a38bf4b0; superseded by the corrected contract (version 2).
