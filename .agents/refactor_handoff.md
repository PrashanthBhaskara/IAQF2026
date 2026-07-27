# Refactor contract

Keep this repository small, import-safe, and reproduction-first.

- Never edit or compile over `IAQF_Inefficient_Markets_2026.tex` or
  `IAQF_Inefficient_Markets_2026.pdf`. Their SHA-256 hashes are pinned in
  `iaqf/config.py`.
- Preserve exactly 11 raw Parquets, six processed Parquets, 36 table artifacts,
  and 14 paper PNGs.
- The PNG reference contract comes from Matplotlib 3.8.4. Exact hashes are
  tested only in the pinned reference environment.
- HAC and Granger use the history-restored complete-case return sample:
  `N=26,489` for both channels and reverse-Granger q-value `0.127898`.
- The submitted bootstrap `567 [216, 1,898]` is frozen provenance, not an
  executable result. The surviving moving-block implementation produces about
  `64 [46, 88]`; do not recreate or fabricate the missing sieve producer.
- Preserve Kraken's historical end boundary and all established missing-data
  choices. Scientific changes require a separate study revision.
- Keep only the two state classes `RepoPaths` and `AnalysisData`. Prefer plain
  functions, deletion, and the standard library; do not add framework or
  compatibility layers.
- Verify with `uv run ruff check .` and `uv run python -m pytest -q`.
