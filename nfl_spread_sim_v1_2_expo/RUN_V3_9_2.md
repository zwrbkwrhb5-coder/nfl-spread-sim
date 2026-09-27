# v3.9.2 Pandas dtype fix

This patch fixes:

```text
TypeError: Invalid value 'closing_candidate' for dtype 'float64'
```

Cause:

The future text columns in the forward-test CSV are blank when the ledger is
first created. After `pd.read_csv`, pandas can infer an all-blank column such
as `closing_snapshot_label` as `float64`. Newer pandas versions reject writing
a string into that float column.

v3.9.2 explicitly restores all known text fields after each ledger read.

It also protects later settlement fields such as:

```text
result = win / loss / push
```

from the same problem.

## Install

```bash
unzip -o nfl_spread_totals_v3_9_2_pandas_dtype_fix.zip
source .venv-model/bin/activate

python -m pytest v39_tests v391_tests v392_tests -q
```

## Retry the SAME close command

You do not need to recapture the qualifiers.

```bash
python -m score_model.forward_test_v39 close \
  --all-candidates-csv artifacts/live/week3_2026_closing_candidates_v39.csv \
  --game-id 2026_03_KC_MIA \
  --season 2026 \
  --week 3 \
  --snapshot-label closing_candidate
```

Then:

```bash
python -m score_model.forward_test_v39 report \
  --season 2026 \
  --week 3
```

The existing ledger remains the source of the original entry prices.
