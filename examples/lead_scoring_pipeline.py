"""
Real-project integration example: a lead-scoring pipeline.

This is the kind of script AK-SCI is built for -- a small freelance/
automation job that reads messy CSV exports (from a CRM, a form tool, a
spreadsheet someone hand-edited), scores each lead, and writes a ranked
CSV back out. Real-world input like this is never perfectly clean: some
rows are missing a column that got renamed upstream, some numeric fields
come through blank, and one row has a literal 0 that would blow up a
ratio calculation.

Two integration patterns are shown together:

  1. `resolver.install_global_hook()` -- a process-wide safety net. Any
     *uncaught* exception anywhere in the script gets AK-SCI's diagnosis
     printed before the normal traceback, so a hurried debugging session
     starts with a category and a suggested fix instead of just a stack
     trace.

  2. `@resolver.safe_run(..., auto_fix_context=...)` on small, single-
     purpose helpers -- a *local*, opt-in safety net around the exact
     pieces of code expected to hit messy data. Auto-fix replaces a
     function's entire return value for that call, so each helper here
     does exactly one risky thing (look up a column, read a numeric
     field, divide two numbers) rather than several -- see the note on
     `ErrorResolver.safe_run` for why that split matters.

Run with: python examples/lead_scoring_pipeline.py
No API key required -- this uses only the offline local model.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from aksci import ErrorResolver, UnifiedFrame

# --- 1. Set up the resolver and the process-wide safety net -----------------
#
# auto_fix=True turns on the whitelisted auto-fix rules (see
# aksci.error_handler.autofix) for any safe_run-decorated function that
# opts in with auto_fix_context. install_global_hook() is separate -- it
# only affects uncaught exceptions that would otherwise crash the script.
resolver = ErrorResolver(auto_fix=True)
resolver.install_global_hook()


# --- 2. Simulate a messy CRM export ------------------------------------------
#
# Real exports from tools like HubSpot, a Google Form, or a hand-edited
# spreadsheet routinely have this kind of mess: a renamed column, blank
# numeric fields, a stray zero. AK-SCI doesn't "fix" business logic -- it
# recovers from exactly this shape of low-stakes, well-understood
# formatting noise so the batch job finishes and flags what it changed.
RAW_LEADS_CSV = """lead_id,company,employe_count,monthly_visits,deals_closed,budget
1,Acme Textiles,120,4500,3,15000
2,Blue Ridge Traders,,3900,2,9000
3,Continental Freight,80,0,0,10000
4,Delta Analytics,45,2200,1,7500
5,Ember Retail,,,,
"""
# Row 1: clean.
# Row 2: "employe_count" is a typo'd column name (should be
#        "employee_count") -- fuzzy-matched at lookup time.
# Row 3: monthly_visits and deals_closed are both 0 -- a naive
#        deals-per-visit ratio would divide by zero.
# Row 4: clean.
# Row 5: almost entirely blank -- a stress test for the "give up
#        gracefully" path once a row has nothing left to recover.


def _lookup_column(row: pd.Series, all_columns: list[str], col: str) -> float:
    """Read one field off a row by name. The only risky thing this does
    is the `row[col]` lookup, so if `col` is renamed/mistyped upstream,
    fuzzy-matching it against `all_columns` is the whole fix."""
    return float(row[col])


@resolver.safe_run(
    reraise=False,
    auto_fix_context=lambda row, all_columns, col: {
        "available_columns": all_columns,
        "frame": row,
    },
)
def safe_lookup_column(row: pd.Series, all_columns: list[str], col: str):
    return _lookup_column(row, all_columns, col)


def _require_not_nan(value: float, field_name: str) -> float:
    """Reject a blank numeric field. `float(nan)` doesn't raise on its
    own -- a blank CSV cell is valid Python, just not valid *data* for
    this pipeline -- so this raises explicitly to give null_values_fillna
    something to catch and default."""
    if pd.isna(value):
        raise ValueError(f"Input contains NaN: {field_name!r} is blank")
    return value


@resolver.safe_run(
    reraise=False,
    auto_fix_context=lambda value, field_name: {
        "frame": pd.Series([value]),
        "fill_value": 0.0,
    },
)
def safe_require_not_nan(value: float, field_name: str):
    result = _require_not_nan(value, field_name)
    return result


def safe_get_field(row: pd.Series, all_columns: list[str], col: str) -> Optional[float]:
    """Look up a field and reject it if blank, composing the two
    single-purpose helpers above. Returns None only if a field is both
    unresolvable *and* unrecoverable -- e.g. row 5, where several
    columns are simultaneously blank."""
    value = safe_lookup_column(row, all_columns, col)
    if value is None:
        return None
    cleaned = safe_require_not_nan(value, col)
    if cleaned is None:
        return None
    # null_values_fillna returns a pandas Series (it fills whatever
    # `frame` it was given) -- unwrap the single value back out.
    return float(cleaned.iloc[0]) if hasattr(cleaned, "iloc") else float(cleaned)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


@resolver.safe_run(reraise=False, auto_fix_context=lambda n, d: {"default": 0.0})
def safe_ratio(numerator: float, denominator: float):
    return _ratio(numerator, denominator)


def score_lead(row: pd.Series, all_columns: list[str]) -> Optional[float]:
    """Compute a simple lead score: visits-to-deals ratio weighted by budget.

    Every field read and the division go through their own safe_run-
    wrapped helper, so one bad field (renamed column, blank value, a
    zero denominator) is recovered in place instead of discarding the
    whole row's computation.
    """
    employee_count = safe_get_field(row, all_columns, "employee_count")  # typo'd upstream -> fuzzy match
    visits = safe_get_field(row, all_columns, "monthly_visits")
    deals = safe_get_field(row, all_columns, "deals_closed")
    budget = safe_get_field(row, all_columns, "budget")

    if None in (employee_count, visits, deals, budget):
        return None  # a field was unrecoverable -- e.g. row 5, almost entirely blank

    engagement = safe_ratio(deals, visits)  # 0/0 on row 3 -> auto-fixed to 0.0
    return round(engagement * budget * (1 + employee_count / 100), 2)


def main() -> None:
    print("=" * 70)
    print("Lead-scoring pipeline -- AK-SCI integration example")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "leads_export.csv"
        csv_path.write_text(RAW_LEADS_CSV)

        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} leads from {csv_path.name}")
        print(df.to_string(index=False))

        # UnifiedFrame is the right tool once the data is clean -- e.g. to
        # standardize budget/visits before feeding them into a model. Here
        # it's used for a quick sanity check on the numeric columns.
        numeric_preview = UnifiedFrame(df.select_dtypes(include="number").fillna(0))
        print(f"\n{numeric_preview}")

        print("\n" + "-" * 70)
        print("Scoring each lead (watch stderr for AK-SCI diagnostics)")
        print("-" * 70)

        all_columns = list(df.columns)
        scores = []
        for _, row in df.iterrows():
            score = score_lead(row, all_columns)
            scores.append(score)
            status = "OK" if score is not None else "SKIPPED (unrecoverable)"
            print(f"  lead_id={row['lead_id']}: score={score}  [{status}]")

        df["lead_score"] = scores
        successful = df["lead_score"].notna().sum()

        print("\n" + "=" * 70)
        print(f"Done: {successful}/{len(df)} leads scored successfully.")
        print("=" * 70)

        out_path = Path(tmp) / "leads_scored.csv"
        df.to_csv(out_path, index=False)
        print(f"\n(In a real run, the scored file would be written to: {out_path})")


if __name__ == "__main__":
    main()
