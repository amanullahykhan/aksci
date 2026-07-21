"""
Donor Records Cleaner & Reporter
=================================
A small automation tool built on AK-SCI for cleaning up messy donation
records exported from a CRM/spreadsheet (the kind of export a field
office or volunteer-run donation drive typically produces) and turning
them into a clean summary report.

Why this is a good fit for AK-SCI:
  - Real exports from Excel/Google Forms/a CRM are never perfectly
    clean: renamed columns, blank amounts, a stray "PKR 5,000" string
    where a number was expected, a divide-by-zero when computing an
    average. This script hits all of those on purpose.
  - `ErrorResolver(auto_fix=True)` recovers from exactly that kind of
    low-stakes formatting noise so the batch finishes, while logging
    every recovery so nothing happens silently.
  - `UnifiedFrame` is used for the final numeric summary, so the same
    code would work unchanged if the report were built from a Polars
    DataFrame instead of pandas.

Run with:
    python donor_report.py

No API key or internet connection required -- everything here uses
AK-SCI's offline local model and whitelisted auto-fix rules only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from aksci import ErrorResolver, UnifiedFrame

# -----------------------------------------------------------------------
# 1. Set up AK-SCI
# -----------------------------------------------------------------------
# auto_fix=True turns on AK-SCI's whitelisted recovery rules for any
# safe_run-decorated function that opts in with auto_fix_context.
# install_global_hook() is a separate, process-wide safety net: any
# *uncaught* exception anywhere in this script gets AK-SCI's diagnosis
# printed before the normal Python traceback.
resolver = ErrorResolver(auto_fix=True)
resolver.install_global_hook()


# -----------------------------------------------------------------------
# 2. Simulate a messy donation-drive export
# -----------------------------------------------------------------------
# This is representative of what actually comes out of a hand-run
# donation drive: a column typo from a copy-pasted template, amounts
# entered as text with currency symbols and commas, a couple of blank
# cells, and one row where "number of installments" is 0 -- which would
# break a naive "amount per installment" calculation.
RAW_DONATIONS_CSV = """donor_id,donor_name,city,ammount,installments,campaign
D-101,Fahad Siddiqui,Ghotki,"PKR 15,000",3,Ramadan Drive
D-102,Sana Bhatti,Sukkur,"PKR 8,500",1,Orphan Care
D-103,Imran Leghari,Ghotki,,4,Ramadan Drive
D-104,Ayesha Memon,Rohri,"PKR 22,000",0,WASH Fund
D-105,Bilal Rind,Ghotki,"PKR 5,000",2,Orphan Care
D-106,,Kandhkot,"PKR 12,000",1,Ramadan Drive
D-107,Nadia Chandio,Ghotki,"PKR 9,000/-",2,WASH Fund
"""
# D-101, D-102, D-105: clean once the currency text is parsed.
# D-103: blank amount -- needs a safe default, not a crash.
# D-104: installments=0 -- amount-per-installment would divide by zero.
# D-106: donor_name is blank -- kept in the report but flagged, not
#        silently dropped or guessed at (a name is not something AK-SCI
#        should ever invent).
# D-107: amount has a trailing "/-" (a common way PKR amounts get
#        handwritten/typed in Pakistan), which float() can't parse.
#        type_coercion's fix is deliberately conservative -- it only
#        strips whitespace, never guesses at stray punctuation -- so
#        this row correctly falls through *unfixed* and gets flagged
#        for a human to check, instead of auto-fix inventing a number.
#        That's the auto-fix boundary working as designed, not a bug.


def parse_amount(raw: str) -> float:
    """Turn 'PKR 15,000' into 15000.0. Deliberately naive -- assumes the
    text is always well-formed, the way a first draft usually does."""
    cleaned = raw.replace("PKR", "").replace(",", "").strip()
    return float(cleaned)


@resolver.safe_run(
    reraise=False,
    auto_fix_context=lambda raw: {"raw_value": raw, "target_type": float},
)
def safe_parse_amount(raw: str):
    return parse_amount(raw)


def per_installment(amount: float, installments: float) -> float:
    return amount / installments


@resolver.safe_run(reraise=False, auto_fix_context=lambda a, i: {"default": 0.0})
def safe_per_installment(amount: float, installments: float):
    return per_installment(amount, installments)


def clean_donation_row(row: pd.Series) -> dict:
    """Clean one row of the raw export into a normalized record.

    Each risky operation goes through its own safe_run-wrapped helper
    (see the note in ErrorResolver.safe_run's docstring: auto-fix
    replaces a function's *entire* return value, so it only makes sense
    to wrap something that does exactly one risky thing).
    """
    raw_amount = row["ammount"]  # note: source column is genuinely misspelled
    if pd.isna(raw_amount) or str(raw_amount).strip() == "":
        amount = 0.0
        amount_note = "missing amount, defaulted to 0"
    else:
        parsed = safe_parse_amount(str(raw_amount))
        if parsed is None:
            # type_coercion declined -- the text has stray punctuation
            # (e.g. "9000/-") it won't guess at. Flag it rather than
            # silently recording a wrong or zeroed amount.
            amount = 0.0
            amount_note = f"unparsable amount ({raw_amount!r}), needs manual review"
        else:
            amount = parsed
            amount_note = ""

    installments = float(row["installments"])
    per_install = safe_per_installment(amount, installments)

    donor_name = row["donor_name"]
    name_note = ""
    if pd.isna(donor_name) or str(donor_name).strip() == "":
        donor_name = "(name not recorded)"
        name_note = "donor name missing"

    notes = "; ".join(n for n in (amount_note, name_note) if n)

    return {
        "donor_id": row["donor_id"],
        "donor_name": donor_name,
        "city": row["city"],
        "campaign": row["campaign"],
        "amount_pkr": amount,
        "per_installment_pkr": round(per_install, 2) if per_install is not None else 0.0,
        "notes": notes,
    }


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_rows = [clean_donation_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(cleaned_rows)


def main() -> None:
    print("=" * 72)
    print("Donor Records Cleaner & Reporter -- built with AK-SCI")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "donations_export.csv"
        csv_path.write_text(RAW_DONATIONS_CSV)

        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} raw donation records from {csv_path.name}")
        print(df.to_string(index=False))

        print("\n" + "-" * 72)
        print("Cleaning records (watch stderr for AK-SCI diagnostics)")
        print("-" * 72)
        report = build_report(df)

        print("\n" + "-" * 72)
        print("Cleaned report")
        print("-" * 72)
        print(report.to_string(index=False))

        # UnifiedFrame gives a consistent summary API regardless of
        # whether this report were backed by pandas or Polars downstream.
        numeric = UnifiedFrame(report[["amount_pkr", "per_installment_pkr"]])
        print(f"\n{numeric}")
        print(numeric.describe())

        total = report["amount_pkr"].sum()
        by_campaign = report.groupby("campaign")["amount_pkr"].sum().sort_values(ascending=False)
        flagged = report[report["notes"] != ""]

        print("\n" + "=" * 72)
        print(f"Total raised: PKR {total:,.0f} across {len(report)} donors")
        print("\nBy campaign:")
        for campaign, amount in by_campaign.items():
            print(f"  {campaign:<20s} PKR {amount:,.0f}")
        print(f"\n{len(flagged)} record(s) flagged for follow-up:")
        for _, row in flagged.iterrows():
            print(f"  {row['donor_id']} ({row['donor_name']}): {row['notes']}")
        print("=" * 72)

        out_path = Path(tmp) / "donations_report.csv"
        report.to_csv(out_path, index=False)
        print(f"\n(In a real run, the cleaned report would be saved to: {out_path})")


if __name__ == "__main__":
    main()
