"""
AK-SCI end-to-end demo.

Run with: python examples/demo.py
Shows: error diagnosis, the unified frame API, gradient descent, and a
chunked micro-pipeline over a CSV file -- no API key required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from aksci import ErrorResolver, MicroPipeline, UnifiedFrame, ml, stats

# --- 1. Error diagnosis -----------------------------------------------------
print("=" * 60)
print("1. AI-assisted error diagnosis (offline, no API key needed)")
print("=" * 60)

resolver = ErrorResolver()


@resolver.safe_run(reraise=False)
def load_price(row: dict) -> float:
    return row["price"]  # typo further down will trigger this on purpose


sample_row = {"prise": 19.99}  # deliberately misspelled
result = load_price(sample_row)
print(f"Result (swallowed after diagnosis): {result}\n")

# Fuzzy suggestion: you wanted "price", here's what's actually in the data:
suggestion = resolver.suggest_column("price", list(sample_row.keys()))
print(f"Column 'price' not found. Closest match in the data: {suggestion!r}\n")

# --- 2. Unified frame API ---------------------------------------------------
print("=" * 60)
print("2. UnifiedFrame: one API over pandas and Polars")
print("=" * 60)

df = pd.DataFrame({"revenue": [100, 200, 300, 400], "cost": [60, 90, 150, 210]})
uf = UnifiedFrame(df).standardize()
print(uf)
print(uf.to_pandas())
print()

# --- 3. Gradient descent, written from scratch ------------------------------
print("=" * 60)
print("3. Gradient descent (hand-rolled, for learning)")
print("=" * 60)

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 1))
y = 4.0 * X[:, 0] + 2.0 + rng.normal(scale=0.1, size=100)

gd_result = ml.gradient_descent(X, y, learning_rate=0.1, epochs=200)
print(f"Learned weight: {gd_result.weights[0]:.3f} (true: 4.0)")
print(f"Learned bias  : {gd_result.bias:.3f} (true: 2.0)")
print(f"Final loss    : {gd_result.loss_history[-1]:.5f}\n")

corr = stats.correlation(X[:, 0], y)
print(f"Correlation between X and y: r={corr['r']:.3f}\n")

# --- 4. Micro-pipeline over a CSV file --------------------------------------
print("=" * 60)
print("4. Memory-bounded micro-pipeline over a CSV file")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmp:
    csv_path = Path(tmp) / "sales.csv"
    pd.DataFrame({"revenue": range(1, 21), "cost": range(1, 21)}).to_csv(csv_path, index=False)

    pipeline = MicroPipeline(chunk_size=5)  # only 5 rows in memory at a time
    pipeline.add_stage("add_margin", lambda chunk: chunk.assign(margin=chunk["revenue"] - chunk["cost"] * 0.6))

    for i, chunk in enumerate(pipeline.run_csv(str(csv_path))):
        print(f"Chunk {i}: {len(chunk)} rows, columns={list(chunk.columns)}")

print("\nDone.")
