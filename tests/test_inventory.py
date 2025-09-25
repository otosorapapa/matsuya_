from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from streamlit_app.analytics import inventory


def test_inventory_overview_handles_missing_numeric_values():
    sales_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "store": ["Tokyo"],
            "product": ["Coffee"],
            "sales_qty": [5],
            "cogs_amount": [5000],
        }
    )

    inventory_df = pd.DataFrame(
        {
            "store": ["Tokyo"],
            "product": ["Coffee"],
            "category": ["Beverage"],
            "opening_stock": pd.Series([pd.NA], dtype="Float64"),
            "planned_purchase": pd.Series([pd.NA], dtype="Float64"),
            "safety_stock": pd.Series([pd.NA], dtype="Float64"),
        }
    )

    overview = inventory.inventory_overview(sales_df, inventory_df)

    assert not overview.empty
    first_row = overview.iloc[0]
    assert first_row["estimated_stock"] == 0.0
    assert first_row["avg_inventory"] == 0.0
    assert first_row["safety_lower"] == 0.0
    assert first_row["safety_upper"] >= 0.0
    assert first_row["coverage_days"] == 0.0
    assert first_row["safety_buffer_days"] == inventory.DEFAULT_SAFETY_BUFFER_DAYS
    assert first_row["analysis_window"] == inventory.DEFAULT_ROLLING_WINDOW
