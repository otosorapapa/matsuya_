from datetime import date
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from streamlit_app.transformers import FilterState, compute_comparison_period


def test_compute_comparison_period_handles_leap_year():
    filters = FilterState(
        stores=["本店"],
        start_date=date(2024, 2, 29),
        end_date=date(2024, 2, 29),
        categories=["ケーキ"],
    )

    comparison_filters = compute_comparison_period(filters)

    assert comparison_filters.start_date == date(2023, 2, 28)
    assert comparison_filters.end_date == date(2023, 2, 28)
    assert comparison_filters.stores == list(filters.stores)
    assert comparison_filters.categories == list(filters.categories)
    assert comparison_filters.period_granularity == filters.period_granularity
    assert comparison_filters.breakdown_dimension == filters.breakdown_dimension
