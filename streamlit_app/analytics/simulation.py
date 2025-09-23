"""Simulation utilities for management scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class SimulationInputs:
    gross_margin: float
    fixed_cost: float
    target_profit: float


DEFAULT_MARGIN_RANGE = np.linspace(0.3, 0.8, num=11)


def breakeven_sales_curve(margins: Iterable[float], fixed_cost: float) -> pd.DataFrame:
    """Return sensitivity analysis data for various gross margins."""
    records = []
    for margin in margins:
        if margin <= 0:
            breakeven = float("inf")
        else:
            breakeven = fixed_cost / margin
        records.append({"gross_margin": margin, "breakeven_sales": breakeven})
    return pd.DataFrame(records)


def required_sales(inputs: SimulationInputs) -> Dict[str, float]:
    """Calculate breakeven and target profit sales requirements."""
    gm = max(inputs.gross_margin, 1e-6)
    breakeven = inputs.fixed_cost / gm
    target_sales = (inputs.fixed_cost + inputs.target_profit) / gm
    return {"breakeven": breakeven, "target_sales": target_sales}


def consolidate_fixed_costs(costs: Dict[str, float]) -> float:
    return float(sum(costs.values()))
