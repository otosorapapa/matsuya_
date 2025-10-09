"""Simulation utilities for management scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

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


def annual_cash_flow(*, sales: float, gross_margin: float, fixed_cost: float) -> float:
    """Return annual operating cash flow for a scenario."""

    gross_profit = max(sales * gross_margin, 0.0)
    return gross_profit - fixed_cost


def project_cash_flows(
    annual_cash_flow: float, growth_rate: float, periods: int
) -> List[float]:
    """Generate a list of future cash flows applying a constant growth rate."""

    flows: List[float] = []
    if periods <= 0:
        return flows
    value = float(annual_cash_flow)
    for _ in range(periods):
        flows.append(value)
        value *= 1 + growth_rate
    return flows


def net_present_value(
    cash_flows: Iterable[float], discount_rate: float, *, initial_investment: float = 0.0
) -> float:
    """Calculate the net present value for a series of future cash flows."""

    npv = -float(initial_investment)
    period = 1
    for flow in cash_flows:
        discount_factor = (1 + discount_rate) ** period
        if discount_factor == 0:
            period += 1
            continue
        npv += float(flow) / discount_factor
        period += 1
    return npv
