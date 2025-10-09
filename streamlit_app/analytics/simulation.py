"""Simulation utilities for management scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class SimulationInputs:
    gross_margin: float
    fixed_cost: float
    target_profit: float


DEFAULT_MARGIN_RANGE = np.linspace(0.3, 0.8, num=11)


@dataclass
class MonteCarloConfig:
    """Configuration options for the Monte Carlo simulation."""

    iterations: int = 2000
    demand_growth_mean: float = 0.02
    demand_growth_std: float = 0.05
    margin_std: float = 0.03
    fixed_cost_std: float = 0.02
    random_seed: int = 42


@dataclass
class MonteCarloResult:
    """Container with the raw trial outputs and summary statistics."""

    trials: pd.DataFrame
    summary: pd.DataFrame
    probability_of_target: float
    expected_profit: float


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


def run_monte_carlo(
    inputs: SimulationInputs,
    *,
    base_sales: float,
    config: MonteCarloConfig | None = None,
) -> MonteCarloResult:
    """Simulate profit outcomes under probabilistic demand and cost scenarios."""

    if config is None:
        config = MonteCarloConfig()

    iterations = max(int(config.iterations), 1)
    rng = np.random.default_rng(config.random_seed)

    demand_growth = rng.normal(
        loc=config.demand_growth_mean, scale=config.demand_growth_std, size=iterations
    )
    simulated_sales = np.maximum(base_sales * (1 + demand_growth), 0.0)

    margin_draw = rng.normal(
        loc=inputs.gross_margin, scale=config.margin_std, size=iterations
    )
    simulated_margin = np.clip(margin_draw, 0.01, 0.95)

    fixed_cost_draw = rng.normal(
        loc=inputs.fixed_cost, scale=inputs.fixed_cost * config.fixed_cost_std, size=iterations
    )
    simulated_fixed_cost = np.maximum(fixed_cost_draw, 0.0)

    gross_profit = simulated_sales * simulated_margin
    operating_profit = gross_profit - simulated_fixed_cost
    gap_to_target = operating_profit - inputs.target_profit
    achieved_target = operating_profit >= inputs.target_profit

    trials = pd.DataFrame(
        {
            "sales": simulated_sales,
            "gross_margin": simulated_margin,
            "fixed_cost": simulated_fixed_cost,
            "operating_profit": operating_profit,
            "gap_to_target": gap_to_target,
            "achieved_target": achieved_target,
        }
    )

    percentiles = [5, 25, 50, 75, 95]
    summary_records: List[Dict[str, float]] = []
    for percentile in percentiles:
        summary_records.append(
            {
                "percentile": percentile,
                "sales": float(np.percentile(simulated_sales, percentile)),
                "operating_profit": float(np.percentile(operating_profit, percentile)),
            }
        )
    summary = pd.DataFrame(summary_records)

    probability = float(np.mean(achieved_target))
    expected_profit = float(np.mean(operating_profit))

    return MonteCarloResult(
        trials=trials,
        summary=summary,
        probability_of_target=probability,
        expected_profit=expected_profit,
    )


def sensitivity_analysis(
    inputs: SimulationInputs,
    *,
    base_sales: float,
    variation_steps: Dict[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Return a sensitivity table for margin, cost and demand variations."""

    if variation_steps is None:
        variation_steps = {
            "gross_margin": (-0.05, -0.02, 0.0, 0.02, 0.05),
            "fixed_cost": (-0.1, -0.05, 0.0, 0.05, 0.1),
            "demand": (-0.1, -0.05, 0.0, 0.05, 0.1),
        }

    records: List[Dict[str, float | str]] = []

    for parameter, steps in variation_steps.items():
        for step in steps:
            if parameter == "gross_margin":
                margin = float(np.clip(inputs.gross_margin * (1 + step), 0.01, 0.95))
                sales = base_sales
                fixed_cost = inputs.fixed_cost
            elif parameter == "fixed_cost":
                margin = inputs.gross_margin
                sales = base_sales
                fixed_cost = max(inputs.fixed_cost * (1 + step), 0.0)
            else:
                margin = inputs.gross_margin
                sales = max(base_sales * (1 + step), 0.0)
                fixed_cost = inputs.fixed_cost

            gross_profit = sales * margin
            operating_profit = gross_profit - fixed_cost
            breakeven = fixed_cost / max(margin, 1e-6)
            records.append(
                {
                    "parameter": parameter,
                    "change_pct": float(step),
                    "sales": float(sales),
                    "gross_margin": float(margin),
                    "fixed_cost": float(fixed_cost),
                    "operating_profit": float(operating_profit),
                    "gap_to_target": float(operating_profit - inputs.target_profit),
                    "breakeven_sales": float(breakeven),
                }
            )

    result = pd.DataFrame(records)
    return result.sort_values(["parameter", "change_pct"]).reset_index(drop=True)
