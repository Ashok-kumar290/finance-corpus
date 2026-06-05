"""Finance / economics / quant topic taxonomy — generic subject labels (our own).

These are *topic names*, which are uncopyrightable facts/ideas. The model writes original
explanations of each; we never copy any publisher's prose. Difficulty tiers loosely span
foundational → professional. ANGLES diversify prompts so requests don't collapse to
near-duplicate questions (variety has to come from the prompt — Opus 4.8 has no
temperature parameter).
"""
from __future__ import annotations

TOPICS: dict[str, list[str]] = {
    "quantitative_methods": [
        "time value of money", "discounting and compounding", "probability distributions",
        "hypothesis testing", "regression analysis", "correlation and covariance",
        "sampling and estimation",
    ],
    "economics": [
        "supply and demand", "elasticity", "market structures", "monetary policy",
        "fiscal policy", "inflation and CPI", "exchange rates", "business cycles",
    ],
    "financial_statements": [
        "income statement", "balance sheet", "cash flow statement", "accruals",
        "financial ratios", "revenue recognition", "inventory accounting",
    ],
    "corporate_finance": [
        "capital budgeting", "NPV and IRR", "cost of capital (WACC)", "capital structure",
        "dividend policy", "working capital management",
    ],
    "equity_valuation": [
        "dividend discount model", "free cash flow valuation", "relative valuation multiples",
        "the equity risk premium", "growth and terminal value",
    ],
    "fixed_income": [
        "bond pricing", "yield to maturity", "duration and convexity", "the yield curve",
        "credit spreads", "interest-rate risk",
    ],
    "derivatives": [
        "forwards and futures", "option payoffs", "put-call parity",
        "the Black-Scholes idea", "the Greeks", "hedging with derivatives",
    ],
    "portfolio_management": [
        "expected return and risk", "diversification", "the efficient frontier",
        "the CAPM", "beta and systematic risk", "the Sharpe ratio", "factor models",
    ],
    "alternative_investments": [
        "real estate", "private equity", "hedge fund strategies", "commodities",
    ],
    "ethics_and_risk": [
        "fiduciary duty", "conflicts of interest", "risk management frameworks",
        "value at risk (VaR)",
    ],
}

DIFFICULTIES = ["foundational", "intermediate", "advanced"]

ANGLES = [
    "a numerical worked problem with specific figures",
    "a conceptual 'why' question that probes understanding",
    "a multiple-choice question with one correct and three plausible-but-wrong options",
    "a short scenario applying the concept to a realistic situation",
    "a 'compare and contrast' question between two related ideas",
    "a common-misconception question that corrects a typical error",
]


def iter_specs(limit: int | None = None):
    """Yield (topic, subtopic, difficulty, angle) generation specs, round-robined across
    angles/difficulties so coverage stays balanced. `limit` caps the count."""
    n = 0
    for topic, subs in TOPICS.items():
        for i, sub in enumerate(subs):
            difficulty = DIFFICULTIES[i % len(DIFFICULTIES)]
            angle = ANGLES[(i + n) % len(ANGLES)]
            yield (topic, sub, difficulty, angle)
            n += 1
            if limit is not None and n >= limit:
                return


def total_specs() -> int:
    return sum(len(s) for s in TOPICS.values())
