"""ORIGINAL seed examples authored directly by Claude (no API key, no cost).

Run this to write a real starter corpus you can fine-tune on immediately. These are
original, worked finance/econ/quant items — enough to validate the fine-tune loop and do a
small fine-tune. To SCALE to thousands, either ask Claude (in-session) for more batches,
use the Batch API pipeline (generate_corpus.py), or mix in an open dataset.

    python scripts/seed_examples.py            # -> data/corpus.jsonl
"""
from __future__ import annotations

from finance_corpus.schema import Example, write_jsonl

TRAIN_SYSTEM = (
    "You are a precise finance, economics, and quantitative-methods tutor. "
    "Explain with clear step-by-step reasoning, then state the final answer."
)

# (topic, subtopic, difficulty, question, answer, reasoning)
SEED = [
    ("quantitative_methods", "time value of money", "foundational",
     "You deposit $1,000 at 6% annual interest compounded annually. What is it worth after 3 years?",
     "$1,191.02",
     "Future value: FV = PV·(1+r)^n = 1000·(1.06)^3 = 1000·1.191016 = $1,191.02."),
    ("quantitative_methods", "time value of money", "foundational",
     "What is the present value of $5,000 to be received in 4 years if the discount rate is 8%?",
     "$3,675.15",
     "PV = FV / (1+r)^n = 5000 / (1.08)^4 = 5000 / 1.360489 = $3,675.15."),
    ("quantitative_methods", "annuities", "intermediate",
     "What is the present value of a 5-year ordinary annuity paying $2,000 at year-end, at 5%?",
     "$8,658.95",
     "PV = PMT·[1−(1+r)^−n]/r = 2000·[1−(1.05)^−5]/0.05 = 2000·4.329477 = $8,658.95."),
    ("fixed_income", "bond pricing", "intermediate",
     "Price a 3-year bond: $1,000 face, 5% annual coupon, required yield 6%.",
     "$973.27",
     "Coupons are $50/yr. Price = 50/1.06 + 50/1.06^2 + 1050/1.06^3 = 47.17 + 44.50 + 881.60 "
     "= $973.27. It trades at a discount because the 5% coupon is below the 6% market yield."),
    ("fixed_income", "interest-rate risk", "foundational",
     "Why does a bond's price fall when market interest rates rise?",
     "Because its fixed cash flows are discounted at a higher rate, lowering their present value.",
     "A bond pays fixed coupons and face value. Price is the present value of those cash flows. "
     "When the discount rate (market yield) rises, each cash flow is worth less today, so the "
     "price falls. This inverse price–yield relationship is the core of interest-rate risk."),
    ("fixed_income", "duration", "advanced",
     "Of two otherwise identical bonds, one has a 4% coupon and one a 8% coupon. Which has the "
     "higher duration, and why?",
     "The 4% (lower-coupon) bond has the higher duration.",
     "Duration is the weighted-average time to receive a bond's cash flows. A lower coupon means "
     "more of the value comes from the distant face-value payment, pushing the weighted average "
     "further out — so the lower-coupon bond has higher duration and is more sensitive to rate "
     "changes."),
    ("equity_valuation", "dividend discount model", "intermediate",
     "A stock will pay a $2 dividend next year, growing 4% forever; required return is 9%. "
     "Value it with the Gordon growth model.",
     "$40.00",
     "Gordon growth: P = D1/(r−g) = 2 / (0.09 − 0.04) = 2 / 0.05 = $40.00."),
    ("portfolio_management", "CAPM", "intermediate",
     "Risk-free rate 3%, expected market return 9%, beta 1.2. Expected return via CAPM?",
     "10.2%",
     "CAPM: E[R] = Rf + β·(E[Rm] − Rf) = 3% + 1.2·(9% − 3%) = 3% + 7.2% = 10.2%."),
    ("portfolio_management", "beta", "foundational",
     "What does a stock's beta measure?",
     "Its sensitivity to overall market movements (systematic risk).",
     "Beta measures how much a stock tends to move relative to the market. β=1 moves with the "
     "market; β>1 amplifies market moves; β<1 dampens them. It captures systematic (undiversifiable) "
     "risk — the only risk CAPM says is rewarded."),
    ("portfolio_management", "the Sharpe ratio", "foundational",
     "A portfolio returns 12% with 15% standard deviation; the risk-free rate is 2%. Sharpe ratio?",
     "0.67",
     "Sharpe = (Rp − Rf)/σ = (12% − 2%)/15% = 10/15 = 0.67. It is excess return per unit of total risk."),
    ("portfolio_management", "diversification", "foundational",
     "Why does combining assets that are not perfectly correlated reduce portfolio risk?",
     "Because their price movements partially offset, so the portfolio's volatility is less than the "
     "weighted average of the individual volatilities.",
     "Portfolio variance depends on covariances, not just individual variances. When assets are less "
     "than perfectly correlated (ρ<1), some moves cancel out, so total volatility falls below the "
     "weighted-average of the parts. This is the free 'only free lunch' of diversification."),
    ("corporate_finance", "NPV and IRR", "intermediate",
     "A project costs $10,000 today and returns $4,000 per year for 3 years; discount rate 10%. "
     "Compute NPV and give the decision.",
     "NPV ≈ −$52.59; reject the project.",
     "PV of inflows = 4000·[1−(1.10)^−3]/0.10 = 4000·2.486852 = $9,947.41. "
     "NPV = 9,947.41 − 10,000 = −$52.59. Since NPV < 0, the project destroys value — reject it."),
    ("corporate_finance", "cost of capital (WACC)", "intermediate",
     "A firm is 60% equity (cost 12%) and 40% debt (pre-tax cost 6%); tax rate 25%. Compute WACC.",
     "9.0%",
     "After-tax cost of debt = 6%·(1−0.25) = 4.5%. WACC = 0.6·12% + 0.4·4.5% = 7.2% + 1.8% = 9.0%."),
    ("corporate_finance", "IRR", "foundational",
     "State the IRR decision rule for an independent project.",
     "Accept the project if its IRR exceeds the required rate of return (hurdle rate).",
     "IRR is the discount rate that makes NPV = 0. If IRR > hurdle rate, the project earns more than "
     "investors require, so NPV at the hurdle rate is positive — accept. If IRR < hurdle rate, reject."),
    ("financial_statements", "financial ratios", "foundational",
     "Current assets are $300,000 and current liabilities are $200,000. Current ratio and what it tells you?",
     "1.5 — short-term assets cover short-term obligations 1.5×.",
     "Current ratio = current assets / current liabilities = 300,000 / 200,000 = 1.5. A ratio above 1 "
     "means current assets exceed current liabilities, a basic check on short-term liquidity."),
    ("financial_statements", "inventory accounting", "intermediate",
     "In a period of rising prices, does FIFO or LIFO report higher net income, and why?",
     "FIFO reports higher net income.",
     "FIFO expenses the oldest (cheapest) inventory as COGS, leaving lower COGS and thus higher gross "
     "profit and net income when prices are rising. LIFO expenses the newest (most expensive) units, "
     "raising COGS and lowering reported income."),
    ("derivatives", "option payoffs", "foundational",
     "You buy a call option, strike $50, for a $3 premium. At expiry the stock is $58. Profit per share?",
     "$5",
     "Call payoff at expiry = max(S − K, 0) = max(58 − 50, 0) = $8. Profit = payoff − premium = "
     "8 − 3 = $5 per share."),
    ("derivatives", "put-call parity", "advanced",
     "A 1-year European call (strike 100) trades at $8 on a stock at $100 with a 5% risk-free rate. "
     "What is the parity price of the matching put?",
     "≈ $3.24",
     "Put-call parity: C − P = S − PV(K), so P = C − S + PV(K) = 8 − 100 + 100/1.05 "
     "= 8 − 100 + 95.238 = $3.24."),
    ("economics", "elasticity", "foundational",
     "When price rises 10%, quantity demanded falls 5%. Compute price elasticity of demand and classify it.",
     "−0.5 — demand is inelastic.",
     "Elasticity = %ΔQ / %ΔP = (−5%)/(+10%) = −0.5. Since the absolute value is below 1, demand is "
     "inelastic: quantity responds less than proportionally to price."),
    ("economics", "monetary policy", "foundational",
     "How does a central bank raising its policy interest rate typically affect inflation?",
     "It tends to slow inflation by cooling demand.",
     "Higher rates raise borrowing costs, reducing consumption and investment. Weaker aggregate demand "
     "eases upward pressure on prices, so inflation tends to fall — with a lag."),
    ("quantitative_methods", "hypothesis testing", "intermediate",
     "In plain terms, what does a p-value represent?",
     "The probability of observing data at least as extreme as what you got, assuming the null "
     "hypothesis is true.",
     "A small p-value means your data would be unlikely under the null hypothesis, giving evidence "
     "against it. It is NOT the probability the null is true — only how surprising the data are if it were."),
    ("quantitative_methods", "regression analysis", "foundational",
     "In the simple regression Y = a + bX, what does the slope coefficient b represent?",
     "The expected change in Y for a one-unit increase in X.",
     "b is the partial effect of X on Y: holding the model fixed, increasing X by one unit changes the "
     "predicted Y by b. The intercept a is the predicted Y when X = 0."),
    ("quantitative_methods", "probability distributions", "foundational",
     "A bet pays +$100 with probability 0.3 and −$40 with probability 0.7. What is its expected value?",
     "+$2",
     "E[X] = Σ p·x = 0.3·(100) + 0.7·(−40) = 30 − 28 = +$2. On average the bet is slightly positive."),
    ("ethics_and_risk", "value at risk (VaR)", "intermediate",
     "What does a 1-day 95% Value at Risk of $1,000,000 mean?",
     "On about 5% of days, the loss is expected to exceed $1,000,000.",
     "A 95% 1-day VaR of $1M means there is a 5% chance the portfolio loses more than $1M over one day "
     "(equivalently, 95% confidence the daily loss won't exceed $1M). It does not say how large the loss "
     "is on those worst days — that's the limitation VaR is criticized for."),
    ("ethics_and_risk", "conflicts of interest", "foundational",
     "An advisor earns a commission for selling a fund that is more expensive but no better for the "
     "client. What ethical problem is this, and what is the duty?",
     "A conflict of interest; the advisor owes a duty to act in the client's best interest and to disclose.",
     "The advisor's pay incentive conflicts with the client's interest in low costs. Acting on it breaches "
     "the duty of loyalty/care; the obligation is to put the client first and, at minimum, fully disclose "
     "the conflict so the client can judge the recommendation."),
    ("financial_statements", "accruals", "intermediate",
     "Under accrual accounting, when is revenue recognized?",
     "When it is earned (goods/services delivered), not necessarily when cash is received.",
     "Accrual accounting matches revenue to the period in which the performance obligation is satisfied. "
     "Cash may arrive earlier (deferred revenue, a liability) or later (accounts receivable, an asset); "
     "the income statement reflects when the revenue is earned, not when cash moves."),
]


def main() -> None:
    examples = [
        Example(instruction=q, output=a, reasoning=r, topic=topic, subtopic=sub,
                difficulty=diff, source="claude-authored")
        for (topic, sub, diff, q, a, r) in SEED
    ]
    records = [ex.to_sft(TRAIN_SYSTEM) for ex in examples]
    n = write_jsonl(records, "data/corpus.jsonl")
    print(f"wrote {n} original examples -> data/corpus.jsonl  (no API key used)")


if __name__ == "__main__":
    main()
