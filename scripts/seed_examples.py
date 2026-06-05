"""ORIGINAL seed examples authored directly by Claude (no API key, no cost).

Run this to write a real starter corpus you can fine-tune on immediately. These are
original, worked finance/econ/quant items — enough to validate the fine-tune loop and do a
small fine-tune. To SCALE to thousands, either ask Claude (in-session) for more batches,
use the Batch API pipeline (generate_corpus.py), or mix in an open dataset.

    python scripts/seed_examples.py            # -> data/corpus.jsonl
"""
from __future__ import annotations

from finance_corpus import TRAIN_SYSTEM
from finance_corpus.schema import Example, write_jsonl

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
    ("fixed_income", "current yield", "foundational",
     "A bond with a $40 annual coupon trades at $950. What is its current yield?",
     "4.21%",
     "Current yield = annual coupon / price = 40 / 950 = 0.0421 = 4.21%. It ignores capital gain/loss "
     "to maturity, unlike yield to maturity."),
    ("fixed_income", "coupon vs yield", "intermediate",
     "If a bond's coupon rate is below its yield to maturity, will it trade at a premium, par, or discount?",
     "At a discount (below face value).",
     "When the coupon (5% say) is below what the market demands (the YTM), investors will only buy the "
     "bond for less than face value, so its lower coupon plus the price gain to par equals the market yield. "
     "Coupon > YTM → premium; coupon = YTM → par; coupon < YTM → discount."),
    ("equity_valuation", "relative valuation multiples", "foundational",
     "A stock trades at $40 with earnings per share of $2. What is its P/E ratio?",
     "20×",
     "P/E = price / EPS = 40 / 2 = 20. Investors are paying $20 per $1 of annual earnings."),
    ("financial_statements", "income statement", "foundational",
     "A company has net income of $2,000,000 and 500,000 shares outstanding. What is its EPS?",
     "$4.00",
     "EPS = net income / shares outstanding = 2,000,000 / 500,000 = $4.00 per share."),
    ("portfolio_management", "expected return and risk", "foundational",
     "You buy a stock at $100, receive a $2 dividend, and sell it for $108. What is your holding-period return?",
     "10%",
     "HPR = (ending price − beginning price + income) / beginning price = (108 − 100 + 2) / 100 = 10 / 100 = 10%."),
    ("portfolio_management", "systematic vs unsystematic risk", "intermediate",
     "What is the difference between systematic and unsystematic risk, and which can be diversified away?",
     "Systematic (market) risk cannot be diversified away; unsystematic (firm-specific) risk can.",
     "Unsystematic risk is specific to a company or sector and shrinks as you add uncorrelated holdings. "
     "Systematic risk affects the whole market (rates, recessions) and remains no matter how diversified you "
     "are — which is why CAPM rewards only systematic risk (beta)."),
    ("quantitative_methods", "correlation and covariance", "foundational",
     "What is the possible range of a correlation coefficient, and what does +1 mean?",
     "−1 to +1; +1 means the two variables move together perfectly and proportionally.",
     "Correlation is bounded in [−1, +1]. +1 is perfect positive linear co-movement, −1 perfect inverse, "
     "and 0 no linear relationship. Diversification benefit grows as correlation falls below +1."),
    ("derivatives", "the Greeks", "intermediate",
     "What does the delta of a call option represent, and what is its range?",
     "The sensitivity of the option's price to a $1 change in the underlying; it ranges from 0 to 1 for a call.",
     "Delta ≈ ∂(option price)/∂(underlying). A call delta near 1 is deep in-the-money (moves nearly one-for-one "
     "with the stock); near 0 is far out-of-the-money. It also approximates the probability of finishing in-the-money."),
    ("economics", "exchange rates", "intermediate",
     "If the US dollar appreciates against the euro, what happens to US exporters?",
     "US goods become more expensive for European buyers, tending to reduce US exports.",
     "A stronger dollar means Europeans need more euros to buy the same dollar-priced US goods, raising the "
     "effective price abroad and dampening export demand. It conversely makes imports cheaper for US buyers."),
    ("economics", "monetary policy", "advanced",
     "If the reserve requirement is 10%, what is the simple money multiplier?",
     "10",
     "Simple money multiplier = 1 / reserve ratio = 1 / 0.10 = 10. A $1 increase in reserves can support up "
     "to $10 of new deposits through the banking system (ignoring cash holdings and excess reserves)."),
    ("economics", "inflation and CPI", "intermediate",
     "Nominal interest is 6% and inflation is 2%. Approximately what is the real interest rate?",
     "≈ 4%",
     "By the Fisher approximation, real rate ≈ nominal rate − inflation = 6% − 2% = 4%. (Exactly, "
     "1.06/1.02 − 1 = 3.92%.) The real rate is what your purchasing power actually grows by."),
    ("financial_statements", "depreciation", "intermediate",
     "An asset costs $50,000, has a $5,000 salvage value and a 5-year life. Annual straight-line depreciation?",
     "$9,000 per year",
     "Straight-line depreciation = (cost − salvage) / useful life = (50,000 − 5,000) / 5 = 45,000 / 5 = $9,000/yr."),
    ("corporate_finance", "working capital management", "foundational",
     "Current assets are $300,000 and current liabilities are $200,000. What is net working capital?",
     "$100,000",
     "Net working capital = current assets − current liabilities = 300,000 − 200,000 = $100,000. Positive NWC "
     "means short-term assets exceed short-term obligations."),
    ("alternative_investments", "hedge fund strategies", "intermediate",
     "What does a long/short equity hedge fund do, and why?",
     "It buys (longs) stocks it expects to outperform and short-sells stocks it expects to underperform.",
     "By being long the winners and short the losers, the fund profits from relative performance and reduces "
     "exposure to broad market moves (it can be roughly market-neutral), isolating stock-selection skill."),
    ("alternative_investments", "commodities", "advanced",
     "A futures curve is in 'contango.' What does that mean?",
     "Futures prices are higher than the current spot price (the curve slopes upward with maturity).",
     "Contango means longer-dated futures cost more than near-dated/spot, often reflecting storage and carrying "
     "costs. A long futures position that must roll contracts can lose value rolling up the curve (negative roll yield)."),
    ("portfolio_management", "factor models", "advanced",
     "In performance evaluation, what does a portfolio's 'alpha' measure?",
     "Return earned beyond what its risk exposure (e.g., beta) would predict.",
     "Alpha is the intercept in a regression of portfolio excess returns on factor (e.g., market) excess returns "
     "— the risk-adjusted excess return attributable to skill rather than to taking on systematic risk."),
    ("ethics_and_risk", "fiduciary duty", "intermediate",
     "An analyst learns material non-public information about a company. What must they not do?",
     "They must not trade on it or tip others — that would be illegal insider trading.",
     "Material non-public information, if acted upon, gives an unfair advantage and breaches duties to the market "
     "and the information's source. The analyst must refrain from trading and from sharing it until it is public."),
    ("ethics_and_risk", "risk management frameworks", "intermediate",
     "Name one key limitation of Value at Risk (VaR) as a risk measure.",
     "It says nothing about how large losses can be beyond the VaR threshold (the tail).",
     "VaR gives a loss level at a confidence level (e.g., 95%) but is silent on the severity of the worst 5% of "
     "outcomes. Conditional VaR / expected shortfall addresses this by averaging losses in the tail."),
    ("quantitative_methods", "probability distributions", "intermediate",
     "In a normal distribution, approximately what percentage of observations fall within ±1 standard deviation of the mean?",
     "About 68%",
     "By the empirical rule for a normal distribution: ~68% within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ."),
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
