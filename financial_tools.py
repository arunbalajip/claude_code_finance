"""
Financial analysis tools: DCF valuation, Black-Scholes option pricing,
and market data retrieval via yfinance.
"""
import json
import math
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_stock_info(ticker: str) -> dict:
    """Fetch key stock metrics from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info

    hist = stock.history(period="1y")
    if hist.empty:
        return {"error": f"No historical data found for {ticker}"}

    current_price = float(hist["Close"].iloc[-1])

    # Historical volatility (annualised, log-returns)
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    hist_vol = float(log_returns.std() * math.sqrt(252))

    return {
        "ticker": ticker.upper(),
        "company_name": info.get("longName", ticker),
        "current_price": current_price,
        "historical_volatility": hist_vol,
        "market_cap": info.get("marketCap"),
        "revenue": info.get("totalRevenue"),
        "free_cash_flow": info.get("freeCashflow"),
        "eps": info.get("trailingEps"),
        "pe_ratio": info.get("trailingPE"),
        "beta": info.get("beta", 1.0),
        "shares_outstanding": info.get("sharesOutstanding"),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "price_52w_high": info.get("fiftyTwoWeekHigh"),
        "price_52w_low": info.get("fiftyTwoWeekLow"),
    }


def get_financial_statements(ticker: str) -> dict:
    """Fetch income statement and cash flow data."""
    stock = yf.Ticker(ticker)

    try:
        cf = stock.cashflow
        income = stock.income_stmt

        fcf_values = []
        if not cf.empty and "Free Cash Flow" in cf.index:
            fcf_values = [
                float(v) for v in cf.loc["Free Cash Flow"].values if not math.isnan(v)
            ][:4]

        revenue_values = []
        if not income.empty and "Total Revenue" in income.index:
            revenue_values = [
                float(v)
                for v in income.loc["Total Revenue"].values
                if not math.isnan(v)
            ][:4]

        return {
            "free_cash_flows_historical": fcf_values,
            "revenue_historical": revenue_values,
        }
    except Exception as exc:
        return {"error": str(exc), "free_cash_flows_historical": [], "revenue_historical": []}


# ---------------------------------------------------------------------------
# DCF valuation
# ---------------------------------------------------------------------------

def calculate_dcf(
    ticker: str,
    cash_value: float,
    interest_rate: float,
    growth_rate: float = 0.05,
    terminal_growth_rate: float = 0.025,
    projection_years: int = 5,
) -> dict:
    """
    Discounted Cash Flow valuation.

    Parameters
    ----------
    ticker            : stock symbol
    cash_value        : current cash & equivalents (USD)
    interest_rate     : discount rate / WACC (e.g. 0.10 for 10%)
    growth_rate       : FCF growth rate for projection period
    terminal_growth_rate : perpetual growth rate for terminal value
    projection_years  : number of years to project
    """
    stock_info = get_stock_info(ticker)
    if "error" in stock_info:
        return stock_info

    fin_data = get_financial_statements(ticker)
    historical_fcf = fin_data.get("free_cash_flows_historical", [])

    # Estimate base FCF
    if historical_fcf:
        base_fcf = historical_fcf[0]
    elif stock_info.get("free_cash_flow"):
        base_fcf = stock_info["free_cash_flow"]
    else:
        # Fallback: rough estimate from market cap
        base_fcf = (stock_info.get("market_cap") or 0) * 0.05

    if base_fcf <= 0:
        base_fcf = abs(base_fcf) if base_fcf != 0 else 1e8

    # Project FCFs
    projected_fcfs = []
    pv_fcfs = []
    for year in range(1, projection_years + 1):
        fcf = base_fcf * ((1 + growth_rate) ** year)
        pv = fcf / ((1 + interest_rate) ** year)
        projected_fcfs.append(fcf)
        pv_fcfs.append(pv)

    # Terminal value (Gordon Growth)
    terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (interest_rate - terminal_growth_rate)
    pv_terminal = terminal_value / ((1 + interest_rate) ** projection_years)

    total_pv = sum(pv_fcfs) + pv_terminal
    intrinsic_value = total_pv + cash_value

    shares = stock_info.get("shares_outstanding") or 1
    intrinsic_per_share = intrinsic_value / shares if shares else None

    current_price = stock_info["current_price"]
    upside = ((intrinsic_per_share - current_price) / current_price * 100) if intrinsic_per_share else None

    return {
        "ticker": ticker.upper(),
        "current_price": current_price,
        "base_fcf": base_fcf,
        "projected_fcfs": projected_fcfs,
        "pv_of_fcfs": pv_fcfs,
        "total_pv_fcfs": sum(pv_fcfs),
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal,
        "total_enterprise_value": total_pv,
        "cash_added": cash_value,
        "total_intrinsic_value": intrinsic_value,
        "shares_outstanding": shares,
        "intrinsic_value_per_share": intrinsic_per_share,
        "current_price_per_share": current_price,
        "upside_percentage": upside,
        "discount_rate_used": interest_rate,
        "growth_rate_used": growth_rate,
        "terminal_growth_rate_used": terminal_growth_rate,
        "projection_years": projection_years,
    }


# ---------------------------------------------------------------------------
# Black-Scholes option pricing
# ---------------------------------------------------------------------------

def black_scholes_call(
    S: float,   # spot price
    K: float,   # strike price
    T: float,   # time to expiry in years
    r: float,   # risk-free rate
    sigma: float,  # volatility
) -> dict:
    """Return call option price, delta, gamma, theta, vega."""
    if T <= 0:
        return {"price": max(S - K, 0), "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100

    return {
        "price": round(call_price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


def calculate_options_analysis(
    ticker: str,
    interest_rate: float,
    strike_offsets: list[float] | None = None,
    expiry_months: list[int] | None = None,
) -> dict:
    """
    Compute Black-Scholes call option prices for a grid of strikes and expiries.

    Parameters
    ----------
    ticker         : stock symbol
    interest_rate  : risk-free rate (e.g. 0.05)
    strike_offsets : list of % offsets from spot, e.g. [-10, -5, 0, 5, 10]
    expiry_months  : months to expiry, e.g. [1, 3, 6, 12]
    """
    if strike_offsets is None:
        strike_offsets = [-10, -5, 0, 5, 10, 20]
    if expiry_months is None:
        expiry_months = [1, 3, 6, 9, 12]

    stock_info = get_stock_info(ticker)
    if "error" in stock_info:
        return stock_info

    S = stock_info["current_price"]
    sigma = stock_info["historical_volatility"]
    r = interest_rate

    today = datetime.today()
    results = []

    for months in expiry_months:
        expiry_date = today + timedelta(days=30 * months)
        T = months / 12.0
        row = {
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "months_to_expiry": months,
            "options": [],
        }
        for offset in strike_offsets:
            K = round(S * (1 + offset / 100), 2)
            moneyness = "ATM" if offset == 0 else ("ITM" if offset < 0 else "OTM")
            bs = black_scholes_call(S, K, T, r, sigma)
            row["options"].append(
                {
                    "strike": K,
                    "offset_pct": offset,
                    "moneyness": moneyness,
                    **bs,
                }
            )
        results.append(row)

    return {
        "ticker": ticker.upper(),
        "spot_price": S,
        "historical_volatility": sigma,
        "risk_free_rate": r,
        "analysis_date": today.strftime("%Y-%m-%d"),
        "options_grid": results,
    }


# ---------------------------------------------------------------------------
# JSON-serialisable helpers (used by the agent tools)
# ---------------------------------------------------------------------------

def run_full_analysis(ticker: str, cash_value: float, interest_rate: float) -> str:
    """Aggregate DCF + options analysis into a single JSON string."""
    dcf = calculate_dcf(ticker, cash_value, interest_rate)
    opts = calculate_options_analysis(ticker, interest_rate)
    stock = get_stock_info(ticker)

    payload = {
        "stock_info": stock,
        "dcf_analysis": dcf,
        "options_analysis": opts,
    }
    return json.dumps(payload, default=str)
