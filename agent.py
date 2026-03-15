"""
Claude Agent orchestrator for the Financial Statement Analyzer.

The agent is given three custom MCP tools:
  - get_stock_data        : fetch live market data for a ticker
  - run_dcf_valuation     : run a DCF model
  - run_options_analysis  : price a grid of call options (Black-Scholes)

It then reasons over the results and produces:
  - Intrinsic value (DCF) vs current price
  - Options grid with prices, delta, theta, vega for multiple expiries
  - A BUY / HOLD / SELL recommendation with rationale
"""
import json

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    tool,
    create_sdk_mcp_server,
)

from financial_tools import (
    get_stock_info,
    calculate_dcf,
    calculate_options_analysis,
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool(
    "get_stock_data",
    "Fetch current price, volatility, P/E, market cap and other metrics for a stock ticker.",
    {"ticker": str},
)
async def get_stock_data_tool(args: dict) -> dict:
    data = get_stock_info(args["ticker"])
    return {"content": [{"type": "text", "text": json.dumps(data, default=str)}]}


@tool(
    "run_dcf_valuation",
    (
        "Run a Discounted Cash Flow (DCF) valuation for a company. "
        "Returns projected free cash flows, terminal value, intrinsic value per share, "
        "and the upside/downside vs the current market price."
    ),
    {
        "ticker": str,
        "cash_value": float,
        "interest_rate": float,
        "growth_rate": float,
        "terminal_growth_rate": float,
    },
)
async def run_dcf_tool(args: dict) -> dict:
    result = calculate_dcf(
        ticker=args["ticker"],
        cash_value=float(args["cash_value"]),
        interest_rate=float(args["interest_rate"]),
        growth_rate=float(args.get("growth_rate", 0.05)),
        terminal_growth_rate=float(args.get("terminal_growth_rate", 0.025)),
    )
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


@tool(
    "run_options_analysis",
    (
        "Price European call options using Black-Scholes for a range of strikes "
        "(expressed as % offset from spot) and expiry dates. "
        "Returns price, delta, gamma, theta, vega for each contract."
    ),
    {
        "ticker": str,
        "interest_rate": float,
        "strike_offsets": list,
        "expiry_months": list,
    },
)
async def run_options_tool(args: dict) -> dict:
    result = calculate_options_analysis(
        ticker=args["ticker"],
        interest_rate=float(args["interest_rate"]),
        strike_offsets=args.get("strike_offsets"),
        expiry_months=args.get("expiry_months"),
    )
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert financial analyst.

When asked to analyse a company you MUST:
1. Call `get_stock_data` to fetch the current price and key metrics.
2. Call `run_dcf_valuation` to compute the intrinsic value.
3. Call `run_options_analysis` to price a matrix of call options.

Then produce a structured report with these exact sections:

## Company Overview
Brief summary of the company and its sector.

## DCF Valuation
Present the key DCF numbers in a table:
| Metric | Value |
| --- | --- |
Include: base FCF, total PV of FCFs, terminal value, enterprise value, intrinsic value/share,
current price, upside %, discount rate, growth rate, terminal growth rate.

## Options Analysis
For each expiry date, show a table of call options:
| Strike | Offset % | Moneyness | Price | Delta | Gamma | Theta | Vega |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Investment Recommendation
Give a clear BUY / HOLD / SELL recommendation with concise reasoning covering:
- DCF upside/downside
- Options market signals (volatility, skew)
- Key risks
"""

async def run_financial_analysis(
    ticker: str,
    cash_value: float,
    interest_rate: float,
) -> str:
    """
    Run the full financial analysis agent and return the formatted report.

    Parameters
    ----------
    ticker        : e.g. "AAPL"
    cash_value    : cash & equivalents in USD (e.g. 50_000_000_000)
    interest_rate : discount / risk-free rate as a decimal (e.g. 0.07)
    """
    server = create_sdk_mcp_server(
        "finance-tools",
        tools=[get_stock_data_tool, run_dcf_tool, run_options_tool],
    )

    prompt = (
        f"Please perform a complete financial analysis for **{ticker.upper()}**.\n\n"
        f"Inputs provided by the user:\n"
        f"- Cash & equivalents: ${cash_value:,.0f}\n"
        f"- Discount / risk-free rate: {interest_rate*100:.2f}%\n\n"
        f"Use these inputs in your DCF and options calculations."
    )

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"finance": server},
        model="claude-opus-4-6",
        max_turns=20,
    )

    report_parts: list[str] = []

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        report_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                if message.result and message.result.strip():
                    report_parts.append(message.result)

    return "\n".join(report_parts).strip()
