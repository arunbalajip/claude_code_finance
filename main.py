#!/usr/bin/env python3
"""
Financial Statement Analyzer
=============================
A Claude-agent-powered app that performs DCF valuation, Black-Scholes
option pricing, and generates an investment recommendation.

Usage
-----
    python main.py                          # interactive prompts
    python main.py AAPL 50000000000 0.07   # positional args
    python main.py --ticker AAPL --cash 50000000000 --rate 0.07
"""
import argparse
import os
import sys

import anyio
from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Financial Statement Analyzer powered by Claude Agent SDK"
    )
    parser.add_argument(
        "ticker_pos",
        nargs="?",
        metavar="TICKER",
        help="Company ticker symbol (positional)",
    )
    parser.add_argument(
        "cash_pos",
        nargs="?",
        type=float,
        metavar="CASH",
        help="Cash value in USD (positional)",
    )
    parser.add_argument(
        "rate_pos",
        nargs="?",
        type=float,
        metavar="RATE",
        help="Interest/discount rate as decimal (positional)",
    )
    parser.add_argument("--ticker", "-t", help="Company ticker symbol")
    parser.add_argument("--cash", "-c", type=float, help="Cash value in USD")
    parser.add_argument(
        "--rate",
        "-r",
        type=float,
        help="Interest/discount rate as decimal (e.g. 0.07 for 7%%)",
    )
    return parser.parse_args()


def get_inputs(args: argparse.Namespace) -> tuple[str, float, float]:
    """Resolve inputs from CLI args or interactive prompts."""
    ticker = args.ticker or args.ticker_pos
    cash = args.cash if args.cash is not None else args.cash_pos
    rate = args.rate if args.rate is not None else args.rate_pos

    if not ticker:
        ticker = input("Enter company ticker symbol (e.g. AAPL): ").strip().upper()
    if not ticker:
        print("Error: ticker symbol is required.", file=sys.stderr)
        sys.exit(1)

    if cash is None:
        raw = input("Enter cash & equivalents in USD (e.g. 50000000000): ").strip()
        try:
            cash = float(raw.replace(",", "").replace("$", ""))
        except ValueError:
            print(f"Error: invalid cash value '{raw}'.", file=sys.stderr)
            sys.exit(1)

    if rate is None:
        raw = input("Enter discount / risk-free rate (e.g. 0.07 for 7%): ").strip()
        try:
            rate = float(raw.replace("%", ""))
            if rate > 1:          # user typed "7" instead of "0.07"
                rate /= 100
        except ValueError:
            print(f"Error: invalid rate '{raw}'.", file=sys.stderr)
            sys.exit(1)

    return ticker, cash, rate


def check_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'",
            file=sys.stderr,
        )
        sys.exit(1)


async def async_main() -> None:
    from agent import run_financial_analysis

    check_api_key()
    args = parse_args()
    ticker, cash_value, interest_rate = get_inputs(args)

    print(f"\n{'='*60}")
    print(f"  Financial Statement Analyzer")
    print(f"{'='*60}")
    print(f"  Ticker       : {ticker}")
    print(f"  Cash         : ${cash_value:,.0f}")
    print(f"  Discount rate: {interest_rate*100:.2f}%")
    print(f"{'='*60}\n")
    print("Running analysis via Claude Agent… (this may take 30-60 seconds)\n")

    try:
        report = await run_financial_analysis(ticker, cash_value, interest_rate)
        print(report)
    except Exception as exc:
        print(f"\nError during analysis: {exc}", file=sys.stderr)
        raise


def main() -> None:
    anyio.run(async_main)


if __name__ == "__main__":
    main()
