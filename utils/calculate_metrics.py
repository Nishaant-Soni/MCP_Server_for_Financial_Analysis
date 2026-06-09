import pandas as pd


def calculate_growth_rates(series: list) -> dict:
    """
    Calculates growth rates for a time series (e.g., revenue, earnings).
    Args:
        series: List of numbers ordered oldest to newest.
    Returns:
        Dict with annualized growth rate and list of period-over-period growth rates.
    """
    growth_rates = []
    try:
        for i in range(1, len(series)):
            prev = series[i - 1]
            curr = series[i]
            if prev != 0:
                growth = (curr - prev) / abs(prev)
            else:
                growth = None
            growth_rates.append(growth)
        if len(series) > 1 and series[0] != 0:
            annualized = (series[-1] / series[0]) ** (1 / (len(series) - 1)) - 1
        else:
            annualized = None
    except Exception as e:
        return {"error": str(e)}
    return {"annualized_growth_rate": annualized, "period_growth_rates": growth_rates}


def calculate_technical_indicators(
    price_data: list, indicator: str, window: int = 14
) -> dict:
    """
    Calculates technical indicators on stock price data.
    Args:
        price_data: List of closing prices (numbers) OR list of OHLCV dicts with keys
                    'open', 'high', 'low', 'close'. OHLCV form is required for 'atr'
                    and 'stochastic'; either form works for all others.
        indicator: 'sma', 'ema', 'rsi', 'macd', 'volatility', 'bollinger_bands', 'atr', 'stochastic'
        window: Window size for the indicator.
    Returns:
        Dict with indicator values.
    """
    result = {}
    if price_data and isinstance(price_data[0], dict):
        closes = pd.Series([d["close"] for d in price_data])
        highs = pd.Series([d.get("high", d["close"]) for d in price_data])
        lows = pd.Series([d.get("low", d["close"]) for d in price_data])
    else:
        closes = pd.Series(price_data)
        highs = closes
        lows = closes
    try:
        if indicator == "sma":
            result["sma"] = closes.rolling(window=window).mean().tolist()
        elif indicator == "ema":
            result["ema"] = closes.ewm(span=window, adjust=False).mean().tolist()
        elif indicator == "rsi":
            delta = closes.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            alpha = 1.0 / window
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            result["rsi"] = rsi.tolist()
        elif indicator == "macd":
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            result["macd"] = macd.tolist()
            result["macd_signal"] = signal.tolist()
        elif indicator == "volatility":
            result["volatility"] = (
                closes.pct_change().rolling(window=window).std().tolist()
            )
        elif indicator == "bollinger_bands":
            middle = closes.rolling(window=window).mean()
            std = closes.rolling(window=window).std()
            result["upper"] = (middle + 2 * std).tolist()
            result["middle"] = middle.tolist()
            result["lower"] = (middle - 2 * std).tolist()
        elif indicator == "atr":
            prev_close = closes.shift(1)
            tr = pd.concat(
                [
                    highs - lows,
                    (highs - prev_close).abs(),
                    (lows - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            result["atr"] = tr.ewm(alpha=1.0 / window, adjust=False).mean().tolist()
        elif indicator == "stochastic":
            lowest_low = lows.rolling(window=window).min()
            highest_high = highs.rolling(window=window).max()
            denom = highest_high - lowest_low
            k = 100 * (closes - lowest_low) / denom.replace(0, float("nan"))
            d = k.rolling(window=3).mean()
            result["stochastic_k"] = k.tolist()
            result["stochastic_d"] = d.tolist()
        else:
            result["error"] = f"Unknown indicator: {indicator}"
    except Exception as e:
        result["error"] = str(e)
    return result


def calculate_trading_opportunities(
    prices: list, short_window: int, long_window: int
) -> pd.DataFrame:
    """
    Calculates trading opportunities from price data. Create signals based on crossovers:
    - A 'Golden Cross' (bullish signal) occurs when the short-term MA crosses above the long-term MA.
    - A 'Death Cross' (bearish signal) occurs when the short-term MA crosses below the long-term MA.
    Args:
        prices: List of dicts with OHLCV and date.
        short_window: Short window for the short moving average.
        long_window: Long window for the long moving average.
    Returns:
        pd.DataFrame with trading opportunities.
    """
    prices["short_ma"] = prices["close"].rolling(window=short_window).mean()
    prices["long_ma"] = prices["close"].rolling(window=long_window).mean()
    prices["prev_short"] = prices["short_ma"].shift(1)
    prices["prev_long"] = prices["long_ma"].shift(1)
    prices["signal"] = 0
    prices.loc[
        (prices["short_ma"] > prices["long_ma"])
        & (prices["prev_short"] <= prices["prev_long"]),
        "signal",
    ] = 1
    prices.loc[
        (prices["short_ma"] < prices["long_ma"])
        & (prices["prev_short"] >= prices["prev_long"]),
        "signal",
    ] = -1
    return prices


def backtest_strategy(
    prices: list, short_window: int = 9, long_window: int = 21
) -> dict:
    """
    Backtests a moving-average crossover strategy (Golden/Death Cross) on historical prices.
    Args:
        prices: List of dicts with keys 'date', 'close' (and optionally 'open','high','low','volume').
        short_window: Short MA window.
        long_window: Long MA window.
    Returns:
        Dict with total_return, buy_and_hold_return, num_trades, win_rate, max_drawdown,
        sharpe_ratio, and a list of individual trade records.
    """
    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])
    df = calculate_trading_opportunities(df, short_window, long_window)

    position = 0
    entry_price = 0.0
    trades = []

    for _, row in df.iterrows():
        if row["signal"] == 1 and position == 0:
            position = 1
            entry_price = row["close"]
            trades.append(
                {"type": "buy", "date": str(row["date"].date()), "price": entry_price}
            )
        elif row["signal"] == -1 and position == 1:
            position = 0
            pnl = (row["close"] - entry_price) / entry_price
            trades.append(
                {
                    "type": "sell",
                    "date": str(row["date"].date()),
                    "price": row["close"],
                    "pnl": round(pnl, 4),
                }
            )

    if position == 1:
        pnl = (df.iloc[-1]["close"] - entry_price) / entry_price
        trades.append(
            {
                "type": "sell",
                "date": str(df.iloc[-1]["date"].date()),
                "price": df.iloc[-1]["close"],
                "pnl": round(pnl, 4),
                "note": "open position closed at period end",
            }
        )

    sell_trades = [t for t in trades if t["type"] == "sell"]
    if not sell_trades:
        return {
            "error": "No completed trades in this period — try a longer date range or different windows",
            "trades": trades,
        }

    pnls = [t["pnl"] for t in sell_trades]

    total_return = 1.0
    for p in pnls:
        total_return *= 1 + p
    total_return -= 1

    buy_hold = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

    equity = pd.Series([1.0])
    for p in pnls:
        equity = pd.concat(
            [equity, pd.Series([equity.iloc[-1] * (1 + p)])], ignore_index=True
        )
    rolling_max = equity.cummax()
    max_drawdown = ((equity - rolling_max) / rolling_max).min()

    if len(pnls) > 1:
        mean_r = sum(pnls) / len(pnls)
        std_r = (sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
        sharpe = (mean_r / std_r) if std_r > 0 else None
    else:
        sharpe = None

    return {
        "total_return": round(total_return, 4),
        "buy_and_hold_return": round(buy_hold, 4),
        "outperformance": round(total_return - buy_hold, 4),
        "num_trades": len(sell_trades),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "trades": trades,
    }


def custom_formula_evaluator(formula: str, data: dict) -> dict:
    """
    Evaluates a custom formula using variables from data dict.
    Args:
        formula: A string formula, e.g. '(net_income / shareholders_equity) * 100'
        data: Dict of variable values.
    Returns:
        Dict with result or error.
    """
    try:
        result = eval(formula, {"__builtins__": None}, data)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def extract_value(financial_data: dict, key: str, preferred_sources: list) -> float:
    """
    Extracts a value from a nested financial data dict.
    """
    for src in preferred_sources:
        if src in financial_data and key in financial_data[src]:
            return financial_data[src][key]
    return None


def calculate_financial_metric(financial_data: dict, indicator: str) -> dict:
    """
    Calculates a specified financial metric from the provided data. The metrics are:
    - gross_margin (need retrieve income_statement)
    - operating_margin (need retrieve income_statement)
    - net_profit_margin (need retrieve income_statement)
    - ebitda (need retrieve income_statement)
    - debt_to_equity (need retrieve balance_sheet)
    - current_ratio (need retrieve balance_sheet)
    - quick_ratio (need retrieve balance_sheet)
    - book_value_per_share (need retrieve balance_sheet)
    - free_cash_flow (need retrieve cash_flow)
    - cash_flow_margin (need retrieve cash_flow)
    - roe (need retrieve income_statement and balance_sheet)
    - roa (need retrieve income_statement and balance_sheet)
    - pe_ratio (need retrieve income_statement and market)
    - pb_ratio (need retrieve income_statement and market)
    - dividend_yield (need retrieve income_statement and market)
    Args:
        financial_data: dict (flat or nested)
        indicator: str, one of the supported metric names
    Returns:
        Dict with keys: 'indicator', 'value', and optionally 'error'.
    """

    # Helper for safe division
    def safe_div(num, denom):
        return num / denom if num is not None and denom not in (None, 0) else None

    # Metric calculation logic
    if indicator == "gross_margin":
        revenue = extract_value(financial_data, "revenue", ["income_statement"])
        cogs = extract_value(financial_data, "cost_of_goods_sold", ["income_statement"])
        value = (
            safe_div(revenue - cogs, revenue)
            if revenue is not None and cogs is not None
            else None
        )
    elif indicator == "operating_margin":
        revenue = extract_value(financial_data, "revenue", ["income_statement"])
        operating_income = extract_value(
            financial_data, "operating_income", ["income_statement"]
        )
        value = safe_div(operating_income, revenue)
    elif indicator == "net_profit_margin":
        revenue = extract_value(financial_data, "revenue", ["income_statement"])
        net_income = extract_value(financial_data, "net_income", ["income_statement"])
        value = safe_div(net_income, revenue)
    elif indicator == "ebitda":
        ebitda = extract_value(financial_data, "ebitda", ["income_statement"])
        if ebitda is not None:
            value = ebitda
        else:
            operating_income = extract_value(
                financial_data, "operating_income", ["income_statement"]
            )
            depreciation = extract_value(
                financial_data, "depreciation", ["income_statement", "cash_flow"]
            )
            amortization = extract_value(
                financial_data, "amortization", ["income_statement", "cash_flow"]
            )
            value = (
                operating_income + depreciation + amortization
                if None not in (operating_income, depreciation, amortization)
                else None
            )
    elif indicator == "debt_to_equity":
        total_liabilities = extract_value(
            financial_data, "total_liabilities", ["balance_sheet"]
        )
        shareholders_equity = extract_value(
            financial_data, "shareholders_equity", ["balance_sheet"]
        )
        value = safe_div(total_liabilities, shareholders_equity)
    elif indicator == "current_ratio":
        current_assets = extract_value(
            financial_data, "current_assets", ["balance_sheet"]
        )
        current_liabilities = extract_value(
            financial_data, "current_liabilities", ["balance_sheet"]
        )
        value = safe_div(current_assets, current_liabilities)
    elif indicator == "quick_ratio":
        current_assets = extract_value(
            financial_data, "current_assets", ["balance_sheet"]
        )
        inventory = extract_value(financial_data, "inventory", ["balance_sheet"])
        current_liabilities = extract_value(
            financial_data, "current_liabilities", ["balance_sheet"]
        )
        value = (
            safe_div(current_assets - inventory, current_liabilities)
            if None not in (current_assets, inventory, current_liabilities)
            else None
        )
    elif indicator == "book_value_per_share":
        shareholders_equity = extract_value(
            financial_data, "shareholders_equity", ["balance_sheet"]
        )
        shares_outstanding = extract_value(
            financial_data, "shares_outstanding", ["market", "income_statement"]
        )
        value = safe_div(shareholders_equity, shares_outstanding)
    elif indicator == "free_cash_flow":
        operating_cash_flow = extract_value(
            financial_data, "operating_cash_flow", ["cash_flow"]
        )
        capex = extract_value(financial_data, "capital_expenditures", ["cash_flow"])
        value = (
            operating_cash_flow - capex
            if None not in (operating_cash_flow, capex)
            else None
        )
    elif indicator == "cash_flow_margin":
        operating_cash_flow = extract_value(
            financial_data, "operating_cash_flow", ["cash_flow"]
        )
        revenue = extract_value(financial_data, "revenue", ["income_statement"])
        value = safe_div(operating_cash_flow, revenue)
    elif indicator == "roe":
        net_income = extract_value(financial_data, "net_income", ["income_statement"])
        shareholders_equity = extract_value(
            financial_data, "shareholders_equity", ["balance_sheet"]
        )
        value = safe_div(net_income, shareholders_equity)
    elif indicator == "roa":
        net_income = extract_value(financial_data, "net_income", ["income_statement"])
        total_assets = extract_value(financial_data, "total_assets", ["balance_sheet"])
        value = safe_div(net_income, total_assets)
    elif indicator == "pe_ratio":
        price = extract_value(financial_data, "price", ["market"])
        net_income = extract_value(financial_data, "net_income", ["income_statement"])
        shares_outstanding = extract_value(
            financial_data, "shares_outstanding", ["market", "income_statement"]
        )
        eps = safe_div(net_income, shares_outstanding)
        value = safe_div(price, eps) if eps not in (None, 0) else None
    elif indicator == "pb_ratio":
        price = extract_value(financial_data, "price", ["market"])
        shareholders_equity = extract_value(
            financial_data, "shareholders_equity", ["balance_sheet"]
        )
        shares_outstanding = extract_value(
            financial_data, "shares_outstanding", ["market", "income_statement"]
        )
        book_value_per_share = safe_div(shareholders_equity, shares_outstanding)
        value = (
            safe_div(price, book_value_per_share)
            if book_value_per_share not in (None, 0)
            else None
        )
    elif indicator == "dividend_yield":
        dividends_per_share = extract_value(
            financial_data, "dividends_per_share", ["income_statement", "market"]
        )
        price = extract_value(financial_data, "price", ["market"])
        value = safe_div(dividends_per_share, price)
    else:
        return {"indicator": indicator, "value": None, "error": "Unknown indicator"}
    return {"indicator": indicator, "value": value}
