"""Generate technical-analysis HTML reports for Yahoo Finance equities."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
import yfinance as yf
from plotly.subplots import make_subplots


METHODS = [
    "Moving average trend (SMA 50/200)",
    "MACD momentum and crossover",
    "RSI momentum and extremes",
    "Bollinger Bands and volatility squeeze",
    "ADX / directional movement trend strength",
    "ATR volatility and price scenarios",
    "Classic pivot support and resistance",
    "Fibonacci retracement levels",
    "Stochastic oscillator",
    "On-balance volume trend",
    "Donchian channel breakout",
    "Ichimoku cloud trend",
]
ANALYSIS_KEYS = (
    "sma", "macd", "rsi", "bollinger", "adx", "atr", "pivots", "fibonacci",
    "stochastic", "obv", "donchian", "ichimoku",
)
SUPPORTED_ANALYSES = dict(zip(ANALYSIS_KEYS, METHODS))
METHOD_HELP = {
    METHODS[0]: "Compares the 50-session and 200-session simple moving averages. A rising 50-day average above the 200-day average is often read as a long-term uptrend; the reverse suggests a downtrend. Crosses lag price and are not guarantees.",
    METHODS[1]: "MACD is the difference between 12- and 26-session exponential averages; its 9-session signal average helps identify momentum changes. MACD above its signal is commonly read as positive momentum, below as negative. Crosses can whipsaw in sideways markets.",
    METHODS[2]: "The 14-session Relative Strength Index measures recent gains against losses on a 0-100 scale. Values above 70 are often called overbought and below 30 oversold, but strong trends can remain extreme. The 50 level is a simple momentum midpoint.",
    METHODS[3]: "Bollinger Bands place two standard deviations around a 20-session average. Price near a band may indicate unusually strong movement, not an automatic reversal. Narrow bands indicate low recent volatility and a possible expansion, without predicting its direction.",
    METHODS[4]: "ADX estimates trend strength; +DI and -DI indicate directional movement. ADX below about 20 is often considered weak or range-bound. When ADX is higher, +DI above -DI favors an uptrend and the reverse favors a downtrend.",
    METHODS[5]: "Average True Range measures typical price movement, including gaps. It describes volatility, not direction. The displayed five-session envelope scales ATR by the square root of five as an illustrative range, not a forecast or price target.",
    METHODS[6]: "Classic pivot levels use the latest high, low, and close to calculate a pivot plus support (S1/S2) and resistance (R1/R2) references. Traders watch them as possible reaction areas; they are short-term levels and can be crossed easily.",
    METHODS[7]: "Fibonacci retracements mark 38.2%, 50%, and 61.8% pullbacks across the recent 126-session high-low range. They are commonly watched as possible reaction zones, but depend on the chosen swing and are not predictive on their own.",
    METHODS[8]: "The Stochastic oscillator compares the close with its 14-session high-low range. %K above %D suggests improving short-term momentum; below suggests weakening momentum. Readings above 80 or below 20 are commonly called overbought or oversold, not automatic reversal signals.",
    METHODS[9]: "On-Balance Volume cumulatively adds volume on up-closes and subtracts it on down-closes. OBV above its 20-session average suggests volume flow is rising; below suggests it is weakening. It is most useful as confirmation alongside price, not as a standalone signal.",
    METHODS[10]: "Donchian Channels show the prior 20-session highest high and lowest low. A close above or below the channel can indicate a breakout or breakdown; price inside the channel is not a breakout. Range breaks can fail, especially in choppy markets.",
    METHODS[11]: "Ichimoku compares price with a conversion line, base line, and a forward-shifted cloud. Price above the cloud with conversion above base is a common bullish alignment; below with conversion below base is bearish. The cloud can act as a broad support/resistance area, but signals lag.",
}
MARKER_CATEGORIES = {
    "RSI thresholds": ("RSI oversold", "RSI overbought"),
    "MACD crosses": ("MACD bullish cross", "MACD bearish cross"),
    "SMA 50/200 crosses": ("Golden cross", "Death cross"),
    "Stochastic crosses": ("Stochastic bullish cross", "Stochastic bearish cross"),
    "Donchian breakouts": ("Donchian upside breakout", "Donchian downside breakdown"),
    "DMI crosses": ("Bullish DMI cross", "Bearish DMI cross"),
}
MARKER_CATEGORY_BY_LABEL = {
    label: category
    for category, labels in MARKER_CATEGORIES.items()
    for label in labels
}
CHART_VIEWS = {
    "price": {"label": "Price", "xaxis": "xaxis", "yaxis": "y"},
    "volume": {"label": "Volume", "xaxis": "xaxis2", "yaxis": "y2"},
    "macd": {"label": "MACD", "xaxis": "xaxis3", "yaxis": "y3"},
    "rsi": {"label": "RSI", "xaxis": "xaxis4", "yaxis": "y4"},
    "adx": {"label": "ADX / DMI", "xaxis": "xaxis5", "yaxis": "y5"},
    "stochastic": {"label": "Stochastic", "xaxis": "xaxis6", "yaxis": "y6"},
    "obv": {"label": "On-balance volume", "xaxis": "xaxis7", "yaxis": "y7"},
}


def resolve_symbol(instrument: str | dict[str, str]) -> tuple[str, str]:
    """Return (Yahoo symbol, display label) for a ticker or search query."""
    if isinstance(instrument, str):
        query = instrument.strip()
        if not query:
            raise ValueError("An instrument entry cannot be empty.")
        is_query = bool(re.search(r"\s", query)) or bool(re.fullmatch(r"[A-Za-z]{2}[A-Za-z0-9]{10}", query))
        if not is_query:
            return query.upper(), query
        label = query
    elif isinstance(instrument, dict):
        fields = [(key, instrument.get(key)) for key in ("ticker", "name", "isin")]
        key, value = next(((key, value) for key, value in fields if value), (None, None))
        if not value:
            raise ValueError("Each instrument object needs a ticker, name, or isin field.")
        if key == "ticker":
            return str(value).strip().upper(), str(value).strip()
        query = str(value).strip()
        label = query
    else:
        raise ValueError("Each instrument must be a ticker string or an object.")

    search = yf.Search(query, max_results=10, news_count=0)
    quotes = [quote for quote in search.quotes if quote.get("quoteType") == "EQUITY" and quote.get("symbol")]
    if not quotes:
        raise ValueError(f"Yahoo Finance could not find an equity for '{query}'. Try specifying its ticker.")
    return quotes[0]["symbol"], label


def download_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"Yahoo Finance returned no price history for {symbol}.")
    if isinstance(data.columns, pd.MultiIndex):
        if symbol in data.columns.get_level_values(0):
            data = data.xs(symbol, axis=1, level=0)
        else:
            data = data.droplevel(-1, axis=1)
    data.columns = [str(column).title() for column in data.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Yahoo Finance data is missing columns: {', '.join(sorted(missing))}.")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if len(data) < 30:
        raise ValueError(f"Only {len(data)} usable price bars were returned for {symbol}; at least 30 are needed.")
    return data


def calculate_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]

    result["SMA20"] = close.rolling(20).mean()
    result["SMA50"] = close.rolling(50).mean()
    result["SMA200"] = close.rolling(200).mean()

    result["EMA12"] = close.ewm(span=12, adjust=False).mean()
    result["EMA26"] = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = result["EMA12"] - result["EMA26"]
    result["MACDSignal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACDHist"] = result["MACD"] - result["MACDSignal"]

    change = close.diff()
    average_gain = change.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    average_loss = -change.clip(upper=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_loss == 0) & (average_gain > 0), "RSI"] = 100

    result["BBMid"] = result["SMA20"]
    deviation = close.rolling(20).std(ddof=0)
    result["BBUpper"] = result["BBMid"] + 2 * deviation
    result["BBLower"] = result["BBMid"] - 2 * deviation
    result["BBWidth"] = (result["BBUpper"] - result["BBLower"]) / result["BBMid"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    result["ATR"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    upward_move = high.diff()
    downward_move = -low.diff()
    positive_dm = upward_move.where((upward_move > downward_move) & (upward_move > 0), 0.0)
    negative_dm = downward_move.where((downward_move > upward_move) & (downward_move > 0), 0.0)
    smoothed_tr = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    result["PlusDI"] = 100 * positive_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / smoothed_tr
    result["MinusDI"] = 100 * negative_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / smoothed_tr
    directional_sum = (result["PlusDI"] + result["MinusDI"]).replace(0, np.nan)
    dx = 100 * (result["PlusDI"] - result["MinusDI"]).abs() / directional_sum
    result["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    stochastic_low = low.rolling(14).min()
    stochastic_high = high.rolling(14).max()
    stochastic_range = (stochastic_high - stochastic_low).replace(0, np.nan)
    result["StochK"] = 100 * (close - stochastic_low) / stochastic_range
    result["StochD"] = result["StochK"].rolling(3).mean()

    volume_direction = np.sign(close.diff()).fillna(0)
    result["OBV"] = (volume_direction * result["Volume"]).cumsum()
    result["OBVSignal"] = result["OBV"].rolling(20).mean()

    result["DonchianHigh"] = high.rolling(20).max().shift(1)
    result["DonchianLow"] = low.rolling(20).min().shift(1)

    result["IchimokuConversion"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    result["IchimokuBase"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    result["IchimokuSpanA"] = ((result["IchimokuConversion"] + result["IchimokuBase"]) / 2).shift(26)
    result["IchimokuSpanB"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return result


def _number(value: Any, digits: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.{digits}f}"


def analyze(
    indicators: pd.DataFrame, selected_methods: list[str] | None = None
) -> tuple[list[dict[str, str]], dict[str, float]]:
    latest = indicators.iloc[-1]
    close = float(latest["Close"])
    findings: list[dict[str, str]] = []

    def add(method: str, signal: str, detail: str, tone: str = "neutral") -> None:
        findings.append({"method": method, "signal": signal, "detail": detail, "tone": tone})

    sma50, sma200 = latest["SMA50"], latest["SMA200"]
    if pd.notna(sma50) and pd.notna(sma200):
        signal = "Bullish" if sma50 > sma200 and close > sma50 else "Bearish" if sma50 < sma200 and close < sma50 else "Mixed"
        detail = f"Close {_number(close)}; SMA 50 {_number(sma50)}; SMA 200 {_number(sma200)}."
        average_gap = indicators["SMA50"] - indicators["SMA200"]
        recent_crosses = average_gap[(average_gap.shift(1) <= 0) & (average_gap > 0)]
        recent_death_crosses = average_gap[(average_gap.shift(1) >= 0) & (average_gap < 0)]
        crosses = [(index, "Golden") for index in recent_crosses.index] + [(index, "Death") for index in recent_death_crosses.index]
        recent = [(index, kind) for index, kind in crosses if indicators.index.get_loc(index) >= max(1, len(indicators) - 5)]
        if recent:
            index, kind = max(recent, key=lambda item: indicators.index.get_loc(item[0]))
            sessions_ago = len(indicators) - 1 - indicators.index.get_loc(index)
            detail += f" Recent {kind.lower()} cross {sessions_ago} session(s) ago."
    else:
        signal, detail = "Insufficient history", "SMA 50/200 comparison needs at least 200 price bars."
    add(METHODS[0], signal, detail, "bullish" if signal == "Bullish" else "bearish" if signal == "Bearish" else "neutral")

    macd, macd_signal = latest["MACD"], latest["MACDSignal"]
    difference = indicators["MACD"] - indicators["MACDSignal"]
    bullish_crosses = difference[(difference.shift(1) <= 0) & (difference > 0)]
    bearish_crosses = difference[(difference.shift(1) >= 0) & (difference < 0)]
    macd_crosses = [(index, "Bullish") for index in bullish_crosses.index] + [(index, "Bearish") for index in bearish_crosses.index]
    recent_macd_crosses = [(index, direction) for index, direction in macd_crosses if indicators.index.get_loc(index) >= max(1, len(indicators) - 5)]
    macd_detail = f"MACD {_number(macd, 3)} is {'above' if macd > macd_signal else 'below'} its signal {_number(macd_signal, 3)}."
    if recent_macd_crosses:
        index, direction = max(recent_macd_crosses, key=lambda item: indicators.index.get_loc(item[0]))
        sessions_ago = len(indicators) - 1 - indicators.index.get_loc(index)
        macd_detail += f" {direction} crossover {sessions_ago} session(s) ago."
    add(
        METHODS[1],
        "Bullish" if macd > macd_signal else "Bearish",
        macd_detail,
        "bullish" if macd > macd_signal else "bearish",
    )

    rsi = latest["RSI"]
    rsi_signal = "Overbought" if rsi >= 70 else "Oversold" if rsi <= 30 else "Bullish momentum" if rsi >= 50 else "Bearish momentum"
    rsi_tone = "neutral" if rsi >= 70 or rsi <= 30 else "bullish" if rsi >= 50 else "bearish"
    add(METHODS[2], rsi_signal, f"14-period RSI is {_number(rsi, 1)} (70/30 are commonly watched thresholds).", rsi_tone)

    upper, lower, width = latest["BBUpper"], latest["BBLower"], latest["BBWidth"]
    if close >= upper:
        bb_signal, bb_detail = "Upper-band pressure", f"Close {_number(close)} is at or above the upper band {_number(upper)}."
    elif close <= lower:
        bb_signal, bb_detail = "Lower-band pressure", f"Close {_number(close)} is at or below the lower band {_number(lower)}."
    else:
        bb_signal, bb_detail = "Within bands", f"Close is between {_number(lower)} and {_number(upper)}."
    width_history = indicators["BBWidth"].dropna().tail(126)
    if len(width_history) >= 20 and width <= width_history.quantile(0.2):
        bb_detail += " Band width is in its trailing 20th percentile (a possible squeeze)."
    add(METHODS[3], bb_signal, bb_detail, "neutral")

    adx, plus_di, minus_di = latest["ADX"], latest["PlusDI"], latest["MinusDI"]
    if pd.isna(adx):
        adx_signal, adx_detail = "Insufficient history", "ADX requires additional price bars."
    elif adx < 20:
        adx_signal, adx_detail = "Weak trend", f"ADX {_number(adx, 1)} is below 20; directional trend strength is limited."
    else:
        adx_signal = "Bullish trend" if plus_di > minus_di else "Bearish trend"
        adx_detail = f"ADX {_number(adx, 1)}; +DI {_number(plus_di, 1)}; -DI {_number(minus_di, 1)}."
    adx_tone = "bullish" if adx_signal == "Bullish trend" else "bearish" if adx_signal == "Bearish trend" else "neutral"
    add(METHODS[4], adx_signal, adx_detail, adx_tone)

    atr = latest["ATR"]
    atr_pct = float(atr / close * 100) if pd.notna(atr) else float("nan")
    five_day_move = float(atr * np.sqrt(5)) if pd.notna(atr) else float("nan")
    add(
        METHODS[5],
        "Volatility context",
        f"ATR(14) {_number(atr)} ({_number(atr_pct, 2)}% of price); illustrative 5-session range {_number(close - five_day_move)} to {_number(close + five_day_move)}.",
        "neutral",
    )

    last_bar = indicators.iloc[-1]
    pivot = (float(last_bar["High"]) + float(last_bar["Low"]) + close) / 3
    pivot_levels = {
        "S2": pivot - (float(last_bar["High"]) - float(last_bar["Low"])),
        "S1": 2 * pivot - float(last_bar["High"]),
        "Pivot": pivot,
        "R1": 2 * pivot - float(last_bar["Low"]),
        "R2": pivot + (float(last_bar["High"]) - float(last_bar["Low"])),
    }
    support = max((value for value in pivot_levels.values() if value < close), default=float("nan"))
    resistance = min((value for value in pivot_levels.values() if value > close), default=float("nan"))
    add(METHODS[6], "Pivot levels", f"Nearest pivot support {_number(support)}; nearest pivot resistance {_number(resistance)} (from latest bar H/L/C).", "neutral")

    window = indicators.tail(min(126, len(indicators)))
    swing_low, swing_high = float(window["Low"].min()), float(window["High"].max())
    fib_levels = {f"Fib {ratio * 100:g}%": swing_high - (swing_high - swing_low) * ratio for ratio in (0.382, 0.5, 0.618)}
    fib_support = max((value for value in fib_levels.values() if value < close), default=float("nan"))
    fib_resistance = min((value for value in fib_levels.values() if value > close), default=float("nan"))
    add(
        METHODS[7],
        "Retracement levels",
        f"126-bar range {_number(swing_low)} to {_number(swing_high)}; nearest Fibonacci reference below/above price: {_number(fib_support)} / {_number(fib_resistance)}.",
        "neutral",
    )

    stoch_k, stoch_d = latest["StochK"], latest["StochD"]
    stoch_tone = "bullish" if stoch_k > stoch_d else "bearish" if stoch_k < stoch_d else "neutral"
    stoch_zone = "overbought" if stoch_k >= 80 else "oversold" if stoch_k <= 20 else "mid-range"
    stoch_signal = "Bullish momentum" if stoch_tone == "bullish" else "Bearish momentum" if stoch_tone == "bearish" else "Neutral"
    add(METHODS[8], stoch_signal, f"%K {_number(stoch_k, 1)}; %D {_number(stoch_d, 1)}; {stoch_zone} zone.", stoch_tone)

    obv, obv_signal = latest["OBV"], latest["OBVSignal"]
    obv_tone = "bullish" if obv > obv_signal else "bearish" if obv < obv_signal else "neutral"
    obv_reading = "Rising volume flow" if obv_tone == "bullish" else "Weakening volume flow" if obv_tone == "bearish" else "Neutral volume flow"
    add(METHODS[9], obv_reading, f"OBV {_number(obv, 0)} is {'above' if obv_tone == 'bullish' else 'below' if obv_tone == 'bearish' else 'near'} its 20-session average {_number(obv_signal, 0)}.", obv_tone)

    donchian_high, donchian_low = latest["DonchianHigh"], latest["DonchianLow"]
    if close > donchian_high:
        donchian_signal, donchian_detail, donchian_tone = "Upside breakout", f"Close {_number(close)} is above the prior 20-session high {_number(donchian_high)}.", "bullish"
    elif close < donchian_low:
        donchian_signal, donchian_detail, donchian_tone = "Downside breakdown", f"Close {_number(close)} is below the prior 20-session low {_number(donchian_low)}.", "bearish"
    else:
        donchian_signal, donchian_detail, donchian_tone = "Within channel", f"Prior 20-session channel: {_number(donchian_low)} to {_number(donchian_high)}.", "neutral"
    add(METHODS[10], donchian_signal, donchian_detail, donchian_tone)

    cloud_top = max(latest["IchimokuSpanA"], latest["IchimokuSpanB"])
    cloud_bottom = min(latest["IchimokuSpanA"], latest["IchimokuSpanB"])
    if close > cloud_top and latest["IchimokuConversion"] > latest["IchimokuBase"]:
        ichimoku_signal, ichimoku_tone = "Bullish alignment", "bullish"
    elif close < cloud_bottom and latest["IchimokuConversion"] < latest["IchimokuBase"]:
        ichimoku_signal, ichimoku_tone = "Bearish alignment", "bearish"
    else:
        ichimoku_signal, ichimoku_tone = "Mixed / inside cloud", "neutral"
    add(
        METHODS[11], ichimoku_signal,
        f"Close {_number(close)}; cloud {_number(cloud_bottom)} to {_number(cloud_top)}; conversion {_number(latest['IchimokuConversion'])}; base {_number(latest['IchimokuBase'])}.",
        ichimoku_tone,
    )

    levels = {
        **pivot_levels, **fib_levels,
        "Swing low": swing_low, "Swing high": swing_high,
        "Donchian low": float(donchian_low), "Donchian high": float(donchian_high),
    }
    zone_width = max(float(atr) * 0.7, close * 0.005) if pd.notna(atr) else close * 0.005
    local_lows = window["Low"].where(window["Low"] == window["Low"].rolling(5, center=True, min_periods=3).min()).dropna().tolist()
    local_highs = window["High"].where(window["High"] == window["High"].rolling(5, center=True, min_periods=3).max()).dropna().tolist()
    support_candidates = local_lows + [value for value in levels.values() if np.isfinite(value) and value < close]
    resistance_candidates = local_highs + [value for value in levels.values() if np.isfinite(value) and value > close]

    def clustered_zones(candidates: list[float], direction: str) -> list[tuple[float, float]]:
        candidates = sorted(value for value in candidates if value < close) if direction == "support" else sorted(value for value in candidates if value > close)
        clusters: list[list[float]] = []
        for value in candidates:
            if not clusters or value - clusters[-1][-1] > zone_width:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        ordered_clusters = sorted(clusters, key=lambda values: np.mean(values), reverse=direction == "support")
        zones = []
        for cluster in ordered_clusters[:3]:
            if direction == "support":
                upper = max(cluster)
                zones.append((upper - zone_width, upper))
            else:
                lower = min(cluster)
                zones.append((lower, lower + zone_width))
        return zones

    support_zones = clustered_zones(support_candidates, "support")
    resistance_zones = clustered_zones(resistance_candidates, "resistance")
    for index, label in enumerate(("Support zone", "Second support zone", "Third support zone")):
        levels[f"{label} low"], levels[f"{label} high"] = support_zones[index] if index < len(support_zones) else (float("nan"), float("nan"))
    for index, label in enumerate(("Resistance zone", "Second resistance zone", "Third resistance zone")):
        levels[f"{label} low"], levels[f"{label} high"] = resistance_zones[index] if index < len(resistance_zones) else (float("nan"), float("nan"))
    levels["ATR lower"] = close - five_day_move if pd.notna(five_day_move) else float("nan")
    levels["ATR upper"] = close + five_day_move if pd.notna(five_day_move) else float("nan")
    if selected_methods is not None:
        selected_names = {SUPPORTED_ANALYSES[key] for key in selected_methods}
        findings = [finding for finding in findings if finding["method"] in selected_names]
    return findings, levels


def detect_signal_events(indicators: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def add_events(mask: pd.Series, label: str, tone: str, panel: str, value_column: str | None = None) -> None:
        for index in indicators.index[mask.fillna(False)]:
            event = {
                "index": index,
                "label": label,
                "tone": tone,
                "price": float(indicators.at[index, "Close"]),
                "value": float(indicators.at[index, value_column]) if value_column else float(indicators.at[index, "Close"]),
                "panel": panel,
                "category": MARKER_CATEGORY_BY_LABEL[label],
            }
            events.append({**event, "panel": "price", "value": event["price"]})
            if panel != "price":
                events.append(event)

    rsi = indicators["RSI"]
    add_events((rsi <= 30) & (rsi.shift(1) > 30), "RSI oversold", "neutral", "rsi", "RSI")
    add_events((rsi >= 70) & (rsi.shift(1) < 70), "RSI overbought", "neutral", "rsi", "RSI")

    macd_gap = indicators["MACD"] - indicators["MACDSignal"]
    add_events((macd_gap > 0) & (macd_gap.shift(1) <= 0), "MACD bullish cross", "bullish", "macd", "MACD")
    add_events((macd_gap < 0) & (macd_gap.shift(1) >= 0), "MACD bearish cross", "bearish", "macd", "MACD")

    average_gap = indicators["SMA50"] - indicators["SMA200"]
    add_events((average_gap > 0) & (average_gap.shift(1) <= 0), "Golden cross", "bullish", "price")
    add_events((average_gap < 0) & (average_gap.shift(1) >= 0), "Death cross", "bearish", "price")

    stochastic_gap = indicators["StochK"] - indicators["StochD"]
    add_events((stochastic_gap > 0) & (stochastic_gap.shift(1) <= 0) & (indicators["StochK"] <= 20), "Stochastic bullish cross", "bullish", "stochastic", "StochK")
    add_events((stochastic_gap < 0) & (stochastic_gap.shift(1) >= 0) & (indicators["StochK"] >= 80), "Stochastic bearish cross", "bearish", "stochastic", "StochK")

    upside_breakout = (indicators["Close"] > indicators["DonchianHigh"]) & (indicators["Close"].shift(1) <= indicators["DonchianHigh"].shift(1))
    downside_breakdown = (indicators["Close"] < indicators["DonchianLow"]) & (indicators["Close"].shift(1) >= indicators["DonchianLow"].shift(1))
    add_events(upside_breakout, "Donchian upside breakout", "bullish", "price")
    add_events(downside_breakdown, "Donchian downside breakdown", "bearish", "price")

    directional_gap = indicators["PlusDI"] - indicators["MinusDI"]
    strong_trend = indicators["ADX"] >= 20
    add_events((directional_gap > 0) & (directional_gap.shift(1) <= 0) & strong_trend, "Bullish DMI cross", "bullish", "adx", "PlusDI")
    add_events((directional_gap < 0) & (directional_gap.shift(1) >= 0) & strong_trend, "Bearish DMI cross", "bearish", "adx", "MinusDI")
    return events


def trading_xaxis_breaks(index: pd.Index) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(index).normalize().unique().sort_values()
    breaks: list[dict[str, Any]] = [{"bounds": ["sat", "mon"]}]
    if len(dates) < 2 or dates.to_series().diff().median() > pd.Timedelta(days=4):
        return breaks
    calendar_dates = pd.date_range(dates.min(), dates.max(), freq="D")
    missing_weekdays = calendar_dates.difference(dates)
    weekday_closures = missing_weekdays[missing_weekdays.dayofweek < 5]
    if len(weekday_closures):
        breaks.append({"values": [date.strftime("%Y-%m-%d") for date in weekday_closures]})
    return breaks


def create_chart(indicators: pd.DataFrame, levels: dict[str, float], symbol: str) -> str:
    chart_data = indicators.tail(180)
    current_price = float(indicators["Close"].iloc[-1])
    figure = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.36, 0.13, 0.11, 0.085, 0.095, 0.085, 0.135],
        subplot_titles=("Price, trend averages and support / resistance zones", "Trading volume", "MACD", "RSI (14)", "ADX / DMI", "Stochastic (14, 3)", "On-balance volume"),
    )
    figure.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["DonchianHigh"], name="Donchian high", legendgroup="donchian", line={"color": "#7c91a3", "width": 1}), row=1, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["DonchianLow"], name="Donchian low", legendgroup="donchian", line={"color": "#7c91a3", "width": 1}, fill="tonexty", fillcolor="rgba(91,125,151,0.08)"), row=1, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["IchimokuSpanA"], name="Ichimoku span A", legendgroup="ichimoku", line={"color": "rgba(59,128,91,0.6)", "width": 1}), row=1, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["IchimokuSpanB"], name="Ichimoku span B", legendgroup="ichimoku", line={"color": "rgba(191,112,73,0.6)", "width": 1}, fill="tonexty", fillcolor="rgba(122,157,118,0.12)"), row=1, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["IchimokuConversion"], name="Ichimoku conversion", legendgroup="ichimoku", line={"color": "#c16b45", "width": 1}), row=1, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["IchimokuBase"], name="Ichimoku base", legendgroup="ichimoku", line={"color": "#356a9a", "width": 1}), row=1, col=1)
    for column, label, color in (("SMA50", "SMA 50", "#e8793e"), ("SMA200", "SMA 200", "#356a9a")):
        figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data[column], name=label, line={"color": color, "width": 1.5}), row=1, col=1)
    for label in ("S1", "R1", "Fib 38.2%", "Fib 50%", "Fib 61.8%"):
        value = levels.get(label)
        if value is not None and np.isfinite(value):
            line_style = "dash" if label in ("S1", "R1") else "dot"
            figure.add_hline(y=value, line_dash=line_style, line_color="#728078", opacity=0.65, row=1, col=1)
    for low_key, high_key, label, color in (
        ("Support zone low", "Support zone high", "Support zone 1", "rgba(145,190,137,0.28)"),
        ("Second support zone low", "Second support zone high", "Support zone 2", "rgba(145,190,137,0.16)"),
        ("Third support zone low", "Third support zone high", "Support zone 3", "rgba(145,190,137,0.10)"),
        ("Resistance zone low", "Resistance zone high", "Resistance zone 1", "rgba(230,145,91,0.27)"),
        ("Second resistance zone low", "Second resistance zone high", "Resistance zone 2", "rgba(230,145,91,0.16)"),
        ("Third resistance zone low", "Third resistance zone high", "Resistance zone 3", "rgba(230,145,91,0.10)"),
    ):
        zone_low, zone_high = levels[low_key], levels[high_key]
        if np.isfinite(zone_low) and np.isfinite(zone_high):
            figure.add_hrect(y0=zone_low, y1=zone_high, fillcolor=color, line_width=0, layer="below", name=label, annotation_text=label, annotation_position="top left", row=1, col=1)
    figure.add_hline(y=current_price, line_color="#263b39", line_dash="dash", line_width=1.5, annotation_text=f"Current {_number(current_price)}", annotation_position="top right", row=1, col=1)

    volume_colors = ["#4c9b76" if close >= open_price else "#d76c54" for open_price, close in zip(chart_data["Open"], chart_data["Close"])]
    figure.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", marker_color=volume_colors, opacity=0.7), row=2, col=1)

    histogram_colors = ["#4c9b76" if value >= 0 else "#d76c54" for value in chart_data["MACDHist"]]
    figure.add_trace(go.Bar(x=chart_data.index, y=chart_data["MACDHist"], name="Histogram", marker_color=histogram_colors), row=3, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MACD"], name="MACD", line={"color": "#356a9a"}), row=3, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MACDSignal"], name="Signal", line={"color": "#e8793e"}), row=3, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["RSI"], name="RSI", line={"color": "#655c49"}), row=4, col=1)
    figure.add_hline(y=70, line_dash="dot", line_color="#d76c54", row=4, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="#4c9b76", row=4, col=1)
    for column, label, color in (("ADX", "ADX", "#655c49"), ("PlusDI", "+DI", "#4c9b76"), ("MinusDI", "-DI", "#d76c54")):
        figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data[column], name=label, line={"color": color}), row=5, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["StochK"], name="%K", line={"color": "#356a9a"}), row=6, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["StochD"], name="%D", line={"color": "#e8793e"}), row=6, col=1)
    figure.add_hline(y=80, line_dash="dot", line_color="#d76c54", row=6, col=1)
    figure.add_hline(y=20, line_dash="dot", line_color="#4c9b76", row=6, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["OBV"], name="OBV", line={"color": "#356a9a"}), row=7, col=1)
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["OBVSignal"], name="OBV 20-session average", line={"color": "#e8793e"}), row=7, col=1)

    panel_rows = {"price": 1, "macd": 3, "rsi": 4, "adx": 5, "stochastic": 6}
    event_colors = {"bullish": "#2a8a5b", "bearish": "#c54e42", "neutral": "#77818a"}
    grouped_events: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for event in detect_signal_events(chart_data):
        key = (event["panel"], event["label"], event["tone"])
        grouped_events.setdefault(key, []).append(event)
    for (panel, label, tone), points in grouped_events.items():
        figure.add_trace(
            go.Scatter(
                x=[point["index"] for point in points],
                y=[point["value"] for point in points],
                customdata=[point["price"] for point in points],
                text=[point["label"] for point in points],
                mode="markers",
                name=f"{label} markers" if panel == "price" else f"{label} ({panel})",
                marker={"symbol": "circle", "size": 9, "color": event_colors[tone], "line": {"color": "#ffffff", "width": 1.2}},
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>Price: %{customdata:.2f}<extra></extra>",
                meta={"marker_category": points[0]["category"], "marker_panel": panel},
                showlegend=panel == "price",
            ),
            row=panel_rows[panel],
            col=1,
        )
    figure.update_layout(
        template="plotly_white",
        height=1030,
        margin={"l": 55, "r": 35, "t": 190, "b": 40},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.16,
            "xanchor": "left",
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#dce2dc",
            "borderwidth": 1,
            "groupclick": "togglegroup",
            "tracegroupgap": 8,
        },
        font={"family": "Arial, sans-serif", "color": "#25312f"},
    )
    figure.update_xaxes(rangebreaks=trading_xaxis_breaks(chart_data.index))
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    figure.update_yaxes(title_text="MACD", row=3, col=1)
    figure.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
    figure.update_yaxes(title_text="Index", row=5, col=1)
    figure.update_yaxes(title_text="%K / %D", range=[0, 100], row=6, col=1)
    figure.update_yaxes(title_text="Volume flow", row=7, col=1)
    return figure.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})


def build_report(
    symbol: str,
    label: str,
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    findings: list[dict[str, str]],
    levels: dict[str, float],
    output_path: Path,
) -> None:
    close = float(indicators["Close"].iloc[-1])
    previous = float(indicators["Close"].iloc[-2])
    change_pct = (close / previous - 1) * 100
    bullish = sum(item["tone"] == "bullish" for item in findings)
    bearish = sum(item["tone"] == "bearish" for item in findings)
    neutral = len(findings) - bullish - bearish
    signal_class = "positive" if bullish > bearish else "negative" if bearish > bullish else "neutral"
    table_rows = "\n".join(
        f'<tr><td>{html.escape(item["method"])} '
        f'<button class="help-button" type="button" aria-label="Help: {html.escape(item["method"], quote=True)}" '
        f'data-method="{html.escape(item["method"], quote=True)}" data-help="{html.escape(METHOD_HELP[item["method"]], quote=True)}" '
        f'aria-haspopup="dialog" aria-controls="methodHelp" title="About this method">i</button></td>'
        f'<td><span class="tag signal-{item["tone"]}">{html.escape(item["signal"])}</span></td>'
        f'<td>{html.escape(item["detail"])}</td></tr>'
        for item in findings
    )
    marker_options = "\n".join(
        f'<label><input class="marker-filter" type="checkbox" value="{html.escape(category, quote=True)}" checked> {html.escape(category)}</label>'
        for category in MARKER_CATEGORIES
    )
    zone_definitions = (
        ("support1", "Support zone 1", "Support zone low", "Support zone high"),
        ("support2", "Support zone 2", "Second support zone low", "Second support zone high"),
        ("support3", "Support zone 3", "Third support zone low", "Third support zone high"),
        ("resistance1", "Resistance zone 1", "Resistance zone low", "Resistance zone high"),
        ("resistance2", "Resistance zone 2", "Second resistance zone low", "Second resistance zone high"),
        ("resistance3", "Resistance zone 3", "Third resistance zone low", "Third resistance zone high"),
    )
    zone_option_rows = []
    for key, zone_label, low_key, high_key in zone_definitions:
        available = np.isfinite(levels[low_key]) and np.isfinite(levels[high_key])
        checked = " checked" if available else ""
        disabled = "" if available else " disabled"
        unavailable = "" if available else " (unavailable)"
        zone_option_rows.append(
            f'<label><input class="zone-filter" type="checkbox" value="{key}" data-zone-name="{zone_label}"{checked}{disabled}> {zone_label}{unavailable}</label>'
        )
    zone_options = "\n".join(zone_option_rows)
    chart_view_options = "\n".join(
        f'<option value="{key}">{html.escape(view["label"])}</option>'
        for key, view in CHART_VIEWS.items()
    )
    pivot_values = sorted(levels[key] for key in ("S2", "S1", "Pivot", "R1", "R2") if levels[key] > close)
    first_reference = pivot_values[0] if pivot_values else float("nan")
    second_reference = pivot_values[1] if len(pivot_values) > 1 else float("nan")
    price_targets = (
        f"Nearest overhead pivot reference: {_number(first_reference)}; next reference: {_number(second_reference)}. "
        f"Support zone 1: {_number(levels.get('Support zone low'))}–{_number(levels.get('Support zone high'))}; "
        f"resistance zone 1: {_number(levels.get('Resistance zone low'))}–{_number(levels.get('Resistance zone high'))}. "
        f"Support zone 2: {_number(levels.get('Second support zone low'))}–{_number(levels.get('Second support zone high'))}; "
        f"resistance zone 2: {_number(levels.get('Second resistance zone low'))}–{_number(levels.get('Second resistance zone high'))}. "
        f"Support zone 3: {_number(levels.get('Third support zone low'))}–{_number(levels.get('Third support zone high'))}; "
        f"resistance zone 3: {_number(levels.get('Third resistance zone low'))}–{_number(levels.get('Third resistance zone high'))}. "
        f"Reference levels: S1 {_number(levels.get('S1'))}, R1 {_number(levels.get('R1'))}; "
        f"Fibonacci 38.2% {_number(levels.get('Fib 38.2%'))}, 50% {_number(levels.get('Fib 50%'))}, "
        f"61.8% {_number(levels.get('Fib 61.8%'))}. "
        f"Illustrative five-session ATR envelope: {_number(levels.get('ATR lower'))} to {_number(levels.get('ATR upper'))}."
    )
    chart = create_chart(indicators, levels, symbol)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(label)} ({html.escape(symbol)}) | Technical Analysis</title>
  <style>
    :root {{ color-scheme: light; --ink:#25312f; --muted:#66716c; --line:#dce2dc; --paper:#f5f6f2; --white:#fff; --green:#286d50; --red:#a74736; --accent:#df7545; --surface-muted:#edf0eb; --summary:#e9eee8; --detail:#495650; }}
    body[data-theme="dark"] {{ color-scheme:dark; --ink:#e4e9e6; --muted:#a6b1ac; --line:#46524e; --paper:#171d20; --white:#222a2e; --green:#8bd4a5; --red:#f09a89; --accent:#ef9b70; --surface-muted:#303a3e; --summary:#263533; --detail:#c1cbc6; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.55 Arial,sans-serif }}
    main {{ max-width:1200px; margin:0 auto; padding:36px 28px 60px }} header {{ border-bottom:2px solid var(--ink); padding-bottom:20px; display:flex; justify-content:space-between; gap:20px; align-items:end }}
    h1 {{ font:600 34px/1.1 Georgia,serif; margin:0 0 6px }} h2 {{ font:600 22px Georgia,serif; margin:32px 0 12px }} p {{ margin:8px 0 }} .muted {{ color:var(--muted) }}
    .quote {{ text-align:right; white-space:nowrap }} .quote strong {{ font-size:27px; display:block }} .positive {{ color:var(--green) }} .negative {{ color:var(--red) }}
    .summary {{ margin:22px 0; padding:16px 18px; background:var(--summary); border-left:4px solid var(--accent) }} .chart {{ background:var(--white); border:1px solid var(--line); padding:8px; }}
    .table-wrap {{ overflow-x:auto; background:var(--white); border:1px solid var(--line) }} table {{ width:100%; border-collapse:collapse; min-width:760px }} th,td {{ text-align:left; vertical-align:top; padding:12px 14px; border-bottom:1px solid var(--line) }} th {{ background:var(--surface-muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em }}
    .tag {{ display:inline-block; padding:2px 7px; border:1px solid #cbd3cc; border-radius:3px; white-space:nowrap }}
    .signal-bullish {{ color:#1f6543; background:#e1f0e5; border-color:#a8cbb1 }} .signal-bearish {{ color:#923f2e; background:#f9e6df; border-color:#e5b5a7 }} .signal-neutral {{ color:#59636d; background:#eef0f2; border-color:#cbd1d6 }}
    body[data-theme="dark"] .signal-bullish {{ color:#a6e4b8; background:#253b30; border-color:#466e55 }} body[data-theme="dark"] .signal-bearish {{ color:#ffb4a5; background:#452e2b; border-color:#805149 }} body[data-theme="dark"] .signal-neutral {{ color:#d2d8dc; background:#343c41; border-color:#59636b }}
    .header-actions {{ display:flex; align-items:center; gap:18px }} .theme-toggle {{ border:1px solid var(--line); border-radius:3px; padding:8px 11px; background:var(--white); color:var(--ink); font:600 13px Arial,sans-serif; cursor:pointer }} .theme-toggle:hover,.theme-toggle:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px }}
    .chart-heading {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-top:32px }} .chart-heading h2 {{ margin:0 0 12px }} .chart-controls {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px }} .chart-picker-label {{ color:var(--muted); font-size:13px }} .chart-view-select {{ max-width:180px; border:1px solid var(--line); border-radius:3px; padding:7px 9px; background:var(--white); color:var(--ink); font:13px Arial,sans-serif }} .open-chart-button {{ border:1px solid var(--line); border-radius:3px; padding:8px 10px; background:var(--white); color:var(--ink); font:600 13px Arial,sans-serif; cursor:pointer }} .open-chart-button:hover,.open-chart-button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px }} .marker-picker,.zone-picker {{ position:relative; z-index:5; margin-bottom:12px }} .marker-picker summary,.zone-picker summary {{ cursor:pointer; border:1px solid var(--line); border-radius:3px; padding:7px 10px; background:var(--white); color:var(--ink); font-size:13px; list-style:none }} .marker-picker summary::-webkit-details-marker,.zone-picker summary::-webkit-details-marker {{ display:none }} .marker-picker summary::after,.zone-picker summary::after {{ content:" ▾"; color:var(--muted) }} .marker-picker[open] summary::after,.zone-picker[open] summary::after {{ content:" ▴" }}
    .marker-options {{ position:absolute; top:calc(100% + 5px); right:0; min-width:220px; display:grid; gap:8px; padding:12px; background:var(--white); color:var(--ink); border:1px solid var(--line); box-shadow:0 8px 24px #192a2633 }} .marker-options label {{ display:flex; align-items:center; gap:8px; white-space:nowrap; cursor:pointer; font-size:13px }} .marker-options input {{ accent-color:#356a9a }}
    .help-button {{ width:21px; height:21px; margin-left:5px; padding:0; border:1px solid #9aa7a0; border-radius:50%; background:var(--white); color:var(--ink); font:600 12px/19px Arial,sans-serif; cursor:pointer; vertical-align:middle }} .help-button:hover,.help-button:focus-visible {{ background:var(--surface-muted); outline:2px solid #557568; outline-offset:1px }}
    .zone-key {{ display:inline-block; width:24px; height:12px; margin:0 5px 0 14px; vertical-align:middle }} .zone-support {{ background:rgba(145,190,137,.5) }} .zone-resistance {{ background:rgba(230,145,91,.5) }} .price-key {{ display:inline-block; height:0; width:24px; border-top:2px dashed #263b39; vertical-align:middle }}
    dialog {{ max-width:520px; width:calc(100% - 32px); border:1px solid var(--line); padding:22px 24px; color:var(--ink); background:var(--white); box-shadow:0 16px 48px #192a2633 }} dialog::backdrop {{ background:#15231f88 }} dialog h3 {{ font:600 21px Georgia,serif; margin:0 32px 12px 0 }} dialog p {{ color:var(--detail) }} .dialog-close {{ float:right; border:1px solid var(--line); border-radius:3px; padding:6px 10px; color:var(--ink); background:var(--white); cursor:pointer }}
    .disclaimer {{ margin-top:26px; padding-top:14px; border-top:1px solid var(--line); font-size:13px; color:var(--muted) }}
    #backToReport {{ display:none; color:var(--ink); text-decoration:none; border:1px solid var(--line); border-radius:3px; padding:7px 10px; background:var(--white); font-size:13px }}
    body.separate-chart-view main {{ max-width:none; padding:18px 22px 28px }} body.separate-chart-view main > :not(.chart-heading):not(.chart):not(#backToReport) {{ display:none }} body.separate-chart-view #backToReport {{ display:inline-block }} body.separate-chart-view .chart-heading {{ margin-top:12px }} body.separate-chart-view .chart-heading h2 {{ margin:0 }} body.separate-chart-view .chart {{ padding:0 }}
    @media(max-width:650px) {{ main {{ padding:22px 14px 40px }} header {{ align-items:start; flex-direction:column }} .header-actions {{ width:100%; justify-content:space-between }} .quote {{ text-align:left }} h1 {{ font-size:28px }} .chart-heading {{ align-items:flex-start; flex-direction:column }} .chart-controls {{ width:100%; justify-content:space-between }} .chart-view-select {{ max-width:145px }} .marker-options {{ right:0; min-width:205px }} .chart {{ margin:0 -8px; padding:0 }} }}
  </style>
</head>
<body><main data-symbol="{html.escape(symbol, quote=True)}">
  <header><div><p class="muted">TECHNICAL ANALYSIS REPORT</p><h1>{html.escape(label)}</h1><p class="muted">Yahoo Finance symbol: {html.escape(symbol)} · {html.escape(str(prices.index[0].date()))} to {html.escape(str(prices.index[-1].date()))}</p></div>
    <div class="header-actions"><button class="theme-toggle" id="themeToggle" type="button" aria-label="Switch to dark mode" aria-pressed="false">Dark mode</button><div class="quote"><strong>{_number(close)}</strong><span class="{signal_class}">{change_pct:+.2f}% latest session</span></div></div></header>
    <section class="summary"><strong>Signal balance: <span class="positive">{bullish} bullish</span> / <span class="negative">{bearish} bearish</span> / {neutral} neutral or mixed</strong><p>Color-coded readings are technical context, not a forecast. Mixed signals are common; review the individual methods and levels below.</p></section>
    <div class="chart-heading"><h2 id="chartHeading">Price and indicators</h2><div class="chart-controls"><details class="marker-picker"><summary>Price-chart markers</summary><div class="marker-options" role="group" aria-label="Select price-chart marker categories">{marker_options}</div></details><details class="zone-picker"><summary>S/R zones</summary><div class="marker-options" role="group" aria-label="Choose support and resistance zones">{zone_options}</div></details><label class="chart-picker-label" for="chartViewSelect">Open chart</label><select class="chart-view-select" id="chartViewSelect">{chart_view_options}</select><button class="open-chart-button" id="openChartView" type="button">Open separately</button></div></div><div class="chart">{chart}</div><p class="muted"><span class="zone-key zone-support"></span>Support zones <span class="zone-key zone-resistance"></span>Resistance zones <span class="zone-key price-key"></span>Current price</p><a id="backToReport" href="">Back to full report</a>
    <h2>Levels and scenarios</h2><p>{html.escape(price_targets)}</p><p class="muted">Dashed chart lines mark pivot and Fibonacci references; their exact values are listed above. They are not guaranteed support, resistance, or target prices.</p>
    <h2>{len(findings)}-method analysis</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>Reading</th><th>Details</th></tr></thead><tbody>{table_rows}</tbody></table></div>
    <dialog id="methodHelp" aria-labelledby="helpTitle"><button class="dialog-close" id="closeHelp" type="button">Close</button><h3 id="helpTitle"></h3><p id="helpText"></p></dialog>
    <script>
        const methodHelp = document.getElementById("methodHelp");
        document.querySelectorAll(".help-button").forEach((button) => button.addEventListener("click", () => {{
            document.getElementById("helpTitle").textContent = button.dataset.method;
            document.getElementById("helpText").textContent = button.dataset.help;
            methodHelp.showModal();
        }}));
        document.getElementById("closeHelp").addEventListener("click", () => methodHelp.close());

        const themeToggle = document.getElementById("themeToggle");
        function applyTheme(theme) {{
            const dark = theme === "dark";
            document.body.dataset.theme = theme;
            themeToggle.textContent = dark ? "Light mode" : "Dark mode";
            themeToggle.setAttribute("aria-label", `Switch to ${{dark ? "light" : "dark"}} mode`);
            themeToggle.setAttribute("aria-pressed", String(dark));
            document.querySelectorAll(".js-plotly-plot").forEach((graph) => {{
                const textColor = dark ? "#e4e9e6" : "#25312f";
                const gridColor = dark ? "#3b464a" : "#e5e9ed";
                const layout = {{
                    paper_bgcolor: dark ? "#222a2e" : "#ffffff",
                    plot_bgcolor: dark ? "#222a2e" : "#ffffff",
                    font: {{color: textColor}},
                    "legend.bgcolor": dark ? "rgba(34,42,46,0.96)" : "rgba(255,255,255,0.94)",
                }};
                Object.keys(graph.layout).filter((key) => /^[xy]axis\\d*$/.test(key)).forEach((axis) => {{
                    layout[`${{axis}}.gridcolor`] = gridColor;
                    layout[`${{axis}}.zerolinecolor`] = gridColor;
                    layout[`${{axis}}.color`] = textColor;
                }});
                (graph.layout.annotations || []).forEach((_, index) => {{ layout[`annotations[${{index}}].font.color`] = textColor; }});
                Plotly.relayout(graph, layout);
            }});
        }}
        let initialTheme = "light";
        try {{ if (localStorage.getItem("stock-report-theme") === "dark") initialTheme = "dark"; }} catch {{}}
        applyTheme(initialTheme);
        themeToggle.addEventListener("click", () => {{
            const nextTheme = document.body.dataset.theme === "dark" ? "light" : "dark";
            applyTheme(nextTheme);
            try {{ localStorage.setItem("stock-report-theme", nextTheme); }} catch {{}}
        }});

        const chartSettingsKey = `stock-chart-settings:${{document.querySelector("main").dataset.symbol}}`;
        let savedChartSettings = {{hiddenTraces: {{}}, markerCategories: {{}}, zoneVisibility: {{}}}};
        try {{
            const storedSettings = JSON.parse(localStorage.getItem(chartSettingsKey));
            if (storedSettings && typeof storedSettings === "object") savedChartSettings = storedSettings;
        }} catch {{}}
        if (!savedChartSettings.hiddenTraces || typeof savedChartSettings.hiddenTraces !== "object") savedChartSettings.hiddenTraces = {{}};
        if (!savedChartSettings.markerCategories || typeof savedChartSettings.markerCategories !== "object") savedChartSettings.markerCategories = {{}};
        if (!savedChartSettings.zoneVisibility || typeof savedChartSettings.zoneVisibility !== "object") savedChartSettings.zoneVisibility = {{}};
        const legacyZonesVisible = typeof savedChartSettings.zonesVisible === "boolean" ? savedChartSettings.zonesVisible : true;
        document.querySelectorAll(".marker-filter").forEach((checkbox) => {{
            if (typeof savedChartSettings.markerCategories[checkbox.value] === "boolean") {{
                checkbox.checked = savedChartSettings.markerCategories[checkbox.value];
            }}
        }});
        document.querySelectorAll(".zone-filter").forEach((checkbox) => {{
            if (typeof savedChartSettings.zoneVisibility[checkbox.value] === "boolean") checkbox.checked = savedChartSettings.zoneVisibility[checkbox.value];
            else if (!legacyZonesVisible) checkbox.checked = false;
        }});
        function traceSettingsId(trace) {{
            return trace.meta?.settingsKey || JSON.stringify([trace.name || "", trace.legendgroup || "", trace.xaxis || "", trace.yaxis || "", trace.meta?.marker_panel || "", trace.meta?.marker_category || ""]);
        }}
        function saveChartSettings() {{
            const graph = document.querySelector(".js-plotly-plot");
            if (!graph) return;
            const hiddenTraces = {{...savedChartSettings.hiddenTraces}};
            graph.data.forEach((trace) => {{
                const key = traceSettingsId(trace);
                if (trace.visible === false || trace.visible === "legendonly") hiddenTraces[key] = true;
                else delete hiddenTraces[key];
            }});
            const markerCategories = {{}};
            document.querySelectorAll(".marker-filter").forEach((checkbox) => {{ markerCategories[checkbox.value] = checkbox.checked; }});
            const zoneVisibility = {{}};
            document.querySelectorAll(".zone-filter").forEach((checkbox) => {{ zoneVisibility[checkbox.value] = checkbox.checked; }});
            savedChartSettings.hiddenTraces = hiddenTraces;
            savedChartSettings.markerCategories = markerCategories;
            savedChartSettings.zoneVisibility = zoneVisibility;
            try {{ localStorage.setItem(chartSettingsKey, JSON.stringify(savedChartSettings)); }} catch {{}}
        }}
        function applyPriceMarkerFilter(graph, checkbox) {{
            const traceIndexes = graph.data.map((trace, index) => ({{trace, index}}))
                .filter((item) => item.trace.meta?.marker_panel === "price" && item.trace.meta.marker_category === checkbox.value)
                .map((item) => item.index);
            if (traceIndexes.length) Plotly.restyle(graph, {{visible: checkbox.checked}}, traceIndexes);
        }}
        function applyZoneVisibility(graph) {{
            const update = {{}};
            const zoneVisible = (name) => {{
                const checkbox = Array.from(document.querySelectorAll(".zone-filter")).find((item) => item.dataset.zoneName === name);
                return checkbox ? checkbox.checked : true;
            }};
            (graph.layout.shapes || []).forEach((shape, index) => {{
                if (/^(Support|Resistance) zone [123]$/.test(shape.name || "")) update[`shapes[${{index}}].visible`] = zoneVisible(shape.name);
            }});
            (graph.layout.annotations || []).forEach((annotation, index) => {{
                if (/^(Support|Resistance) zone [123]$/.test(annotation.text || "")) update[`annotations[${{index}}].visible`] = zoneVisible(annotation.text);
            }});
            if (Object.keys(update).length) Plotly.relayout(graph, update);
        }}
        const chartGraph = document.querySelector(".js-plotly-plot");
        if (chartGraph) {{
            const hiddenIndexes = chartGraph.data.map((trace, index) => ({{trace, index}}))
                .filter((item) => savedChartSettings.hiddenTraces[traceSettingsId(item.trace)])
                .map((item) => item.index);
            if (hiddenIndexes.length) Plotly.restyle(chartGraph, {{visible: "legendonly"}}, hiddenIndexes);
            chartGraph.on("plotly_legendclick", () => window.setTimeout(saveChartSettings, 0));
            chartGraph.on("plotly_legenddoubleclick", () => window.setTimeout(saveChartSettings, 0));
            applyZoneVisibility(chartGraph);
            document.querySelectorAll(".zone-filter").forEach((checkbox) => {{
                checkbox.addEventListener("change", () => {{
                    savedChartSettings.zoneVisibility[checkbox.value] = checkbox.checked;
                    applyZoneVisibility(chartGraph);
                    saveChartSettings();
                }});
            }});
            document.querySelectorAll(".marker-filter").forEach((checkbox) => {{
                applyPriceMarkerFilter(chartGraph, checkbox);
                checkbox.addEventListener("change", () => applyPriceMarkerFilter(chartGraph, checkbox));
                checkbox.addEventListener("change", saveChartSettings);
            }});

            const chartViewOptions = {json.dumps(CHART_VIEWS, separators=(',', ':'))};
            document.getElementById("openChartView").addEventListener("click", () => {{
                const target = new URL(window.location.href);
                target.searchParams.set("chartView", document.getElementById("chartViewSelect").value);
                window.open(target.href, "_blank");
            }});
            const requestedChart = new URLSearchParams(window.location.search).get("chartView");
            document.querySelector(".zone-picker").hidden = Boolean(requestedChart && requestedChart !== "price");
            if (chartGraph && chartViewOptions[requestedChart]) {{
                const view = chartViewOptions[requestedChart];
                const axisSuffix = view.yaxis === "y" ? "" : view.yaxis.slice(1);
                const sourceXAxis = chartGraph.layout[view.xaxis] || {{}};
                const sourceYAxis = chartGraph.layout[`yaxis${{axisSuffix}}`] || {{}};
                const selectedTraces = chartGraph.data
                    .filter((trace) => (trace.yaxis || "y") === view.yaxis)
                    .map((trace) => ({{
                        ...trace,
                        xaxis: "x",
                        yaxis: "y",
                        meta: {{...(trace.meta || {{}}), settingsKey: traceSettingsId(trace)}},
                    }}));
                const axisProperties = ["type", "range", "title", "showgrid", "gridcolor", "zeroline", "zerolinecolor", "showline", "linecolor", "tickformat", "tickangle", "tickfont", "tickcolor", "ticklen", "ticks", "side", "rangebreaks", "showticklabels", "fixedrange"];
                const standaloneXAxis = {{}};
                const standaloneYAxis = {{}};
                axisProperties.forEach((key) => {{
                    if (sourceXAxis[key] !== undefined) standaloneXAxis[key] = sourceXAxis[key];
                    if (sourceYAxis[key] !== undefined) standaloneYAxis[key] = sourceYAxis[key];
                }});
                standaloneXAxis.domain = [0, 1];
                standaloneXAxis.anchor = "y";
                standaloneXAxis.showticklabels = true;
                standaloneXAxis.rangeslider = {{visible: false}};
                standaloneYAxis.domain = [0, 1];
                standaloneYAxis.anchor = "x";
                const sourceAnnotations = (chartGraph.layout.annotations || []).filter((annotation) => annotation.yref === view.yaxis || annotation.yref === `${{view.yaxis}} domain`).map((annotation) => ({{
                    ...annotation,
                    xref: annotation.xref === `${{view.xaxis}} domain` ? "x domain" : "x",
                    yref: annotation.yref === `${{view.yaxis}} domain` ? "y domain" : "y",
                }}));
                const sourceShapes = (chartGraph.layout.shapes || []).filter((shape) => shape.yref === view.yaxis || shape.yref === `${{view.yaxis}} domain`).map((shape) => ({{
                    ...shape,
                    xref: shape.xref === `${{view.xaxis}} domain` ? "x domain" : "x",
                    yref: shape.yref === `${{view.yaxis}} domain` ? "y domain" : "y",
                }}));
                const standaloneLayout = {{
                    template: chartGraph.layout.template,
                    paper_bgcolor: chartGraph.layout.paper_bgcolor,
                    plot_bgcolor: chartGraph.layout.plot_bgcolor,
                    font: chartGraph.layout.font,
                    hovermode: chartGraph.layout.hovermode,
                    showlegend: chartGraph.layout.showlegend,
                    legend: chartGraph.layout.legend,
                    margin: {{l: 65, r: 35, t: 70, b: 55}},
                    height: Math.max(620, window.innerHeight - 80),
                    xaxis: standaloneXAxis,
                    yaxis: standaloneYAxis,
                    annotations: sourceAnnotations,
                    shapes: sourceShapes,
                }};
                document.body.classList.add("separate-chart-view");
                document.getElementById("chartHeading").textContent = view.label;
                document.getElementById("chartViewSelect").value = requestedChart;
                document.querySelector(".marker-picker").hidden = requestedChart !== "price";
                const returnUrl = new URL(window.location.href);
                returnUrl.searchParams.delete("chartView");
                document.getElementById("backToReport").href = returnUrl.href;
                const resizeStandalone = () => Plotly.relayout(chartGraph, {{height: Math.max(620, window.innerHeight - 80)}});
                Plotly.react(chartGraph, selectedTraces, standaloneLayout, {{responsive: true, displaylogo: false}})
                    .then(() => applyTheme(document.body.dataset.theme || "light"));
                window.addEventListener("resize", resizeStandalone);
            }}
        }}
    </script>
  <footer class="disclaimer">Generated {generated}. Historical prices are adjusted by Yahoo Finance. Technical analysis is not investment advice. Data availability, symbol mapping, and indicator readings can be imperfect; verify them before making decisions.</footer>
</main></body></html>""",
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain an object at its root.")
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("Configuration must include a non-empty 'symbols' list.")
    provider = str(config.get("provider", "yahoo")).lower()
    if provider != "yahoo":
        raise ValueError("Only the Yahoo Finance provider is currently supported.")
    config["provider"] = provider
    analysis = config.setdefault("analysis", list(ANALYSIS_KEYS))
    if not isinstance(analysis, list) or not analysis:
        raise ValueError("Configuration 'analysis' must be a non-empty list of method keys.")
    unknown_methods = set(analysis).difference(SUPPORTED_ANALYSES)
    if unknown_methods:
        raise ValueError(f"Unknown analysis method(s): {', '.join(sorted(unknown_methods))}.")
    if len(set(analysis)) != len(analysis):
        raise ValueError("Configuration 'analysis' cannot contain duplicate methods.")
    config["analysis"] = analysis
    config.setdefault("period", "2y")
    config.setdefault("interval", "1d")
    config.setdefault("output_dir", "reports")
    return config


def run(config_path: Path, output_override: Path | None = None) -> list[Path]:
    config = load_config(config_path)
    output_dir = output_override or Path(config["output_dir"])
    generated_reports = []
    for instrument in config["symbols"]:
        symbol, label = resolve_symbol(instrument)
        prices = download_prices(symbol, str(config["period"]), str(config["interval"]))
        indicators = calculate_indicators(prices)
        findings, levels = analyze(indicators, config["analysis"])
        safe_symbol = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol)
        report_path = output_dir / f"{safe_symbol}_technical_analysis.html"
        build_report(symbol, label, prices, indicators, findings, levels, report_path)
        generated_reports.append(report_path)
        print(f"Created {report_path.resolve()}")
    return generated_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate technical-analysis HTML reports for configured equities.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML configuration file (default: config.yaml)")
    parser.add_argument("--output", type=Path, help="Override the configured output directory")
    arguments = parser.parse_args()
    try:
        run(arguments.config, arguments.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()