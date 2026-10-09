import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stock_analyzer import METHODS, analyze, build_report, calculate_indicators, detect_signal_events, trading_xaxis_breaks


def sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    close = 100 + np.arange(len(dates)) * 0.15 + np.sin(np.arange(len(dates)) / 5) * 2
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(len(dates), 1_000_000),
        },
        index=dates,
    )


class StockAnalyzerTests(unittest.TestCase):
    def test_chart_breaks_skip_weekends_and_missing_weekdays(self):
        prices = sample_prices().drop(pd.Timestamp("2024-02-19"))

        breaks = trading_xaxis_breaks(prices.index)

        self.assertEqual(breaks[0]["bounds"], ["sat", "mon"])
        self.assertIn("2024-02-19", breaks[1]["values"])

    def test_calculates_twelve_method_findings_and_finite_levels(self):
        indicators = calculate_indicators(sample_prices())
        findings, levels = analyze(indicators)

        self.assertEqual(len(findings), 12)
        self.assertEqual([item["method"] for item in findings], METHODS)
        self.assertTrue(np.isfinite(indicators["RSI"].iloc[-1]))
        self.assertTrue(np.isfinite(indicators["ADX"].iloc[-1]))
        self.assertTrue(np.isfinite(indicators["StochK"].iloc[-1]))
        self.assertTrue(np.isfinite(indicators["OBVSignal"].iloc[-1]))
        self.assertTrue(np.isfinite(indicators["DonchianHigh"].iloc[-1]))
        self.assertTrue(np.isfinite(levels["S1"]))
        self.assertTrue(np.isfinite(levels["Fib 50%"]))
        latest_close = indicators["Close"].iloc[-1]
        self.assertLess(levels["Support zone high"], latest_close)
        self.assertGreater(levels["Resistance zone low"], latest_close)
        zone_prices = sample_prices()
        zone_prices.loc[zone_prices.index[-10:], ["Open", "High", "Low", "Close"]] -= 20
        zone_prices.loc[[zone_prices.index[index] for index in (150, 185, 220)], "Low"] = [113, 105, 97]
        _, zone_levels = analyze(calculate_indicators(zone_prices))
        self.assertTrue(np.isfinite(zone_levels["Second support zone low"]))
        self.assertTrue(np.isfinite(zone_levels["Second resistance zone low"]))
        self.assertTrue(np.isfinite(zone_levels["Third support zone low"]))
        self.assertTrue(np.isfinite(zone_levels["Third resistance zone low"]))
        self.assertLess(zone_levels["Second support zone high"], zone_levels["Support zone low"])
        self.assertLess(zone_levels["Third support zone high"], zone_levels["Second support zone low"])
        self.assertGreater(zone_levels["Second resistance zone low"], zone_levels["Resistance zone high"])
        self.assertGreater(zone_levels["Third resistance zone low"], zone_levels["Second resistance zone high"])
        self.assertTrue(all(item["tone"] in {"bullish", "bearish", "neutral"} for item in findings))

    def test_can_select_a_subset_of_analysis_methods(self):
        indicators = calculate_indicators(sample_prices())
        findings, _ = analyze(indicators, ["rsi", "fibonacci"])

        self.assertEqual([finding["method"] for finding in findings], [METHODS[2], METHODS[7]])

    def test_detects_strong_indicator_events(self):
        indicators = calculate_indicators(sample_prices())
        previous, latest = indicators.index[-2:]
        indicators.loc[previous, ["RSI", "MACD", "MACDSignal", "SMA50", "SMA200", "StochK", "StochD"]] = [35, -1, 0, 99, 100, 10, 12]
        indicators.loc[latest, ["RSI", "MACD", "MACDSignal", "SMA50", "SMA200", "StochK", "StochD"]] = [29, 1, 0, 101, 100, 15, 13]
        indicators.loc[latest, "DonchianHigh"] = indicators.loc[latest, "Close"] - 1

        events = detect_signal_events(indicators)
        labels = {event["label"] for event in events}
        price_rsi_event = next(event for event in events if event["label"] == "RSI oversold" and event["panel"] == "price")
        rsi_panel_event = next(event for event in events if event["label"] == "RSI oversold" and event["panel"] == "rsi")

        self.assertIn("RSI oversold", labels)
        self.assertIn("MACD bullish cross", labels)
        self.assertIn("Golden cross", labels)
        self.assertIn("Stochastic bullish cross", labels)
        self.assertIn("Donchian upside breakout", labels)
        self.assertEqual(price_rsi_event["value"], indicators.loc[latest, "Close"])
        self.assertEqual(rsi_panel_event["value"], indicators.loc[latest, "RSI"])

    def test_builds_html_report_with_chart_and_analysis(self):
        prices = sample_prices()
        prices.loc[prices.index[-10:], ["Open", "High", "Low", "Close"]] -= 20
        prices.loc[[prices.index[index] for index in (150, 185, 220)], "Low"] = [113, 105, 97]
        indicators = calculate_indicators(prices)
        findings, levels = analyze(indicators)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            build_report("TEST", "Example Share", prices, indicators, findings, levels, output)
            report = output.read_text(encoding="utf-8")

        self.assertIn("Example Share", report)
        self.assertIn("Plotly.newPlot", report)
        self.assertIn("12-method analysis", report)
        self.assertIn("Fibonacci retracement levels", report)
        self.assertIn("Ichimoku cloud trend", report)
        self.assertIn("data-help=", report)
        self.assertIn("signal-bullish", report)
        self.assertIn("background:#eef0f2", report)
        self.assertIn('"y":1.16', report)
        self.assertIn('"t":190', report)
        self.assertIn("Support zone", report)
        self.assertIn('class="zone-picker"', report)
        self.assertIn('value="support3"', report)
        self.assertIn('value="resistance3"', report)
        self.assertIn('"name":"Support zone 2"', report)
        self.assertIn('"name":"Support zone 3"', report)
        self.assertIn('"name":"Resistance zone 2"', report)
        self.assertIn('"name":"Resistance zone 3"', report)
        self.assertIn("Support zone 2:", report)
        self.assertIn("resistance zone 2:", report)
        self.assertIn("Support zone 3:", report)
        self.assertIn("resistance zone 3:", report)
        self.assertIn("applyZoneVisibility(chartGraph)", report)
        self.assertIn("savedChartSettings.zoneVisibility", report)
        self.assertIn("annotations[${index}].visible", report)
        self.assertIn("Current", report)
        self.assertIn('"name":"Volume"', report)
        self.assertIn('"legendgroup":"donchian"', report)
        self.assertIn('"legendgroup":"ichimoku"', report)
        self.assertIn('"groupclick":"togglegroup"', report)
        self.assertIn('id="themeToggle"', report)
        self.assertIn('body[data-theme="dark"]', report)
        self.assertIn("Plotly.relayout(graph, layout)", report)
        self.assertIn("Price-chart markers", report)
        self.assertIn('value="RSI thresholds" checked', report)
        self.assertIn('value="MACD crosses" checked', report)
        self.assertIn('item.trace.meta?.marker_panel === "price"', report)
        self.assertIn('id="chartViewSelect"', report)
        self.assertIn('id="openChartView"', report)
        self.assertIn('"volume":{"label":"Volume"', report)
        self.assertIn('window.open(target.href, "_blank")', report)
        self.assertIn("Plotly.react(chartGraph, selectedTraces, standaloneLayout", report)
        self.assertIn('const axisProperties = ["type", "range", "title"', report)
        self.assertIn("stock-chart-settings:", report)
        self.assertIn("plotly_legendclick", report)
        self.assertIn("plotly_legenddoubleclick", report)
        self.assertIn("hiddenTraces", report)
        self.assertIn("markerCategories", report)
        self.assertIn('"rangebreaks"', report)


if __name__ == "__main__":
    unittest.main()