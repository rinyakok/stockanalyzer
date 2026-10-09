# Stock Technical Analysis Reports

A Python command-line program that downloads adjusted historical prices from Yahoo Finance, computes twelve technical-analysis methods, and writes an interactive HTML report for each configured share.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configure and run

Edit `config.yaml`. Each item in `symbols` can be a ticker or a mapping with a `ticker`, `name`, or `isin` field. Tickers are the most reliable identifiers. Name and ISIN entries use Yahoo Finance search and can require manual ticker correction if the result is ambiguous or unavailable.

```yaml
symbols:
  - ticker: AAPL
  - name: Microsoft Corporation
  - isin: US0378331005
provider: yahoo
period: 2y
interval: 1d
output_dir: reports
analysis: [sma, macd, rsi, bollinger, adx, atr, pivots, fibonacci, stochastic, obv, donchian, ichimoku]
```

The `analysis` list accepts any non-empty subset of those method keys, so reports can focus on only the indicators you use.

Each reading has a color-coded bullish, bearish, or neutral/mixed label and a small `i` button with an explanation of the method and common interpretations. The chart marks the latest close and approximate nearby support/resistance zones; the ranges are ATR-tolerant clusters of recent swing and reference levels, not exact boundaries.

Run the analysis from this directory:

```powershell
python stock_analyzer.py
```

Or specify a configuration and output directory:

```powershell
python stock_analyzer.py --config config.yaml --output reports
```

Each report is self-contained, including its interactive Plotly chart with price, volume, and indicator panels, and is written as `reports/<ticker>_technical_analysis.html`. Volume appears directly below price, and weekends plus missing weekdays are compressed out of the time axis. Choose any chart panel from the “Open chart” selector and open it in a separate tab for a larger view. Small circles mark detected strong events such as RSI threshold hits, MACD and moving-average crosses, stochastic crosses, and Donchian breakouts. Use the multi-select “Price-chart markers” menu to choose which signal categories appear on the price panel; indicator-panel marks remain independent. Indicator visibility and marker-category selections are saved in browser storage per ticker and restored when that report is opened again. A header button switches between light and dark mode. Clicking a Donchian or Ichimoku legend item toggles that full indicator group together.

## Included methods

1. 50/200-day simple moving-average trend
2. MACD momentum and signal-line comparison
3. 14-period RSI
4. 20-period Bollinger Bands and squeeze check
5. ADX and directional movement (+DI/-DI)
6. 14-period ATR and an illustrative five-session volatility envelope
7. Classic pivot support and resistance from the latest bar
8. Fibonacci retracements across the latest 126 bars
9. 14-session Stochastic oscillator (%K/%D)
10. On-Balance Volume compared with its 20-session average
11. 20-session Donchian channel breakout
12. Ichimoku cloud alignment

The report summarizes indicator readings, pivot and Fibonacci reference levels, trend averages, and an ATR-based scenario. It also identifies up to three support and resistance zones from clustered recent swing and reference levels; a zone is shown as unavailable when no separate cluster exists. Use the “S/R zones” multi-select to show or hide each band independently; selections are saved per ticker. These zones are not guaranteed targets or investment advice. Yahoo Finance symbol search and data availability vary by exchange and identifier.

## Tests

```powershell
python -m unittest
```