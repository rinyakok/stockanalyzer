"""Local web dashboard for interactive stock analysis reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from stock_analyzer import analyze, build_report, calculate_indicators, download_prices, resolve_symbol


app = Flask(__name__)
IDENTIFIER_TYPES = {"ticker", "name", "isin"}


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.post("/api/analyze")
def analyze_instrument():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with an identifier type and value."), 400

    identifier_type = payload.get("type")
    value = payload.get("value")
    if identifier_type not in IDENTIFIER_TYPES:
        return jsonify(error="Choose ticker, company name, or ISIN."), 400
    if not isinstance(value, str) or not value.strip():
        return jsonify(error="Enter a ticker, company name, or ISIN."), 400
    value = value.strip()
    if len(value) > 120:
        return jsonify(error="Search text must be 120 characters or fewer."), 400

    try:
        symbol, label = resolve_symbol({identifier_type: value})
        prices = download_prices(symbol, period="2y", interval="1d")
        indicators = calculate_indicators(prices)
        findings, levels = analyze(indicators)
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.html"
            build_report(symbol, label, prices, indicators, findings, levels, report_path)
            report_html = report_path.read_text(encoding="utf-8")
    except ValueError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Could not analyze %s", value)
        return jsonify(error="Could not fetch or analyze this instrument. Check the identifier and try again."), 502

    return jsonify(symbol=symbol, label=label, report_html=report_html)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)