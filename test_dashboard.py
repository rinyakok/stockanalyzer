import unittest
from pathlib import Path
from unittest import mock

import dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        dashboard.app.config.update(TESTING=True)
        self.client = dashboard.app.test_client()

    def test_serves_dashboard_shell(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Market Desk", response.data)
        self.assertIn(b"favorites", response.data)

    def test_hidden_empty_state_has_no_layout_display(self):
        with self.client.get("/static/dashboard.css") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"#emptyState[hidden] { display: none; }", response.data)

    def test_rejects_unknown_identifier_type(self):
        response = self.client.post("/api/analyze", json={"type": "exchange", "value": "XYZ"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("ticker, company name, or ISIN", response.json["error"])

    def test_returns_generated_report_for_valid_ticker(self):
        def write_report(symbol, label, prices, indicators, findings, levels, output_path):
            Path(output_path).write_text("<html><title>Mock report</title></html>", encoding="utf-8")

        with (
            mock.patch.object(dashboard, "resolve_symbol", return_value=("MSFT", "Microsoft")) as resolve,
            mock.patch.object(dashboard, "download_prices", return_value="prices") as download,
            mock.patch.object(dashboard, "calculate_indicators", return_value="indicators"),
            mock.patch.object(dashboard, "analyze", return_value=([], {})),
            mock.patch.object(dashboard, "build_report", side_effect=write_report),
        ):
            response = self.client.post("/api/analyze", json={"type": "ticker", "value": "msft"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["symbol"], "MSFT")
        self.assertIn("Mock report", response.json["report_html"])
        resolve.assert_called_once_with({"ticker": "msft"})
        download.assert_called_once_with("MSFT", period="2y", interval="1d")


if __name__ == "__main__":
    unittest.main()