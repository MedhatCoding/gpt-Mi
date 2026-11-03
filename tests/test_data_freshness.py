"""Regression tests for expected EGX session detection."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src import data_fetch


class ExpectedSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prices = Path(self.tmp.name) / "prices"
        self.prices.mkdir()
        self.cfg = {"market": {
            "timezone": "Africa/Cairo",
            "trading_weekdays": [6, 0, 1, 2, 3],
            "session_close_hour": 15,
        }}
        self.patch = patch.object(data_fetch, "PRICES", self.prices)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_before_close_uses_previous_trading_session(self):
        now = datetime(2026, 11, 4, 9, 0, tzinfo=ZoneInfo("Africa/Cairo"))
        self.assertEqual(data_fetch.expected_latest_session(now, self.cfg).isoformat(), "2026-11-03")

    def test_after_close_uses_current_trading_session(self):
        now = datetime(2026, 11, 4, 15, 30, tzinfo=ZoneInfo("Africa/Cairo"))
        self.assertEqual(data_fetch.expected_latest_session(now, self.cfg).isoformat(), "2026-11-04")

    def test_skips_weekends_and_configured_holidays(self):
        (self.prices.parent / "holidays.txt").write_text("2026-11-04\\n", encoding="utf-8")
        now = datetime(2026, 11, 5, 9, 0, tzinfo=ZoneInfo("Africa/Cairo"))
        self.assertEqual(data_fetch.expected_latest_session(now, self.cfg).isoformat(), "2026-11-03")


if __name__ == "__main__":
    unittest.main()
