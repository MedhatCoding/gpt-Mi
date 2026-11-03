"""Offline regression tests for the EGX opening-price review guard."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Ensure repository root is importable when invoked as `python tests/...` in CI.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_daily


class OpeningReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.signals = Path(self.tmp.name)
        self.now = datetime(2026, 10, 28, 11, 0, tzinfo=timezone.utc)
        self.cfg = {"market": {"timezone": "Africa/Cairo", "trading_weekdays": [6, 0, 1, 2, 3], "session_close_hour": 14}}
        self.report_path = self.signals / "latest.json"
        self.report = {
            "generated_at": self.now.isoformat(),
            "data_date": "2026-10-27",
            "picks": [{
                "signal": "BUY", "code": "TEST", "name": "Test",
                "entry_low": 9, "entry_high": 11, "stop": 8, "t1": 13,
            }],
        }
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        self.state = {}
        self.state_patch = patch.multiple(
            run_daily.C,
            SIGNALS=self.signals,
            load_state=lambda: dict(self.state),
            save_state=self.save_state,
        )
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        self.env_patch = patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_CHAT_ID": "test-chat"
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def save_state(self, state):
        self.state.clear()
        self.state.update(state)

    @patch("run_daily.fetch_live_quotes")
    @patch("run_daily.send_message")
    def test_stale_report_is_rejected_without_quote_or_send(self, send, quotes):
        self.report["generated_at"] = "2026-10-27T08:00:00+00:00"
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        result = run_daily.opening_review(self.cfg, self.now)
        self.assertEqual(result["status"], "not_sent")
        quotes.assert_not_called()
        send.assert_not_called()

    @patch("run_daily.fetch_live_quotes")
    @patch("run_daily.send_message")
    def test_attempt_guard_prevents_duplicate_send(self, send, quotes):
        self.state["opening_review_attempt_date"] = self.now.date().isoformat()
        result = run_daily.opening_review(self.cfg, self.now)
        self.assertEqual(result["status"], "skipped")
        quotes.assert_not_called()
        send.assert_not_called()

    @patch("run_daily.fetch_live_quotes", return_value={
        "ok": True, "retrieved_at": "2026-10-28T11:00:00+00:00",
        "quotes": [{"code": "TEST", "price": 10, "change_percent": 0, "as_of": "2026-10-28T11:00:00+00:00"}],
    })
    @patch("run_daily.send_message")
    def test_attempt_is_persisted_before_telegram_send(self, send, quotes):
        def inspect_reserved_state(*args, **kwargs):
            self.assertEqual(self.state.get("opening_review_attempt_date"), self.now.date().isoformat())
            return 1
        send.side_effect = inspect_reserved_state
        result = run_daily.opening_review(self.cfg, self.now)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(self.state.get("last_opening_review_date"), self.now.date().isoformat())
        send.assert_called_once()


    def test_quote_timestamp_parser_accepts_iso_and_epoch(self):
        from src.live_quotes import parse_quote_timestamp
        iso = parse_quote_timestamp("2026-10-28T11:00:00+00:00")
        epoch = parse_quote_timestamp(iso.timestamp())
        self.assertEqual(iso, epoch)

    def test_quote_freshness_rejects_missing_stale_and_future_timestamps(self):
        from src.live_quotes import validate_quote_freshness
        now = datetime(2026, 10, 28, 11, 0, tzinfo=timezone.utc)
        base = {"ok": True, "quotes": [{"code": "T", "as_of": now.isoformat()}]}
        self.assertTrue(validate_quote_freshness(base, now)["ok"])
        stale = {"ok": True, "quotes": [{"code": "T", "as_of": "2026-10-28T10:00:00+00:00"}]}
        self.assertFalse(validate_quote_freshness(stale, now)["ok"])
        missing = {"ok": True, "quotes": [{"code": "T", "as_of": "unknown"}]}
        self.assertFalse(validate_quote_freshness(missing, now)["ok"])
        future = {"ok": True, "quotes": [{"code": "T", "as_of": "2026-10-28T12:00:00+00:00"}]}
        self.assertFalse(validate_quote_freshness(future, now)["ok"])


if __name__ == "__main__":
    unittest.main()
