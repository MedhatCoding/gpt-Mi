"""اختبارات تقويم جلسات EGX المتوقعة (بوابة الحداثة والنشر). التشغيل:

    python tests/test_data_freshness.py   أو   pytest tests/test_data_freshness.py

ملحوظة: نحدد EGX_DATA_DIR قبل استيراد أي شيء من src، لأن المسارات بتتحدد وقت الاستيراد.
يُشغَّل ضمن خطوة التستات في egx-daily-report.yml (python tests/test_data_freshness.py).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="egx_freshness_"))
os.environ["EGX_DATA_DIR"] = str(TMP)

from src import config as C  # noqa: E402
from src.pipeline import expected_last_session  # noqa: E402

CFG = C.load_config()
CAIRO = ZoneInfo("Africa/Cairo")


def test_expected_session_after_close_same_day():
    # الأحد 2026-11-01 الساعة 16:00 (بعد الإغلاق) => الجلسة نفسها
    now = datetime(2026, 11, 1, 16, 0, tzinfo=CAIRO)
    assert expected_last_session(now, CFG) == date(2026, 11, 1)


def test_expected_session_before_close_returns_previous():
    # الأحد 2026-11-01 الساعة 10:00 (قبل الإغلاق) => آخر جلسة متاحة هي الخميس
    now = datetime(2026, 11, 1, 10, 0, tzinfo=CAIRO)
    assert expected_last_session(now, CFG) == date(2026, 10, 29)


def test_expected_session_skips_weekend():
    # الجمعة 2026-10-30 بعد الإغلاق => آخر جلسة هي الخميس 2026-10-29
    now = datetime(2026, 10, 30, 16, 0, tzinfo=CAIRO)
    assert expected_last_session(now, CFG) == date(2026, 10, 29)


def test_expected_session_respects_registered_holiday():
    # الخميس 2026-11-05 مسجل إجازة => آخر جلسة هي الأربعاء 2026-11-04
    (TMP / "holidays.txt").write_text("# إجازات فترية (تجريبي)\n2026-11-05\n", encoding="utf-8")
    now = datetime(2026, 11, 5, 16, 0, tzinfo=CAIRO)
    assert expected_last_session(now, CFG) == date(2026, 11, 4)


def test_expected_session_walks_back_over_long_holiday_gap():
    # إجازة ممتدة أسبوعًا (الاثنين-الجمعة) => أول جلسة سابقة هي الأحد 2026-11-08
    lines = "\n".join(f"2026-11-{d:02d}" for d in range(9, 14))
    (TMP / "holidays.txt").write_text(lines + "\n", encoding="utf-8")
    now = datetime(2026, 11, 13, 16, 0, tzinfo=CAIRO)
    assert expected_last_session(now, CFG) == date(2026, 11, 8)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
