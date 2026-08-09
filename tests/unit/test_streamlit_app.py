from datetime import datetime, timedelta, timezone

from src.web_app.streamlit_app import _format_days_left


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_format_days_left_reports_whole_days_remaining():
    deleted_at = _iso(datetime.now(timezone.utc) - timedelta(days=4))
    assert _format_days_left(deleted_at, retention_days=14) == "Expires in 10 days"


def test_format_days_left_uses_singular_for_one_day():
    deleted_at = _iso(datetime.now(timezone.utc) - timedelta(days=13, hours=1))
    assert _format_days_left(deleted_at, retention_days=14) == "Expires in 1 day"


def test_format_days_left_rounds_up_partial_days():
    # 13.9 days elapsed, 14-day window -> ~2.4 hours left, should round up to 1
    # day rather than truncate to 0 (which would misleadingly read as expired).
    deleted_at = _iso(datetime.now(timezone.utc) - timedelta(days=13, hours=21, minutes=30))
    assert _format_days_left(deleted_at, retention_days=14) == "Expires in 1 day"


def test_format_days_left_reports_pending_cleanup_when_window_elapsed():
    deleted_at = _iso(datetime.now(timezone.utc) - timedelta(days=20))
    assert _format_days_left(deleted_at, retention_days=14) == "Pending cleanup"


def test_format_days_left_handles_malformed_timestamp():
    assert _format_days_left("not-a-timestamp", retention_days=14) == "Expiration unknown"
