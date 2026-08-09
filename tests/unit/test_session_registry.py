import sqlite3

import pytest

from src.workflow import state_management
from src.workflow.state_management import SessionRegistry, derive_session_title


@pytest.fixture
def registry(tmp_path) -> SessionRegistry:
    # Exercises the same SessionRegistry class the postgres backend uses (only
    # the placeholder-adaptation branch differs) — isolated to a temp file, no
    # real network/Supabase access.
    conn = sqlite3.connect(str(tmp_path / "test-sessions.db"))
    return SessionRegistry(conn, backend="sqlite")


def test_ensure_table_is_idempotent(registry):
    registry._ensure_table()  # second call must not raise
    assert registry.list_sessions() == []


def test_record_start_inserts_row(registry):
    registry.record_start("session-1", "My first session")
    sessions = registry.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "session-1"
    assert sessions[0]["title"] == "My first session"
    assert sessions[0]["turn_count"] == 1


def test_record_start_is_noop_on_duplicate_session_id(registry):
    registry.record_start("session-1", "Original title")
    original = registry.list_sessions()[0]

    registry.record_start("session-1", "A different title")
    after = registry.list_sessions()

    assert len(after) == 1
    assert after[0]["title"] == "Original title"
    assert after[0]["created_at"] == original["created_at"]


def test_record_turn_increments_count_and_updates_timestamp(monkeypatch, registry):
    timestamps = iter(["2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00"])
    monkeypatch.setattr(state_management, "_utcnow", lambda: next(timestamps))

    registry.record_start("session-1", "Title")
    registry.record_turn("session-1")

    session = registry.list_sessions()[0]
    assert session["turn_count"] == 2
    assert session["created_at"] == "2026-01-01T00:00:00+00:00"
    assert session["updated_at"] == "2026-01-01T00:05:00+00:00"


def test_list_sessions_orders_by_updated_at_desc(monkeypatch, registry):
    timestamps = iter(
        [
            "2026-01-01T00:00:00+00:00",  # session-a start
            "2026-01-01T00:01:00+00:00",  # session-b start
            "2026-01-01T00:02:00+00:00",  # session-a turn (now most recent)
        ]
    )
    monkeypatch.setattr(state_management, "_utcnow", lambda: next(timestamps))

    registry.record_start("session-a", "A")
    registry.record_start("session-b", "B")
    registry.record_turn("session-a")

    ordered_ids = [s["session_id"] for s in registry.list_sessions()]
    assert ordered_ids == ["session-a", "session-b"]


def test_list_sessions_excludes_archived_by_default(registry):
    registry.record_start("session-1", "Kept")
    registry.record_start("session-2", "Archived")
    registry.set_archived("session-2", True)

    ids = [s["session_id"] for s in registry.list_sessions()]
    assert ids == ["session-1"]

    all_ids = {s["session_id"]: s["archived"] for s in registry.list_sessions(include_archived=True)}
    assert all_ids == {"session-1": 0, "session-2": 1}


def test_set_archived_can_be_toggled_back(registry):
    registry.record_start("session-1", "Title")
    registry.set_archived("session-1", True)
    assert registry.list_sessions() == []

    registry.set_archived("session-1", False)
    assert [s["session_id"] for s in registry.list_sessions()] == ["session-1"]


def test_rename_session_updates_title(registry):
    registry.record_start("session-1", "Old title")
    registry.rename_session("session-1", "New title")
    assert registry.list_sessions()[0]["title"] == "New title"


def test_rename_session_normalization_matches_derive_session_title(registry):
    long_title = (
        "this is a very long user-typed title that should definitely get truncated "
        "at some point because it exceeds sixty characters"
    )
    registry.record_start("session-1", "Title")
    registry.rename_session("session-1", long_title)
    assert registry.list_sessions()[0]["title"] == derive_session_title(long_title)


def test_rename_session_is_noop_when_normalized_title_is_empty(registry):
    registry.record_start("session-1", "Original")
    registry.rename_session("session-1", "   ")
    assert registry.list_sessions()[0]["title"] == "Original"


def test_rename_session_does_not_change_updated_at(monkeypatch, registry):
    timestamps = iter(["2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00"])
    monkeypatch.setattr(state_management, "_utcnow", lambda: next(timestamps))

    registry.record_start("session-1", "Original")
    registry.rename_session("session-1", "Renamed")

    session = registry.list_sessions()[0]
    assert session["title"] == "Renamed"
    assert session["updated_at"] == "2026-01-01T00:00:00+00:00"


def test_rename_session_on_unknown_session_id_is_noop(registry):
    registry.rename_session("does-not-exist", "New title")
    assert registry.list_sessions(include_archived=True) == []


def test_hard_delete_session_removes_session_and_routes(registry):
    registry.record_start("session-1", "Title")
    registry.record_routes("session-1", 1, ["▶ Orchestrator started"])

    registry.hard_delete_session("session-1")

    assert registry.list_sessions(include_archived=True) == []
    assert registry.get_routes("session-1") == {}


def test_soft_delete_session_hides_it_without_removing_data(registry):
    registry.record_start("session-1", "Title")

    registry.soft_delete_session("session-1")

    assert registry.list_sessions() == []
    assert registry.list_sessions(include_archived=True) == []
    trashed = registry.list_trashed_sessions()
    assert len(trashed) == 1
    assert trashed[0]["session_id"] == "session-1"
    assert trashed[0]["deleted_at"] is not None


def test_restore_session_undoes_soft_delete(registry):
    registry.record_start("session-1", "Title")
    registry.soft_delete_session("session-1")

    registry.restore_session("session-1")

    assert [s["session_id"] for s in registry.list_sessions()] == ["session-1"]
    assert registry.list_trashed_sessions() == []


def test_restore_session_keeps_archived_flag(registry):
    registry.record_start("session-1", "Title")
    registry.set_archived("session-1", True)
    registry.soft_delete_session("session-1")

    registry.restore_session("session-1")

    active_incl_archived = registry.list_sessions(include_archived=True)
    assert len(active_incl_archived) == 1
    assert active_incl_archived[0]["archived"] == 1
    # still hidden from the default (non-archived) view
    assert registry.list_sessions() == []


def test_list_expired_trashed_sessions_filters_by_retention_window(registry):
    from datetime import datetime, timedelta, timezone

    registry.record_start("old-session", "Deleted 20 days ago")
    registry.record_start("recent-session", "Deleted 2 days ago")

    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=20)).isoformat()
    recent_ts = (now - timedelta(days=2)).isoformat()

    registry.soft_delete_session("old-session")
    registry.soft_delete_session("recent-session")
    # Backdate deleted_at directly — soft_delete_session always stamps "now".
    registry._conn.execute("UPDATE sessions SET deleted_at = ? WHERE session_id = ?", (old_ts, "old-session"))
    registry._conn.execute("UPDATE sessions SET deleted_at = ? WHERE session_id = ?", (recent_ts, "recent-session"))
    registry._conn.commit()

    expired = registry.list_expired_trashed_sessions(retention_days=14)
    assert [s["session_id"] for s in expired] == ["old-session"]


def test_ensure_table_migrates_sessions_table_missing_archived_column(tmp_path):
    # Simulates a database created before the archived column existed.
    conn = sqlite3.connect(str(tmp_path / "legacy-sessions.db"))
    conn.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            turn_count INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at, turn_count) VALUES (?, ?, ?, ?, 1)",
        ("session-1", "Pre-migration session", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    registry = SessionRegistry(conn, backend="sqlite")

    sessions = registry.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["archived"] == 0
    assert sessions[0]["deleted_at"] is None

    # the migrated deleted_at column is also usable, not just present
    registry.soft_delete_session("session-1")
    assert registry.list_trashed_sessions()[0]["session_id"] == "session-1"


def test_derive_session_title_keeps_short_query_verbatim():
    assert derive_session_title("Write about tea") == "Write about tea"


def test_derive_session_title_truncates_long_query_at_word_boundary():
    query = "this is a very long user query that should definitely get truncated at some point because it exceeds sixty characters"
    title = derive_session_title(query, max_len=60)
    assert len(title) <= 61  # 60 + ellipsis
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")  # truncated cleanly at a word boundary
