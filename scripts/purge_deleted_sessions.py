"""Permanently removes sessions that have sat in Trash (soft-deleted via the
sidebar's delete button) past the retention window — both the sessions
registry row and the checkpointer's own conversation data. Sessions still
within the window, or archived/active sessions, are untouched.

Retention is SESSION_RETENTION_DAYS (config/*.yaml session_store.retention_days),
default 14 days.

Run manually: python scripts/purge_deleted_sessions.py
Schedule periodically (daily is plenty, given a multi-day retention window) via
cron, a scheduled task, or a Docker/Kubernetes CronJob, e.g.:
    0 3 * * * cd /path/to/repo && /path/to/venv/bin/python scripts/purge_deleted_sessions.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402
from src.workflow.state_management import get_registry_connection, open_checkpointer  # noqa: E402


async def _purge(settings) -> int:
    registry = get_registry_connection(settings)
    expired = registry.list_expired_trashed_sessions(settings.session_retention_days)
    if not expired:
        return 0

    async with open_checkpointer(settings) as saver:
        for session in expired:
            await saver.adelete_thread(session["session_id"])
            registry.hard_delete_session(session["session_id"])
            print(f"purged {session['session_id']} ({session['title']!r}, deleted_at={session['deleted_at']})")
    return len(expired)


def main() -> None:
    settings = load_settings()
    if settings.session_store_backend not in ("sqlite", "postgres"):
        print(f"session_store_backend is {settings.session_store_backend!r} — nothing to purge.")
        return

    count = asyncio.run(_purge(settings))
    print(f"purged {count} session(s) past the {settings.session_retention_days}-day retention window.")


if __name__ == "__main__":
    main()
