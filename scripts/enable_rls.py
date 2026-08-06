"""One-time fix for Supabase's "RLS Disabled in Public" / "Sensitive Columns
Exposed" advisor warnings on this app's tables (checkpoints, checkpoint_blobs,
checkpoint_writes, checkpoint_migrations, sessions).

These tables are only ever read/written through this app's own direct
psycopg connection (using the Supabase "postgres" role, which has BYPASSRLS),
never through PostgREST/the Supabase client API. Enabling RLS with no
policies blocks the anon/authenticated PostgREST paths Supabase exposes by
default for every public-schema table, while leaving this app's direct
connection unaffected.

Run once per database: python scripts/enable_rls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402

TABLES = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
    "sessions",
)


def main() -> None:
    settings = load_settings()
    if settings.session_store_backend != "postgres":
        print(f"session_store_backend is {settings.session_store_backend!r}, not 'postgres' — nothing to do.")
        return

    with psycopg.connect(settings.session_store_url, autocommit=True) as conn:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = ANY(%s)",
                (list(TABLES),),
            )
        }
        for table in TABLES:
            if table not in existing:
                print(f"skip {table}: table does not exist yet")
                continue
            conn.execute(f'ALTER TABLE public."{table}" ENABLE ROW LEVEL SECURITY')
            print(f"enabled RLS on public.{table}")


if __name__ == "__main__":
    main()
