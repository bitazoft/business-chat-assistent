#!/usr/bin/env python3
"""
Apply the SQL migrations in database/migrations/.

Exists because `psql` is often not on PATH on a Windows dev machine, and because
002 uses CREATE INDEX CONCURRENTLY, which Postgres refuses inside a transaction
block. Sending a whole file through a driver as one string counts as one - so
this splits each file into statements and runs them individually in autocommit,
which is what `psql -f` does.

Usage:
    python scripts/apply_migrations.py                 # apply all, in order
    python scripts/apply_migrations.py --dry-run       # just list what would run
    python scripts/apply_migrations.py 003 004         # only these
    python scripts/apply_migrations.py --database-url postgresql://...

Every migration is written to be safe to re-run.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "database" / "migrations"

# Migrations that change existing constraints rather than only adding things.
# Not applied by a bare run - the operator has to name them.
OPT_IN = {"005_chat_logs_allow_unregistered.sql"}


def split_statements(sql: str) -> List[str]:
    """Split a SQL script into individual statements.

    Splitting on ';' alone breaks dollar-quoted bodies (`DO $$ ... END $$;` and
    `CREATE FUNCTION ... $$ ... $$;`), which contain semicolons of their own, so
    this tracks dollar-quote tags, string literals and comments.
    """
    statements: List[str] = []
    current: List[str] = []
    i = 0
    length = len(sql)
    dollar_tag = None  # e.g. "$$" or "$body$" while inside a dollar-quoted block

    while i < length:
        char = sql[i]

        if dollar_tag:
            if sql.startswith(dollar_tag, i):
                current.append(dollar_tag)
                i += len(dollar_tag)
                dollar_tag = None
                continue
            current.append(char)
            i += 1
            continue

        # Line comment
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            end = length if end == -1 else end
            current.append(sql[i:end])
            i = end
            continue

        # Block comment
        if sql.startswith("/*", i):
            end = sql.find("*/", i)
            end = length if end == -1 else end + 2
            current.append(sql[i:end])
            i = end
            continue

        # Single-quoted string ('' is an escaped quote)
        if char == "'":
            current.append(char)
            i += 1
            while i < length:
                current.append(sql[i])
                if sql[i] == "'":
                    if i + 1 < length and sql[i + 1] == "'":
                        current.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue

        # Start of a dollar-quoted block
        if char == "$":
            match = re.match(r"\$[A-Za-z_0-9]*\$", sql[i:])
            if match:
                dollar_tag = match.group(0)
                current.append(dollar_tag)
                i += len(dollar_tag)
                continue

        if char == ";":
            statements.append("".join(current).strip())
            current = []
            i += 1
            continue

        current.append(char)
        i += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)

    # Drop anything that is only comments or whitespace.
    return [
        s
        for s in statements
        if s and any(
            line.strip() and not line.strip().startswith("--")
            for line in s.splitlines()
        )
    ]


def summarise(statement: str, width: int = 88) -> str:
    """A one-line label for a statement, for progress output."""
    body = "\n".join(
        line for line in statement.splitlines() if not line.strip().startswith("--")
    )
    body = " ".join(body.split())
    return body[:width] + ("..." if len(body) > width else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply chatbot database migrations")
    parser.add_argument("only", nargs="*", help="Apply only migrations whose filename contains one of these (e.g. 003)")
    parser.add_argument("--database-url", default=None, help="Defaults to DATABASE_URL from the environment / .env")
    parser.add_argument("--dry-run", action="store_true", help="List statements without running them")
    args = parser.parse_args()

    if args.database_url:
        database_url = args.database_url
    else:
        from config.settings import settings

        database_url = settings.database_url

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if args.only:
        files = [f for f in files if any(token in f.name for token in args.only)]
    else:
        # 005 relaxes an existing constraint, so it is opt-in: it only runs when
        # named explicitly. Read the comments at the top of that file first.
        skipped = [f.name for f in files if f.name in OPT_IN]
        files = [f for f in files if f.name not in OPT_IN]
        for name in skipped:
            print(f"Skipping optional migration {name} (run it by name to apply)")
        if skipped:
            print()

    if not files:
        print(f"No migrations found in {MIGRATIONS_DIR}")
        return 1

    safe_url = re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", database_url)
    print(f"Database: {safe_url}")
    print(f"Migrations: {MIGRATIONS_DIR}\n")

    if args.dry_run:
        for path in files:
            statements = split_statements(path.read_text(encoding="utf-8"))
            print(f"{path.name} ({len(statements)} statements)")
            for statement in statements:
                print(f"    {summarise(statement)}")
            print()
        return 0

    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    try:
        conn = psycopg2.connect(database_url, connect_timeout=10)
    except Exception as e:
        print(f"Could not connect: {e}")
        return 1

    # Autocommit, one statement per execute: required for CONCURRENTLY, and it
    # means a failure part-way leaves earlier statements applied rather than
    # rolling back work that was fine.
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    total_ok = total_failed = 0

    for path in files:
        statements = split_statements(path.read_text(encoding="utf-8"))
        print(f"=== {path.name} ({len(statements)} statements) ===")
        for statement in statements:
            cursor = conn.cursor()
            try:
                cursor.execute(statement)
                for notice in conn.notices[-1:]:
                    print(f"    note: {notice.strip()}")
                conn.notices.clear()
                print(f"  ok   {summarise(statement)}")
                total_ok += 1
            except Exception as e:
                message = str(e).strip().splitlines()[0]
                print(f"  FAIL {summarise(statement)}")
                print(f"       {message}")
                total_failed += 1
            finally:
                cursor.close()
        print()

    conn.close()

    print(f"{total_ok} statement(s) applied, {total_failed} failed")
    if total_failed:
        print(
            "\nA failure on an index or extension is usually a permissions issue "
            "(pg_trgm needs a superuser) and is not fatal - the rest still applied."
        )
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
