"""Service settings from the environment and the repository's .env file (KEY=VALUE lines; the environment wins).

The database is MySQL: MYSQL_HOST, MYSQL_PORT (3306), MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE. DATABASE_URL, any
SQLAlchemy URL, overrides them (the tests use sqlite://). See .env.example.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
MYSQL_KEYS = ("MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE")


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE pairs of a .env file; blank lines, # comments and an `export ` prefix are skipped, and one pair of
    surrounding quotes is taken off a value."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def setting(key: str, default: str | None = None) -> str | None:
    """The environment's value, else the .env file's, else default; an empty value (e.g. no password) counts."""
    value = os.environ.get(key)
    if value is None:
        value = read_env_file().get(key)
    return default if value is None else value


def database_url() -> str:
    """The SQLAlchemy URL of the service database; a clear error names what is missing."""
    url = setting("DATABASE_URL")
    if url:
        return url
    # the password may be empty; the others may not
    missing = [key for key in MYSQL_KEYS if setting(key) is None or (key != "MYSQL_PASSWORD" and not setting(key))]
    if missing:
        raise RuntimeError(f"the database is not configured: set {', '.join(missing)} in {ENV_FILE} "
                           "(see .env.example) or DATABASE_URL")
    user, password = quote_plus(setting("MYSQL_USER")), quote_plus(setting("MYSQL_PASSWORD"))
    return (f"mysql+pymysql://{user}:{password}@{setting('MYSQL_HOST')}:{setting('MYSQL_PORT', '3306')}/"
            f"{setting('MYSQL_DATABASE')}?charset=utf8mb4")
