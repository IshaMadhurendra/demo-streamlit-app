"""Thin wrapper around snowflake-connector-python.

Goals:
- One connection per process (lazy, reused across queries).
- Simple `query()` returning list[dict] so the rest of the agent never
  touches cursors.
- A 45-second statement timeout enforced server-side.
- Errors surfaced as `SnowflakeError` so callers can render a friendly
  message without leaking driver internals.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

QUERY_TIMEOUT_SECONDS = 45


class SnowflakeError(RuntimeError):
    """Raised when a Snowflake operation fails. The message is safe to log
    but should NOT be shown raw to end users."""


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str

    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        try:
            return cls(
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
                database=os.environ["SNOWFLAKE_DATABASE"],
                schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
            )
        except KeyError as e:
            raise SnowflakeError(
                f"Missing required Snowflake env var: {e.args[0]}"
            ) from e


class SnowflakeClient:
    """Lazy, thread-safe Snowflake client."""

    def __init__(self, config: SnowflakeConfig) -> None:
        self.config = config
        self._conn: Any = None
        self._lock = threading.Lock()

    def _connect(self) -> Any:
        # Imported lazily so unit tests can run without the driver installed.
        import snowflake.connector

        try:
            conn = snowflake.connector.connect(
                account=self.config.account,
                user=self.config.user,
                password=self.config.password,
                warehouse=self.config.warehouse,
                database=self.config.database,
                schema=self.config.schema,
                client_session_keep_alive=True,
                login_timeout=20,
                network_timeout=QUERY_TIMEOUT_SECONDS,
            )
        except Exception as e:
            raise SnowflakeError(f"Failed to connect to Snowflake: {e}") from e

        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {QUERY_TIMEOUT_SECONDS}"
                )
        except Exception:
            # Non-fatal: the timeout is also passed at query time below.
            logger.warning("Could not set session statement timeout", exc_info=True)

        return conn

    def _get_conn(self) -> Any:
        with self._lock:
            if self._conn is None:
                self._conn = self._connect()
            return self._conn

    def query(
        self, sql: str, params: tuple | None = None, max_rows: int = 1000
    ) -> list[dict[str, Any]]:
        """Execute a SELECT and return rows as a list of dicts.

        `max_rows` is an extra client-side cap so a runaway query can't blow
        up the UI; the LLM is also instructed to LIMIT.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    params,
                    timeout=QUERY_TIMEOUT_SECONDS,
                )
                cols = [c[0] for c in cur.description] if cur.description else []
                rows = cur.fetchmany(max_rows)
        except Exception as e:
            raise SnowflakeError(f"Query failed: {e}") from e

        return [dict(zip(cols, row)) for row in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    logger.debug("Error closing Snowflake connection", exc_info=True)
                self._conn = None


_client_singleton: SnowflakeClient | None = None
_singleton_lock = threading.Lock()


def get_client() -> SnowflakeClient:
    """Process-wide singleton client. Safe to call from any thread."""
    global _client_singleton
    with _singleton_lock:
        if _client_singleton is None:
            _client_singleton = SnowflakeClient(SnowflakeConfig.from_env())
        return _client_singleton


def test_connection() -> bool:
    """Return True if we can open a session and run SELECT 1."""
    try:
        rows = get_client().query("SELECT 1 AS OK")
        return bool(rows) and rows[0].get("OK") == 1
    except Exception:
        logger.exception("Snowflake connection test failed")
        return False


def execute_query(
    sql: str, params: tuple | None = None, max_rows: int = 1000
) -> list[dict[str, Any]]:
    """Module-level convenience wrapper around get_client().query()."""
    return get_client().query(sql, params=params, max_rows=max_rows)


def reset_client_for_tests() -> None:
    """Test hook — drop the singleton so a fresh mock can be installed."""
    global _client_singleton
    with _singleton_lock:
        if _client_singleton is not None:
            try:
                _client_singleton.close()
            except Exception:
                pass
        _client_singleton = None
