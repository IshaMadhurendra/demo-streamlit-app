"""Snowflake access layer."""

from db.schema_loader import get_schema_context
from db.snowflake_client import SnowflakeClient, get_client

__all__ = ["SnowflakeClient", "get_client", "get_schema_context"]
