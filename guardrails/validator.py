"""Rule-based SQL safety checks.

We deliberately do NOT rely on the LLM for this. The whole point of a
guardrail is that it cannot be talked out of doing its job by a clever
prompt-injection in either the user query or the data, so the validator
is plain Python.

The check is intentionally simple and conservative — we'd rather block a
slightly-weird SELECT than ever let a write statement reach Snowflake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Statements we never want to see, even nested in a CTE or string literal.
# These are matched as whole words so we don't false-positive on column
# names like `created_at`.
_BLOCKED_KEYWORDS = (
    "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL", "MERGE", "COPY", "USE",
    "PUT", "GET", "REMOVE", "UNDROP", "REPLACE",
)
_BLOCKED_RE = re.compile(
    r"\b(" + "|".join(_BLOCKED_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)

_LINE_COMMENT_RE = re.compile(r"--")
_BLOCK_COMMENT_RE = re.compile(r"/\*")


@dataclass(frozen=True)
class ValidationResult:
    is_safe: bool
    reason: str | None = None

    def __iter__(self):
        # Allow tuple-unpacking: `is_safe, reason = validate_sql_safety(...)`
        yield self.is_safe
        yield self.reason


def _strip_string_literals(sql: str) -> str:
    """Remove single- and double-quoted string contents so keywords inside
    a literal (e.g. WHERE state ILIKE '%delete me%') don't trigger the
    blocklist. Identifiers in double quotes are also stripped — that's
    fine, our blocklist contains no legitimate identifier we'd lose."""
    out = re.sub(r"'(?:[^']|'')*'", "''", sql)
    out = re.sub(r'"(?:[^"]|"")*"', '""', out)
    return out


def validate_sql_safety(sql: str) -> ValidationResult:
    """Return ValidationResult(is_safe, reason).

    `reason` is a short string suitable for logging; it should NOT be shown
    raw to end users.
    """
    if sql is None or not isinstance(sql, str):
        return ValidationResult(False, "SQL is missing or not a string")

    stripped = sql.strip()
    if not stripped:
        return ValidationResult(False, "SQL is empty")

    # Trim a single trailing semicolon for the leading-keyword check.
    head = stripped.rstrip(";").lstrip()

    # Multiple statements (statement chaining)
    if stripped.rstrip(";").count(";") > 0:
        return ValidationResult(
            False, "Multiple statements detected (statement chaining)"
        )

    # Must START with SELECT (or WITH ... SELECT for a CTE, but only if the
    # final body is a SELECT — we approximate by allowing WITH).
    first_word = head.split(None, 1)[0].upper() if head else ""
    if first_word not in ("SELECT", "WITH"):
        return ValidationResult(
            False, f"Only SELECT statements are allowed (saw {first_word!r})"
        )

    # Comment injection — block both line and block comments.
    if _LINE_COMMENT_RE.search(stripped):
        return ValidationResult(False, "SQL line comments (--) are not allowed")
    if _BLOCK_COMMENT_RE.search(stripped):
        return ValidationResult(False, "SQL block comments (/* */) are not allowed")

    # Blocked keywords — check after stripping string literals so we don't
    # false-positive on user content.
    sanitized = _strip_string_literals(stripped)
    m = _BLOCKED_RE.search(sanitized)
    if m:
        return ValidationResult(
            False, f"Disallowed keyword in SQL: {m.group(1).upper()}"
        )

    return ValidationResult(True, None)
