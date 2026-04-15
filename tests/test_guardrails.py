"""Tests for the rule-based SQL safety validator."""

from __future__ import annotations

import pytest

from guardrails.validator import validate_sql_safety


class TestValidSelects:
    @pytest.mark.parametrize("sql", [
        'SELECT B01001e1 FROM db.sc."2019_CBG_B01" LIMIT 10',
        'select COUNT(*) from db.sc."2019_CBG_B01"',
        '  SELECT * FROM db.sc."2019_CBG_B17"  ',
        'WITH x AS (SELECT 1 AS n) SELECT n FROM x',
        "SELECT name FROM t WHERE name ILIKE '%california%' LIMIT 5",
        'SELECT state FROM db.sc."2019_METADATA_CBG_FIPS_CODES";',
    ])
    def test_valid_select_passes(self, sql):
        result = validate_sql_safety(sql)
        assert result.is_safe, f"Expected safe: {sql} / {result.reason}"
        assert result.reason is None


class TestBlockedKeywords:
    @pytest.mark.parametrize("sql,keyword", [
        ("DROP TABLE foo", "DROP"),
        ("DELETE FROM foo WHERE 1=1", "DELETE"),
        ("INSERT INTO foo VALUES (1)", "INSERT"),
        ("UPDATE foo SET x=1", "UPDATE"),
        ("CREATE TABLE foo (x INT)", "CREATE"),
        ("ALTER TABLE foo ADD COLUMN x INT", "ALTER"),
        ("TRUNCATE TABLE foo", "TRUNCATE"),
        ("GRANT SELECT ON foo TO role", "GRANT"),
        ("REVOKE SELECT ON foo FROM role", "REVOKE"),
        ("CALL my_proc()", "CALL"),
        ("EXEC my_proc", "EXEC"),
        ("MERGE INTO foo USING bar ON ...", "MERGE"),
    ])
    def test_blocked_keyword_at_start(self, sql, keyword):
        result = validate_sql_safety(sql)
        assert not result.is_safe
        assert keyword in (result.reason or "").upper()

    def test_blocked_keyword_in_the_middle(self):
        sql = "SELECT * FROM t; DROP TABLE t"
        result = validate_sql_safety(sql)
        assert not result.is_safe


class TestInjection:
    def test_line_comment_blocked(self):
        sql = "SELECT 1 -- ha"
        r = validate_sql_safety(sql)
        assert not r.is_safe
        assert "comment" in (r.reason or "").lower()

    def test_block_comment_blocked(self):
        sql = "SELECT /* sneaky */ 1"
        r = validate_sql_safety(sql)
        assert not r.is_safe

    def test_statement_chaining_blocked(self):
        sql = "SELECT 1; SELECT 2"
        r = validate_sql_safety(sql)
        assert not r.is_safe
        assert "chain" in (r.reason or "").lower() or "multiple" in (r.reason or "").lower()


class TestEdgeCases:
    def test_empty_is_unsafe(self):
        assert not validate_sql_safety("").is_safe
        assert not validate_sql_safety("   ").is_safe

    def test_none_is_unsafe(self):
        assert not validate_sql_safety(None).is_safe  # type: ignore[arg-type]

    def test_blocked_keyword_inside_string_literal_is_allowed(self):
        # 'DROP' appearing inside a WHERE clause string must NOT trip the
        # validator — otherwise legitimate queries about, say, school
        # dropout rates would fail.
        sql = "SELECT name FROM t WHERE reason = 'DROP out of school'"
        r = validate_sql_safety(sql)
        assert r.is_safe, r.reason

    def test_non_select_verb_rejected(self):
        r = validate_sql_safety("SHOW TABLES")
        assert not r.is_safe

    def test_tuple_unpacking(self):
        is_safe, reason = validate_sql_safety("SELECT 1")
        assert is_safe is True
        assert reason is None
