"""Tests for the LangGraph node functions.

All LLM and Snowflake calls are mocked — these tests run offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agent import nodes
from agent.state import AgentState


def _state(**kw) -> AgentState:
    base: AgentState = {
        "messages": [HumanMessage(content=kw.get("user_query", ""))],
        "user_query": kw.get("user_query", ""),
        "schema_context": kw.get("schema_context", "schema here"),
        "generated_sql": kw.get("generated_sql", ""),
        "query_results": kw.get("query_results", []),
        "final_response": "",
        "error": kw.get("error"),
        "guardrail_status": kw.get("guardrail_status", "pass"),
        "guardrail_reason": kw.get("guardrail_reason"),
        "sql_safe": kw.get("sql_safe", True),
    }
    return base


def _llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=text)
    return llm


# --------------------------------------------------------------------------- #
# Guardrail
# --------------------------------------------------------------------------- #


class TestGuardrailNode:
    def test_on_topic_passes(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "_haiku",
            lambda *a, **kw: _llm_returning('{"status": "pass", "reason": null}'),
        )
        out = nodes.guardrail_node(_state(user_query="What's the population of Texas?"))
        assert out["guardrail_status"] == "pass"
        assert out["guardrail_reason"] is None

    def test_off_topic_blocks(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "_haiku",
            lambda *a, **kw: _llm_returning(
                '{"status": "block", "reason": "medical advice is out of scope"}'
            ),
        )
        out = nodes.guardrail_node(_state(user_query="What are symptoms of flu?"))
        assert out["guardrail_status"] == "block"
        assert "medical" in out["guardrail_reason"]

    def test_malformed_llm_response_defaults_to_pass(self, monkeypatch):
        # If the classifier returns junk, we default to pass rather than
        # black-holing legitimate queries.
        monkeypatch.setattr(
            nodes, "_haiku",
            lambda *a, **kw: _llm_returning("not valid json at all"),
        )
        out = nodes.guardrail_node(_state(user_query="anything"))
        assert out["guardrail_status"] == "pass"

    def test_llm_exception_defaults_to_pass(self, monkeypatch):
        broken = MagicMock()
        broken.invoke.side_effect = RuntimeError("api down")
        monkeypatch.setattr(nodes, "_haiku", lambda *a, **kw: broken)
        out = nodes.guardrail_node(_state(user_query="hi"))
        assert out["guardrail_status"] == "pass"


# --------------------------------------------------------------------------- #
# SQL generation
# --------------------------------------------------------------------------- #


class TestSqlGenerationNode:
    def test_strips_markdown_fences(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "_sonnet",
            lambda *a, **kw: _llm_returning(
                '```sql\nSELECT * FROM t LIMIT 10\n```'
            ),
        )
        out = nodes.sql_generation_node(_state(user_query="q"))
        assert out["generated_sql"].startswith("SELECT")
        assert "```" not in out["generated_sql"]

    def test_passes_through_cannot_answer(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "_sonnet",
            lambda *a, **kw: _llm_returning("CANNOT_ANSWER: no 2099 data"),
        )
        out = nodes.sql_generation_node(_state(user_query="population in 2099"))
        assert out["generated_sql"].startswith("CANNOT_ANSWER")

    def test_llm_failure_sets_error(self, monkeypatch):
        broken = MagicMock()
        broken.invoke.side_effect = RuntimeError("api down")
        monkeypatch.setattr(nodes, "_sonnet", lambda *a, **kw: broken)
        out = nodes.sql_generation_node(_state(user_query="q"))
        assert out.get("error") == "llm_unavailable"


# --------------------------------------------------------------------------- #
# SQL validation
# --------------------------------------------------------------------------- #


class TestSqlValidationNode:
    def test_safe_select(self):
        out = nodes.sql_validation_node(
            _state(generated_sql='SELECT COUNT(*) FROM db.sc."2019_CBG_B01"')
        )
        assert out["sql_safe"] is True
        assert out.get("error") is None

    def test_unsafe_flagged(self):
        out = nodes.sql_validation_node(_state(generated_sql="DROP TABLE x"))
        assert out["sql_safe"] is False
        assert out["error"] == "unsafe_sql"

    def test_cannot_answer_treated_as_not_safe_but_not_error(self):
        out = nodes.sql_validation_node(
            _state(generated_sql="CANNOT_ANSWER: no such data")
        )
        assert out["sql_safe"] is False
        assert out.get("error") is None


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


class TestExecutionNode:
    def test_success_returns_rows(self, monkeypatch):
        fake = MagicMock()
        fake.query.return_value = [{"X": 1}, {"X": 2}]
        monkeypatch.setattr(nodes, "get_client", lambda: fake)
        out = nodes.execution_node(_state(generated_sql="SELECT 1"))
        assert out["query_results"] == [{"X": 1}, {"X": 2}]
        assert "error" not in out or out["error"] is None

    def test_snowflake_error_handled(self, monkeypatch):
        from db.snowflake_client import SnowflakeError

        fake = MagicMock()
        fake.query.side_effect = SnowflakeError("nope")
        monkeypatch.setattr(nodes, "get_client", lambda: fake)
        out = nodes.execution_node(_state(generated_sql="SELECT 1"))
        assert out["query_results"] == []
        assert out["error"] == "snowflake_error"

    def test_unexpected_error_handled(self, monkeypatch):
        fake = MagicMock()
        fake.query.side_effect = ValueError("weird")
        monkeypatch.setattr(nodes, "get_client", lambda: fake)
        out = nodes.execution_node(_state(generated_sql="SELECT 1"))
        assert out["query_results"] == []
        assert out["error"] == "execution_error"


# --------------------------------------------------------------------------- #
# Synthesis — the "do not hallucinate" contract
# --------------------------------------------------------------------------- #


class TestSynthesisNode:
    def test_off_topic_redirect_without_llm_call(self, monkeypatch):
        called = {"n": 0}

        def _llm(*a, **kw):
            called["n"] += 1
            return _llm_returning("unused")

        monkeypatch.setattr(nodes, "_sonnet", _llm)
        out = nodes.synthesis_node(_state(
            guardrail_status="block",
            guardrail_reason="this is off-topic",
        ))
        assert "Census" in out["final_response"]
        assert called["n"] == 0  # no LLM call on the block path

    def test_cannot_answer_uses_template_without_llm(self, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(nodes, "_sonnet",
                            lambda *a, **kw: (called.__setitem__("n", called["n"]+1)
                                              or _llm_returning("unused")))
        out = nodes.synthesis_node(_state(
            generated_sql="CANNOT_ANSWER: no 2099 data",
        ))
        assert called["n"] == 0
        assert "no 2099 data" in out["final_response"]

    def test_connection_error_path(self, monkeypatch):
        monkeypatch.setattr(nodes, "_sonnet",
                            lambda *a, **kw: _llm_returning("unused"))
        out = nodes.synthesis_node(_state(error="snowflake_error"))
        assert "trouble connecting" in out["final_response"].lower()

    def test_empty_results_does_not_hallucinate(self, monkeypatch):
        """Regression guard: when results are empty, the LLM should be told
        so explicitly and is expected to say no data was found. We verify
        the prompt we feed it contains `(no rows returned)` — that's the
        contract the SYNTHESIS_SYSTEM prompt keys off of."""
        captured: list[list] = []

        class _LLM:
            def invoke(self, messages):
                captured.append(messages)
                return SimpleNamespace(
                    content="I didn't find any data for that."
                )

        monkeypatch.setattr(nodes, "_sonnet", lambda *a, **kw: _LLM())
        out = nodes.synthesis_node(_state(
            user_query="population of Atlantis",
            generated_sql='SELECT 1 FROM t',
            query_results=[],
        ))
        assert "didn't find" in out["final_response"].lower()
        # Verify we actually signaled "no rows" to the LLM.
        human_content = captured[0][1].content
        assert "(no rows returned)" in human_content

    def test_results_are_passed_verbatim_to_llm(self, monkeypatch):
        """Make sure we don't silently drop columns/rows on the way in —
        prevents a whole class of 'LLM hallucinated because it was starved
        of data' bugs."""
        captured: list = []

        class _LLM:
            def invoke(self, messages):
                captured.append(messages[1].content)
                return SimpleNamespace(content="The population is 42.")

        monkeypatch.setattr(nodes, "_sonnet", lambda *a, **kw: _LLM())
        rows = [{"STATE": "California", "POP": 39000000}]
        out = nodes.synthesis_node(_state(
            user_query="pop of california",
            generated_sql="SELECT ...",
            query_results=rows,
        ))
        assert "39000000" in captured[0]
        assert "California" in captured[0]
        assert out["final_response"] == "The population is 42."


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


class TestRouting:
    def test_guardrail_block_routes_to_synthesize(self):
        assert nodes.route_after_guardrail(_state(guardrail_status="block")) == "synthesize"

    def test_guardrail_pass_routes_to_schema(self):
        assert nodes.route_after_guardrail(_state(guardrail_status="pass")) == "load_schema"

    def test_validation_unsafe_routes_to_synthesize(self):
        assert nodes.route_after_validation(_state(sql_safe=False,
                                                    generated_sql="DROP x")) == "synthesize"

    def test_validation_cannot_answer_routes_to_synthesize(self):
        assert nodes.route_after_validation(_state(
            sql_safe=False, generated_sql="CANNOT_ANSWER: nope"
        )) == "synthesize"

    def test_validation_safe_routes_to_execute(self):
        assert nodes.route_after_validation(_state(
            sql_safe=True, generated_sql="SELECT 1"
        )) == "execute"

    def test_schema_error_routes_to_synthesize(self):
        assert nodes.route_after_schema(_state(error="schema_unavailable")) == "synthesize"
