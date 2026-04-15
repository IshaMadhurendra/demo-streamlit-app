"""LangGraph node implementations.

Each node takes the AgentState and returns a dict of state updates. Nodes
never raise — any failure is captured in `state["error"]` and routed to a
friendly response by the graph.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import (
    CANNOT_ANSWER_TEMPLATE,
    CONNECTION_ERROR_MESSAGE,
    GENERIC_ERROR_MESSAGE,
    GUARDRAIL_SYSTEM,
    NEED_CLARIFICATION_TEMPLATE,
    OFF_TOPIC_TEMPLATE,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)
from agent.state import AgentState
from db.schema_loader import fips_table_name, get_schema_context
from db.snowflake_client import SnowflakeError, get_client
from guardrails.validator import validate_sql_safety

logger = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-20250514"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


# --------------------------------------------------------------------------- #
# LLM factories — wrapped so tests can monkeypatch them.
# --------------------------------------------------------------------------- #


def _sonnet(temperature: float = 0.0, max_tokens: int = 1024) -> ChatAnthropic:
    return ChatAnthropic(
        model=SONNET_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=45,
    )


def _haiku(temperature: float = 0.0, max_tokens: int = 256) -> ChatAnthropic:
    return ChatAnthropic(
        model=HAIKU_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=15,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _history_text(state: AgentState, max_turns: int = 6) -> str:
    """Render the recent conversation as a short transcript for the LLM."""
    msgs = state.get("messages") or []
    # Drop the very last message — that's the active user query, fed in
    # separately so the LLM doesn't see it twice.
    prior = msgs[:-1] if msgs else []
    prior = prior[-max_turns:]
    if not prior:
        return "(no prior turns)"
    out = []
    for m in prior:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        out.append(f"{role}: {content}")
    return "\n".join(out)


def _strip_sql_fences(text: str) -> str:
    """Models occasionally return ```sql ... ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _format_results_preview(rows: list[dict[str, Any]], max_rows: int = 25) -> str:
    if not rows:
        return "(no rows returned)"
    head = rows[:max_rows]
    try:
        return json.dumps(head, default=str, indent=2)
    except Exception:
        return str(head)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def guardrail_node(state: AgentState) -> dict[str, Any]:
    """Classify the user query as on-topic / off-topic using Haiku."""
    user_query = state.get("user_query", "")
    history = _history_text(state)

    try:
        llm = _haiku()
        resp = llm.invoke([
            SystemMessage(content=GUARDRAIL_SYSTEM),
            HumanMessage(
                content=f"Conversation so far:\n{history}\n\n"
                        f"Latest user message: {user_query}"
            ),
        ])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        # Be tolerant of stray markdown.
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        status = parsed.get("status", "pass").lower()
        reason = parsed.get("reason")
    except Exception:
        logger.warning("Guardrail classification failed — defaulting to pass",
                       exc_info=True)
        status = "pass"
        reason = None

    if status not in ("pass", "block"):
        status = "pass"

    return {"guardrail_status": status, "guardrail_reason": reason}


def schema_context_node(state: AgentState) -> dict[str, Any]:
    """Inject (cached) schema context."""
    try:
        ctx = get_schema_context()
    except Exception:
        logger.exception("Failed to load schema context")
        return {
            "error": "schema_unavailable",
            "schema_context": "",
        }
    return {"schema_context": ctx}


def sql_generation_node(state: AgentState) -> dict[str, Any]:
    """Ask Sonnet to author one Snowflake SELECT for the user's question."""
    schema_context = state.get("schema_context", "")
    user_query = state.get("user_query", "")
    history = _history_text(state)

    database = os.environ.get(
        "SNOWFLAKE_DATABASE",
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    )
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    fips = fips_table_name()

    system = SQL_GENERATION_SYSTEM.format(
        database=database,
        schema=schema,
        fips_table=fips,
        schema_context=schema_context,
        history=history,
    )
    user = SQL_GENERATION_USER.format(user_query=user_query)

    try:
        llm = _sonnet(max_tokens=1024)
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        logger.exception("SQL generation LLM call failed")
        return {"error": "llm_unavailable", "generated_sql": ""}

    sql = _strip_sql_fences(text)
    return {"generated_sql": sql}


def sql_validation_node(state: AgentState) -> dict[str, Any]:
    sql = state.get("generated_sql", "")

    # The SQL generator may legitimately return a CANNOT_ANSWER,
    # NEED_CLARIFICATION, or SMALL_TALK sentinel — treat that as "not
    # safe to execute, but not a security failure" and let the
    # synthesis node handle the user-facing wording.
    head = sql.strip().upper()
    if (head.startswith("CANNOT_ANSWER")
            or head.startswith("NEED_CLARIFICATION")
            or head.startswith("SMALL_TALK")):
        return {"sql_safe": False}

    result = validate_sql_safety(sql)
    if not result.is_safe:
        logger.warning("Generated SQL failed validation: %s", result.reason)
    return {"sql_safe": result.is_safe, "error": None if result.is_safe else "unsafe_sql"}


def execution_node(state: AgentState) -> dict[str, Any]:
    """Run the validated SQL against Snowflake.

    On an "invalid identifier" error (the LLM hallucinated a column
    name), we attempt ONE self-correcting retry: re-prompt the SQL
    author with the failed SQL and the Snowflake error, ask for a
    corrected version, and re-run it. This trades ~5 s of latency for
    a meaningful lift in success rate on long-tail ACS table codes.
    """
    sql = state.get("generated_sql", "")
    client = get_client()
    try:
        rows = client.query(sql, max_rows=1000)
        return {"query_results": rows}
    except SnowflakeError as e:
        err_msg = str(e)
        low = err_msg.lower()
        is_sql_error = (
            "invalid identifier" in low
            or "sql compilation error" in low
            or "syntax error" in low
            or "does not exist" in low
            or "unexpected " in low
            or "ambiguous column" in low
        )
        if is_sql_error and not state.get("_retry_done"):
            logger.info("Retrying SQL after compilation error: %s", err_msg[:200])
            retry_sql = _retry_sql_with_error(state, sql, err_msg)
            if retry_sql and retry_sql != sql:
                try:
                    rows = client.query(retry_sql, max_rows=1000)
                    return {
                        "query_results": rows,
                        "generated_sql": retry_sql,
                        "_retry_done": True,
                    }
                except SnowflakeError:
                    logger.exception("Retry SQL also failed")
            return {
                "query_results": [],
                "error": "query_error",
                "_retry_done": True,
            }
        logger.exception("Snowflake execution failed")
        return {"query_results": [], "error": "snowflake_error"}
    except Exception:
        logger.exception("Unexpected execution failure")
        return {"query_results": [], "error": "execution_error"}


_TABLE_REF_RE = re.compile(r'"(\d{4}_[A-Z0-9_]+)"')


def _actual_columns_for_tables(failed_sql: str) -> str:
    """Look up real column names for every `"YYYY_..."` table referenced
    in the failed SQL. Returns a short block suitable for the retry prompt."""
    names = sorted(set(_TABLE_REF_RE.findall(failed_sql)))
    if not names:
        return ""
    database = os.environ.get(
        "SNOWFLAKE_DATABASE",
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    )
    try:
        client = get_client()
    except Exception:
        return ""
    blocks: list[str] = []
    for t in names:
        try:
            rows = client.query(
                f"SELECT COLUMN_NAME FROM {database}.INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_NAME = %s ORDER BY COLUMN_NAME",
                (t,),
                max_rows=500,
            )
            cols = [r["COLUMN_NAME"] for r in rows]
            if cols:
                blocks.append(f'"{t}" columns: ' + ", ".join(cols))
        except SnowflakeError:
            continue
    return "\n".join(blocks)


def _retry_sql_with_error(
    state: AgentState, failed_sql: str, err_msg: str
) -> str:
    """Ask Sonnet to fix the SQL given the error Snowflake returned."""
    schema_context = state.get("schema_context", "")
    user_query = state.get("user_query", "")
    history = _history_text(state)
    database = os.environ.get(
        "SNOWFLAKE_DATABASE",
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    )
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    from db.schema_loader import fips_table_name
    fips = fips_table_name()

    system = SQL_GENERATION_SYSTEM.format(
        database=database, schema=schema, fips_table=fips,
        schema_context=schema_context, history=history,
    )
    actual_cols = _actual_columns_for_tables(failed_sql)
    correction_user = (
        f"Your previous SQL failed:\n\n{failed_sql}\n\n"
        f"Snowflake error: {err_msg}\n\n"
        f"ACTUAL columns in the referenced tables (use ONLY these):\n"
        f"{actual_cols or '(could not retrieve)'}\n\n"
        f"Output ONLY the corrected SQL (no prose, no markdown fences) "
        f"for the original question: {user_query}\n"
        f"If no available columns can answer the question, output exactly: "
        f"CANNOT_ANSWER: <one-sentence reason>"
    )
    try:
        llm = _sonnet(max_tokens=1024)
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=correction_user),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _extract_sql(text)
    except Exception:
        logger.exception("Retry LLM call failed")
        return ""


def _extract_sql(text: str) -> str:
    """More robust SQL extraction than _strip_sql_fences: if the model
    emitted preamble prose, slice from the first SELECT / WITH / CANNOT_
    token forward."""
    text = _strip_sql_fences(text)
    m = re.search(
        r"(?is)\b(WITH\b|SELECT\b|CANNOT_ANSWER|NEED_CLARIFICATION)",
        text,
    )
    if m:
        return text[m.start():].strip()
    return text


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """Turn results (or an error/empty signal) into a user-facing message."""
    user_query = state.get("user_query", "")
    sql = state.get("generated_sql", "")
    err = state.get("error")
    results = state.get("query_results") or []
    guardrail = state.get("guardrail_status", "pass")
    guardrail_reason = state.get("guardrail_reason") or ""

    # 1. Off-topic — friendly redirect, no LLM call needed.
    if guardrail == "block":
        msg = OFF_TOPIC_TEMPLATE.format(reason=guardrail_reason).strip()
        return _final(msg)

    # 2. The SQL author signalled it's a conversational message, no
    # query needed.
    if sql.strip().upper().startswith("SMALL_TALK"):
        reply = sql.split(":", 1)[1].strip() if ":" in sql else (
            "Happy to help — let me know if there's anything else you'd "
            "like to look up."
        )
        return _final(reply)

    # 3. The SQL author signalled it needs clarification.
    if sql.strip().upper().startswith("NEED_CLARIFICATION"):
        question = sql.split(":", 1)[1].strip() if ":" in sql else (
            "Could you give me a bit more detail about what you're asking?"
        )
        return _final(NEED_CLARIFICATION_TEMPLATE.format(question=question))

    # 4. The SQL author signalled it can't answer.
    if sql.strip().upper().startswith("CANNOT_ANSWER"):
        reason = sql.split(":", 1)[1].strip() if ":" in sql else ""
        return _final(CANNOT_ANSWER_TEMPLATE.format(reason=reason))

    # 3. Hard errors.
    if err == "schema_unavailable" or err == "snowflake_error":
        return _final(CONNECTION_ERROR_MESSAGE)
    if err == "column_unavailable":
        return _final(
            "I don't have the specific columns needed to answer that "
            "question in this dataset. Would you like to try a related "
            "question — for example about population, income, education, "
            "or housing for a specific state or county?"
        )
    if err == "query_error":
        return _final(
            "I had trouble putting together a working query for that one. "
            "Could you try rephrasing, or ask it a slightly different "
            "way? Sometimes breaking the question into two smaller ones "
            "helps."
        )
    if err == "unsafe_sql":
        return _final(GENERIC_ERROR_MESSAGE)
    if err == "llm_unavailable" or err == "execution_error":
        return _final(GENERIC_ERROR_MESSAGE)

    # 4. Happy path — let Sonnet ground a natural-language answer in the
    # rows. The prompt is explicit: don't hallucinate, and handle the
    # empty-results case clearly.
    preview = _format_results_preview(results)
    try:
        llm = _sonnet(max_tokens=800, temperature=0.2)
        resp = llm.invoke([
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=SYNTHESIS_USER.format(
                user_query=user_query,
                n_rows=len(results),
                results_preview=preview,
                history=_history_text(state),
            )),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        logger.exception("Synthesis LLM call failed")
        return _final(GENERIC_ERROR_MESSAGE)

    return _final(text.strip())


def _final(text: str) -> dict[str, Any]:
    return {
        "final_response": text,
        "messages": [AIMessage(content=text)],
    }


# --------------------------------------------------------------------------- #
# Routing helpers
# --------------------------------------------------------------------------- #


def route_after_guardrail(state: AgentState) -> str:
    return "synthesize" if state.get("guardrail_status") == "block" else "load_schema"


def route_after_validation(state: AgentState) -> str:
    sql = state.get("generated_sql", "")
    head = sql.strip().upper()
    if (head.startswith("CANNOT_ANSWER")
            or head.startswith("NEED_CLARIFICATION")
            or head.startswith("SMALL_TALK")):
        return "synthesize"
    if not state.get("sql_safe"):
        return "synthesize"
    return "execute"


def route_after_schema(state: AgentState) -> str:
    if state.get("error") == "schema_unavailable":
        return "synthesize"
    return "generate_sql"
