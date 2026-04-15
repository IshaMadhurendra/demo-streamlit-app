"""Agent state definition for the Census Q&A LangGraph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state that flows through every node of the graph.

    `total=False` so nodes can return partial updates without having to
    re-populate every key on every step.
    """

    messages: Annotated[list, add_messages]
    user_query: str
    schema_context: str
    generated_sql: str
    query_results: list[dict[str, Any]]
    final_response: str
    error: str | None
    guardrail_status: str
    guardrail_reason: str | None
    sql_safe: bool
