"""LangGraph wiring.

Flow:
    user_input
      -> guardrail
          -> (block) synthesize  (off-topic redirect)
          -> (pass)  load_schema
                        -> (schema unavailable) synthesize
                        -> generate_sql
                            -> validate_sql
                                -> (CANNOT_ANSWER) synthesize
                                -> (unsafe)        synthesize
                                -> (safe)          execute -> synthesize
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    execution_node,
    guardrail_node,
    route_after_guardrail,
    route_after_schema,
    route_after_validation,
    schema_context_node,
    sql_generation_node,
    sql_validation_node,
    synthesis_node,
)
from agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guardrail", guardrail_node)
    g.add_node("load_schema", schema_context_node)
    g.add_node("generate_sql", sql_generation_node)
    g.add_node("validate_sql", sql_validation_node)
    g.add_node("execute", execution_node)
    g.add_node("synthesize", synthesis_node)

    g.add_edge(START, "guardrail")
    g.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"load_schema": "load_schema", "synthesize": "synthesize"},
    )
    g.add_conditional_edges(
        "load_schema",
        route_after_schema,
        {"generate_sql": "generate_sql", "synthesize": "synthesize"},
    )
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {"execute": "execute", "synthesize": "synthesize"},
    )
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Process-wide compiled graph singleton."""
    return build_graph()
