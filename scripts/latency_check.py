"""Walk a set of representative queries through the graph and report
per-query latency plus p50 / p95 / max. Run with real creds loaded."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from agent.graph import get_graph  # noqa: E402

# (label, query, prior_turns) — prior_turns is a list of (role, text) pairs
QUERIES = [
    ("US population (simple agg)",
     "What is the total population of the United States?", []),
    ("Highest median income (multi-state rank)",
     "Which state has the highest median household income?", []),
    ("CA college % (percentage)",
     "What percentage of people in California have a college degree?", []),
    ("TX college % (follow-up)",
     "What about Texas?", [
         ("user", "What percentage of people in California have a college degree?"),
         ("assistant", "In California, 34.7% of people have a college degree."),
     ]),
    ("Guardrail block (weather)",
     "What is the weather in New York?", []),
    ("Guardrail block (poem)",
     "Write me a poem", []),
    ("NEED_CLARIFICATION (no geography)",
     "What's the population?", []),
    ("CANNOT_ANSWER with alt (2005)",
     "Show me the population of California in 2005.", []),
    ("Substitution disclosure (avg → median)",
     "What's the average household income in California?", []),
    ("Territory (PR population)",
     "What is the total population of Puerto Rico?", []),
]


def _build_state(q: str, prior: list[tuple[str, str]]):
    msgs = []
    for role, text in prior:
        msgs.append(HumanMessage(content=text) if role == "user"
                    else AIMessage(content=text))
    msgs.append(HumanMessage(content=q))
    return {"messages": msgs, "user_query": q}


def main() -> None:
    g = get_graph()
    # Warm caches (schema, connection pool) so the first query doesn't
    # skew the p50.
    g.invoke({"messages": [HumanMessage(content="hi")], "user_query": "hi"})

    timings: list[float] = []
    print(f"{'label':<45} {'secs':>6}  path")
    print("-" * 85)
    for label, q, prior in QUERIES:
        t0 = time.time()
        r = g.invoke(_build_state(q, prior))
        dt = time.time() - t0
        timings.append(dt)
        path = (
            "block" if r.get("guardrail_status") == "block"
            else "need_clarify" if (r.get("generated_sql") or "").strip().upper().startswith("NEED_CLARIFICATION")
            else "cannot_answer" if (r.get("generated_sql") or "").strip().upper().startswith("CANNOT_ANSWER")
            else "executed"
        )
        print(f"{label:<45} {dt:>6.2f}  {path}")

    print("-" * 85)
    print(f"p50   {statistics.median(timings):.2f}s")
    p95 = sorted(timings)[int(round(0.95 * (len(timings) - 1)))]
    print(f"p95   {p95:.2f}s")
    print(f"max   {max(timings):.2f}s")
    print(f"mean  {statistics.mean(timings):.2f}s")
    over_60 = [t for t in timings if t > 60]
    print(f"over-60s  {len(over_60)} / {len(timings)}")


if __name__ == "__main__":
    main()
