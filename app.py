"""Streamlit chat UI for the Census Q&A agent.

Run locally:
    streamlit run app.py

On Streamlit Community Cloud, set the same env vars in the Secrets
manager (TOML format) — `_load_secrets_into_env` will pull them into
`os.environ` for the rest of the codebase.
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _load_secrets_into_env() -> None:
    """Bridge st.secrets -> os.environ for cloud deploys."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in (
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
        "ANTHROPIC_API_KEY",
    ):
        try:
            if key in secrets and not os.environ.get(key):
                os.environ[key] = str(secrets[key])
        except Exception:
            continue


_load_secrets_into_env()
logging.basicConfig(level=logging.INFO)

from agent.graph import get_graph  # noqa: E402
from agent.prompts import CONNECTION_ERROR_MESSAGE  # noqa: E402


# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Census Insights",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Palette — single accent, Snowflake brand blue (#29B5E8).
         Theme is pinned to light via .streamlit/config.toml so we do
         not need !important hacks to override browser dark mode. */

      :root {
        --accent: #29B5E8;
        --accent-dark: #0C4A6E;
        --accent-soft: #E0F2FE;
        --accent-border: #BAE6FD;
      }

      /* Page background color is set via .streamlit/config.toml
         (backgroundColor = #EEF6FC). We force chat bubbles and the
         query expander to white below so they lift off the tint. */
      .block-container {padding-top: 3.5rem; max-width: 820px;}

      /* Cleaner, analytical title — smaller than st.title, tight
         letter-spacing, no emoji */
      .app-title {font-size: 1.6rem; font-weight: 600;
                  letter-spacing: -0.015em; color: #0F172A;
                  margin: 0 0 0.2rem;}

      /* Chat bubbles — visually distinguish user from assistant.
         User: Snowflake-blue tint, no border.
         Assistant: white card, hairline gray border.
         Both fade in gently when first rendered. */
      @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(6px);}
        to   {opacity: 1; transform: translateY(0);}
      }
      @keyframes pulseDot {
        0%, 100% {opacity: 0.25;}
        50%      {opacity: 1;}
      }
      .stChatMessage {border-radius: 14px;
                      animation: fadeInUp 0.28s ease-out;}
      div[data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="chatAvatarIcon-user"]
      ) {
        background: #E0F2FE !important;
        border: none !important;
      }
      div[data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="chatAvatarIcon-assistant"]
      ) {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
      }

      .subtitle {color: #1E1B4B; font-size: 0.92rem; margin-top: -0.3rem;
                 margin-bottom: 1.8rem; line-height: 1.55;
                 max-width: 62ch;}
      .chip-label {color: #64748B; font-size: 0.78rem; font-weight: 600;
                   letter-spacing: 0.08em; text-transform: uppercase;
                   margin: 2rem 0 0.5rem;}
      .source-note {color: #6b7280; font-size: 0.85rem; margin-top: 0.5rem;}

      /* Thin accent bar under the title */
      .accent-rule {height: 3px; width: 56px;
                    background: var(--accent);
                    border-radius: 3px; margin: 0.2rem 0 1.1rem;}

      /* Sidebar — deep navy to contrast the pale-blue main area,
         Snowflake-corporate vibe. Force light text on every element
         so nothing disappears. */
      [data-testid="stSidebar"] {background: #0F172A !important;
                                 border-right: 1px solid #0F172A;}
      [data-testid="stSidebar"] * {color: #E2E8F0 !important;}
      [data-testid="stSidebar"] a {color: #7DD3FC !important;
                                    text-decoration: underline;}
      [data-testid="stSidebar"] hr {border-color: #1E293B !important;}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
        color: #F1F5F9 !important;
      }
      .sb-brand {color: #29B5E8 !important; font-weight: 600;
                 font-size: 1.05rem; letter-spacing: -0.005em;}
      .sb-tag {color: #94A3B8 !important; font-size: 0.78rem;
               line-height: 1.45; margin: 0.15rem 0 0;}
      .sb-section {color: #94A3B8 !important; font-size: 0.68rem;
                   font-weight: 600; letter-spacing: 0.14em;
                   text-transform: uppercase; margin: 0.2rem 0 0.5rem;}
      .sb-topic {font-size: 0.8rem; padding: 0.15rem 0;
                 line-height: 1.45;
                 color: #E2E8F0 !important;}
      .sb-topic .ico {display: inline-block; width: 1.1rem;
                      font-size: 0.78rem; opacity: 0.85;}
      .sb-topic b {color: #E2E8F0 !important; font-weight: 600;}
      .sb-stage-list {font-size: 0.78rem; line-height: 1.55;
                      color: #CBD5E1 !important;
                      font-family: ui-monospace, SFMono-Regular, monospace;}

      /* Clear-conversation button needs light-on-dark treatment too */
      [data-testid="stSidebar"] .stButton > button {
        background: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1.5px solid #334155 !important;
      }
      [data-testid="stSidebar"] .stButton > button:hover {
        background: #334155 !important;
        border-color: #29B5E8 !important;
      }

      div[data-testid="stExpander"] details summary p {font-size: 0.85rem;}
      div[data-testid="stExpander"] {background: #ffffff !important;
                                      border-radius: 10px;
                                      border: 1px solid #e5e7eb;}

      /* Staged-progress lines */
      .stage-line {color: #6b7280; font-size: 0.9rem;
                   font-family: ui-monospace, SFMono-Regular, monospace;
                   line-height: 1.6;}
      .stage-done {color: #047857;}
      .stage-active {color: var(--accent-dark);}
      .stage-active .pulse {display: inline-block;
                            animation: pulseDot 1.2s ease-in-out infinite;}

      /* Suggestion chip buttons — compact single-line pills */
      .stButton {width: 100%;}
      .stButton > button {width: 100%; border-radius: 999px;
                          padding: 0.25rem 0.75rem; font-size: 0.82rem;
                          border: 1px solid var(--accent-border);
                          background: #ffffff; color: var(--accent-dark);
                          font-weight: 500;
                          white-space: nowrap; line-height: 1.2;
                          min-height: 2rem;
                          overflow: hidden; text-overflow: ellipsis;
                          transition: background 0.15s, border-color 0.15s;}
      .stButton > button:hover {background: var(--accent-soft);
                                border-color: var(--accent);}

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "history" not in st.session_state:
    st.session_state.history = []
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="sb-brand">Census Chat</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-tag">A friendly natural-language guide to US '
        "demographic data — powered by Snowflake and the American "
        "Community Survey.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sb-section">What I can help with</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-topic"><span class="ico">👥</span> <b>Population</b> — by state, county, block group</div>'
        '<div class="sb-topic"><span class="ico">💰</span> <b>Income &amp; poverty</b> — medians, SNAP, poverty rates</div>'
        '<div class="sb-topic"><span class="ico">🏠</span> <b>Housing</b> — values, rents, ownership, vacancy</div>'
        '<div class="sb-topic"><span class="ico">🎓</span> <b>Education</b> — attainment, enrollment, college rates</div>'
        '<div class="sb-topic"><span class="ico">💼</span> <b>Employment</b> — labor force, commuting, occupation</div>'
        '<div class="sb-topic"><span class="ico">🌎</span> <b>Demographics</b> — age, sex, race, language, citizenship</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sb-section">Data source</div>',
                unsafe_allow_html=True)
    st.markdown(
        "[SafeGraph US Open Census Data]"
        "(https://app.snowflake.com/marketplace/listing/GZSUZ7C5UB/safegraph-us-open-census-data-neighborhood-insights-free-dataset) "
        "— via **Snowflake Marketplace**, American Community Survey 2019–2020."
    )
    st.divider()

    if st.button("Clear conversation", use_container_width=True,
                 disabled=not st.session_state.history):
        st.session_state.history = []
        st.session_state.lc_messages = []
        st.session_state.pending_query = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

# On empty state, push header+chips down so they sit roughly centered
# rather than floating at the top with a void above the input bar.
if not st.session_state.history and st.session_state.pending_query is None:
    st.markdown(
        """
        <style>
          [data-testid="stMain"] .block-container {padding-top: 18vh;}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<h2 class="app-title">Chat with the US Census</h2>',
    unsafe_allow_html=True,
)
st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask about US demographic data — population, '
    "income, housing, education, or employment — for any state, county, "
    "or territory. Answered with live numbers from the American Community "
    "Survey, powered by Snowflake.</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Quick-start query chips (only shown on empty chat)
# --------------------------------------------------------------------------- #

SUGGESTIONS = [
    ("California population", "What's the population of California?"),
    ("Median income by state", "Which state has the highest median household income?"),
    ("College rates in Texas", "What percentage of people in Texas have a college degree?"),
    ("Puerto Rico population", "What is the total population of Puerto Rico?"),
]

if not st.session_state.history and st.session_state.pending_query is None:
    st.markdown('<div class="chip-label">Try one of these</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTIONS), gap="small")
    for col, (label, query) in zip(cols, SUGGESTIONS):
        if col.button(label, key=f"chip_{label}", use_container_width=True):
            st.session_state.pending_query = query
            st.rerun()


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #

_STAGE_LABELS = {
    "guardrail": "Checking topic",
    "load_schema": "Loading schema",
    "generate_sql": "Writing query",
    "validate_sql": "Validating query",
    "execute": "Running against Snowflake",
    "synthesize": "Writing answer",
}


def _render_stages(container, stages_done: list[str], active: str | None) -> None:
    """Render the step indicator for the current agent run."""
    lines = []
    for key, label in _STAGE_LABELS.items():
        if key in stages_done:
            lines.append(
                f'<div class="stage-line stage-done">✓ {label}</div>'
            )
        elif key == active:
            lines.append(
                f'<div class="stage-line stage-active">'
                f'<span class="pulse">•</span> {label}…</div>'
            )
    container.markdown("\n".join(lines) or "&nbsp;", unsafe_allow_html=True)


def _results_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame | None:
    if not rows:
        return None
    try:
        df = pd.DataFrame(rows)
        return df
    except Exception:
        return None


def _maybe_render_chart(df: pd.DataFrame | None) -> None:
    """Render a chart inline under the text answer when the shape of
    the result justifies it. Silent otherwise — a confusing chart is
    worse than no chart.

    Shapes we handle:
      1. 2 cols, category+numeric, ≥3 rows
         → top-10 horizontal bar chart.
      2. 2 cols, year+numeric, ≥3 rows → single-line chart over years.
      3. 3 cols (category, year-like, numeric)
         → pivot to wide, line chart per category.
      4. 1 category col + 2+ numeric cols (wide), ≥2 rows
         → grouped bar chart, one series per numeric col. Small-magnitude
           "delta" columns (max < 10% of the largest col) are dropped
           so they don't render as invisible bars alongside big ones.
    """
    if df is None or df.empty or len(df) < 2:
        return

    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in df.columns
                 if not pd.api.types.is_numeric_dtype(df[c])]

    def _is_year_col(col: str) -> bool:
        s = df[col]
        return (pd.api.types.is_integer_dtype(s)
                and s.between(1900, 2100).all())

    # Shape 3: long format (category, year, value) — pivot to wide & line
    if (len(df.columns) == 3 and len(text_cols) == 1
            and len(numeric_cols) == 2):
        year_col = next((c for c in numeric_cols if _is_year_col(c)), None)
        value_col = next((c for c in numeric_cols if c != year_col), None)
        if year_col and value_col:
            cat_col = text_cols[0]
            try:
                wide = df.pivot(
                    index=year_col, columns=cat_col, values=value_col,
                ).sort_index()
                if wide.shape[0] >= 2:
                    st.line_chart(wide, height=280)
                    return
            except Exception:
                pass  # fall through

    # Shape 4: wide multi-series (one category + several numeric cols)
    if (len(text_cols) == 1 and len(numeric_cols) >= 2
            and len(df) >= 2):
        # Drop columns whose max is tiny relative to the largest —
        # typically derived "change" or "growth" columns that would
        # render as invisible bars next to the main series.
        maxes = {c: df[c].abs().max() for c in numeric_cols}
        top_max = max(maxes.values()) or 1
        plot_cols = [c for c, m in maxes.items() if m >= 0.1 * top_max]
        if len(plot_cols) >= 2:
            st.bar_chart(
                df.set_index(text_cols[0])[plot_cols],
                height=300,
            )
            return

    # Shapes 1 + 2: two-column classic
    if len(df.columns) == 2 and len(df) >= 3:
        a, b = df.columns[0], df.columns[1]
        a_num = pd.api.types.is_numeric_dtype(df[a])
        b_num = pd.api.types.is_numeric_dtype(df[b])
        if a_num and b_num:
            if _is_year_col(a) and df[b].nunique() >= 2:
                st.line_chart(df.set_index(a)[b], height=260)
            return
        if a_num == b_num:
            return
        label_col, value_col = (a, b) if b_num else (b, a)
        if df[value_col].nunique() < 2:
            return
        top = (
            df[[label_col, value_col]]
            .dropna()
            .sort_values(value_col, ascending=False)
            .head(10)
        )
        st.bar_chart(
            top.set_index(label_col)[value_col],
            height=max(240, 28 * len(top)),
        )




def _render_retrieval_panel(
    sql: str, rows: list[dict[str, Any]] | None, elapsed: float,
    key: str | None = None,
) -> None:
    df = _results_to_df(rows or [])
    # Stable per-panel key so multiple history entries don't collide
    # on the download_button's auto-generated id.
    panel_key = key or f"panel_{abs(hash(sql)) % (10**9)}"
    with st.expander("View query", expanded=False):
        tabs = st.tabs(["SQL", "Data"]) if df is not None else st.tabs(["SQL"])
        with tabs[0]:
            st.code(sql, language="sql")
        if df is not None:
            with tabs[1]:
                st.dataframe(df, use_container_width=True, hide_index=True)
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button(
                    "Download CSV",
                    buf.getvalue(),
                    file_name="census_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{panel_key}",
                )
        st.markdown(
            f'<div class="source-note">Answered in {elapsed:.1f}s · '
            "Source: SafeGraph US Open Census Data via Snowflake Marketplace "
            "(American Community Survey).</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Render prior history
# --------------------------------------------------------------------------- #

for i, entry in enumerate(st.session_state.history):
    role = entry["role"]
    with st.chat_message(role):
        st.markdown(entry["content"])
        if role == "assistant" and entry.get("sql"):
            _maybe_render_chart(_results_to_df(entry.get("rows") or []))
            _render_retrieval_panel(
                entry["sql"], entry.get("rows"),
                entry.get("elapsed", 0.0),
                key=f"hist_{i}",
            )


# --------------------------------------------------------------------------- #
# Agent invocation with live staging
# --------------------------------------------------------------------------- #


def _run_agent_streaming(user_query: str, stage_placeholder) -> dict[str, Any]:
    """Drive the graph with graph.stream() so we can surface per-node
    progress while it's running."""
    graph = get_graph()
    st.session_state.lc_messages.append(HumanMessage(content=user_query))
    state = {
        "messages": st.session_state.lc_messages,
        "user_query": user_query,
    }

    stages_done: list[str] = []
    current_state: dict[str, Any] = {}
    active = "guardrail"
    _render_stages(stage_placeholder, stages_done, active)

    for event in graph.stream(state, stream_mode="updates"):
        for node_name, node_update in event.items():
            if node_update:
                current_state.update(node_update)
            stages_done.append(node_name)
            active = None
            # Render the just-completed stage plus the next-expected one.
            keys = list(_STAGE_LABELS.keys())
            if node_name in keys:
                idx = keys.index(node_name)
                if idx + 1 < len(keys):
                    active = keys[idx + 1]
            _render_stages(stage_placeholder, stages_done, active)

    _render_stages(stage_placeholder, stages_done, None)
    return current_state


# Resolve user input: either the chat input or a clicked chip
user_query = st.chat_input("Type your question here...")
if st.session_state.pending_query and not user_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        stage_placeholder = st.empty()
        answer_placeholder = st.empty()

        t0 = time.time()
        try:
            result = _run_agent_streaming(user_query, stage_placeholder)
            response_text = (
                result.get("final_response")
                or "I wasn't able to put together a response — could you try rephrasing?"
            )
            sql = result.get("generated_sql") or ""
            rows = result.get("query_results") or []
            guardrail_blocked = result.get("guardrail_status") == "block"
        except Exception:
            logging.exception("Agent invocation crashed")
            response_text = CONNECTION_ERROR_MESSAGE
            sql = ""
            rows = []
            guardrail_blocked = False
        elapsed = time.time() - t0

        # Clear the stage indicator once we have an answer.
        stage_placeholder.empty()

        # Typewriter stream: render the response word-by-word so the
        # answer feels conversational and perceptibly faster even if
        # the total wall-clock time is unchanged.
        def _stream_words(text: str, delay: float = 0.012):
            for i, word in enumerate(text.split(" ")):
                yield (" " if i else "") + word
                time.sleep(delay)

        with answer_placeholder.container():
            st.write_stream(_stream_words(response_text))

        show_sql = bool(sql) and not guardrail_blocked
        if show_sql:
            _maybe_render_chart(_results_to_df(rows))
            _render_retrieval_panel(
                sql, rows, elapsed,
                key=f"live_{len(st.session_state.history)}",
            )

    st.session_state.history.append({
        "role": "assistant",
        "content": response_text,
        "sql": sql if show_sql else None,
        "rows": rows if show_sql else None,
        "elapsed": elapsed,
    })
    st.session_state.lc_messages.append(AIMessage(content=response_text))
