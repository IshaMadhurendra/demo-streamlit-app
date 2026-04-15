# Census Insights — an interactive chat agent over US Census data

An interactive chat agent that answers natural-language questions about US
demographics, housing, income, education, employment, and other indicators
from the American Community Survey (ACS). Built on **Streamlit +
LangGraph + Anthropic Claude + Snowflake**.

> **Live demo:** _paste Streamlit Community Cloud URL here after deploy_
> **Data source:** [SafeGraph US Open Census Data](https://app.snowflake.com/marketplace/listing/GZSUZ7C5UB/safegraph-us-open-census-data-neighborhood-insights-free-dataset), Snowflake Marketplace.

---

## Architecture

```
                    ┌──────────────────┐
  user message ───▶ │ Streamlit chat   │ ◀── st.session_state history
                    │    (app.py)      │
                    └─────────┬────────┘
                              │ invoke()
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                      LangGraph StateGraph                        │
  │                                                                  │
  │   guardrail ──block──▶ synthesize ──▶ END                        │
  │       │pass                                                      │
  │       ▼                                                          │
  │   load_schema ──error──▶ synthesize ──▶ END                      │
  │       │ok                                                        │
  │       ▼                                                          │
  │   generate_sql ──▶ validate_sql ──unsafe/CANNOT──▶ synthesize    │
  │                          │safe                                   │
  │                          ▼                                       │
  │                       execute ──▶ synthesize ──▶ END             │
  └──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
       ┌────────────┐  ┌─────────────┐    ┌──────────────┐
       │  Claude    │  │  Snowflake  │    │ rule-based   │
       │  Sonnet +  │  │   Census    │    │ SQL safety   │
       │  Haiku     │  │   dataset   │    │  validator   │
       └────────────┘  └─────────────┘    └──────────────┘
```

### Node responsibilities

| Node           | Model         | What it does |
|----------------|---------------|--------------|
| `guardrail`    | Claude Haiku  | Fast on-topic / off-topic classifier. Off-topic queries get a polite redirect without a Snowflake round-trip. |
| `load_schema`  | —             | Loads a compact (<3 KB) schema summary from `INFORMATION_SCHEMA`, cached for the life of the process. |
| `generate_sql` | Claude Sonnet | Authors one Snowflake `SELECT` against the schema, using conversation history for follow-ups. Returns `CANNOT_ANSWER: …` when the data can't support the question. |
| `validate_sql` | rule-based    | Statically enforces: `SELECT` only, no DDL/DML keywords, no `--` / `/* */`, no statement chaining. LLM-authored SQL is treated as untrusted. |
| `execute`      | —             | Runs the SQL with a 45-second statement timeout. Driver errors become `state["error"]`, never exceptions. |
| `synthesize`   | Claude Sonnet | Grounds a natural-language answer in the returned rows. Handles off-topic, cannot-answer, connection-error, and empty-result paths without calling the LLM when an unambiguous template suffices. |

### Agent state (`agent/state.py`)

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # full conversation
    user_query: str
    schema_context: str
    generated_sql: str
    query_results: list[dict]
    final_response: str
    error: str | None
    guardrail_status: str                     # "pass" | "block"
    guardrail_reason: str | None
    sql_safe: bool
```

---

## Running locally

1. **Clone & install**
   ```bash
   git clone <this-repo>
   cd census-agent
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure credentials** — copy `.env.example` to `.env` and fill in:
   ```bash
   cp .env.example .env
   # then edit .env with your Snowflake + Anthropic creds
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Run tests** (no network, fully mocked)
   ```bash
   pytest -q
   ```

### Snowflake prerequisites

Your Snowflake role must have `USAGE` on the database
`US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET` and `SELECT` on
every table in the `PUBLIC` schema. The dataset is free on Snowflake
Marketplace.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io)
   pointing at `app.py`.
3. In **Settings → Secrets**, paste TOML-formatted env vars:

   ```toml
   SNOWFLAKE_ACCOUNT = "abc12345.us-east-1"
   SNOWFLAKE_USER = "..."
   SNOWFLAKE_PASSWORD = "..."
   SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
   SNOWFLAKE_DATABASE = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"
   SNOWFLAKE_SCHEMA = "PUBLIC"
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy. First-request cold start is ~5–10 s while the schema is
   introspected; subsequent requests are fast.

---

## Example questions

- "What's the total population of California?"
- "Which five counties in Texas have the highest median household income?"
- "How many people in New York state have a bachelor's degree or higher?"
- "What share of households in Florida receive SNAP benefits?"
- "Compare average commute time between King County, WA and
  Multnomah County, OR."
- _Follow-up:_ "What about for Washington State overall?"

---

## Key design decisions & tradeoffs

**Two-model split (Haiku for guardrails, Sonnet for SQL + synthesis).**
The off-topic classifier runs on every message but needs no reasoning
depth, so Haiku is ~10× cheaper and ~3× faster. The two user-facing LLM
tasks (SQL generation and answer synthesis) do need judgment, so they use
Sonnet. Net effect: off-topic messages cost cents and resolve in ~1 s;
on-topic messages take ~5–15 s end-to-end.

**Rule-based SQL validator, not LLM-based.** A guardrail you can
prompt-inject your way past isn't a guardrail. The validator in
`guardrails/validator.py` is ~60 lines of Python and cannot be talked
out of its job. It runs after SQL generation and before execution, so
even if the generator is compromised (prompt injection, jailbreak, or
just a hallucination) no write statement ever reaches Snowflake.

**Schema summary, not full schema dump.** The Census DB has hundreds of
tables and tens of thousands of columns. We build a ~3 KB summary with
(a) the table legend, (b) the list of available years, (c) the names of
the metadata tables, and (d) small row samples so the LLM sees what
values look like. Cached per-process.

**Let the LLM say "I can't."** The SQL generator is instructed to return
`CANNOT_ANSWER: <reason>` rather than hallucinate table names when the
data truly doesn't support the question. This turned out to be higher
signal than adding after-the-fact checks.

**Friendly, templated error surfaces.** The user never sees Snowflake
errors, SQL, Python tracebacks, or implementation jargon. Connection
errors, empty results, unsafe SQL, and off-topic queries each have a
dedicated path in the `synthesize` node, some of which skip the LLM
entirely for latency.

**Conversation memory via `st.session_state`.** The full message history
rides along on every graph invocation so follow-ups like "what about
California?" resolve naturally against prior turns. No external
thread-store dependency — fits inside the Streamlit session.

### Things I consciously didn't build

- Vector search over the `METADATA_CBG_FIELD_DESCRIPTIONS` table. The
  schema summary is small enough that Sonnet can handle column lookup
  directly, and adding a vector DB is a new external dependency that
  pays for itself only once the schema stops fitting in context.
- A charting / visualization step. The assignment asks for a chat agent,
  and mixing chart rendering into the synthesis path makes the prompt
  contract (text-only, grounded in rows) messier. Streamlit could render
  charts trivially if a product owner asked for it.
- A caching layer on top of Snowflake query results. Census tables are
  static, so this would be a real win for hot queries — but it'd also add
  cache-invalidation surface. Deferred.

---

## Project layout

```
census-agent/
├── app.py                  Streamlit entry point
├── agent/
│   ├── graph.py            StateGraph wiring
│   ├── nodes.py            Node implementations
│   ├── state.py            AgentState TypedDict
│   └── prompts.py          All LLM prompts in one place
├── db/
│   ├── snowflake_client.py Thread-safe singleton client
│   └── schema_loader.py    Introspection + compact rendering
├── guardrails/
│   └── validator.py        Rule-based SQL safety checks
├── tests/                  Offline pytest suite (mocked LLM & SF)
├── requirements.txt
├── .env.example
├── README.md
└── REFLECTION.md
```
