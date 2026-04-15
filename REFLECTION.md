# Reflection

> **Live demo** → <https://census-agent-assignment.streamlit.app/>

## Development process

I started from the data, not the UI. Before writing any node code I
spent time understanding the Census schema: the `{YEAR}_CBG_{CODE}`
pattern, the role of the three metadata tables (`FIPS_CODES`,
`FIELD_DESCRIPTIONS`, `GEOGRAPHIC_DATA`), and the fact that table
names starting with a digit force double-quoting in every generated
SQL. Those details ended up being the single biggest driver of prompt
design: once the LLM knows *how* to reference these tables correctly,
everything else falls out of standard SQL reasoning.

Module order was deliberate:

1. **`db/`** first. Until Snowflake access and schema introspection
   worked, nothing downstream could be validated end-to-end.
2. **`guardrails/validator.py`** next. Pure Python, fastest to test,
   and a concrete contract ("no write statement ever reaches Snowflake")
   the rest of the system can depend on.
3. **`agent/nodes.py` + `graph.py`**, assembled once the pieces they
   compose were already trusted.
4. **`app.py`** last, a thin shell around the graph. This keeps the UI
   layer boring on purpose; the interesting behavior lives in the graph.

The second half of the work was iterative UX tightening: catching each
case where the agent gave a misleading or unfriendly response, tracing
it to the specific prompt / node / error path, and adjusting.

## Key architectural decisions

### Two-model split (Haiku + Sonnet)

The guardrail classifier runs on every message but only needs to emit one
of two labels. Sonnet for that task would be wasteful. Putting Haiku at
the front of the graph costs ~cents and ~1 s per off-topic interaction
instead of the 5–10 s a full Sonnet round-trip would take, which
matters both for cost and for user perception of snappiness.

The two places where judgment actually matters, authoring SQL and
grounding a natural-language answer in results, use Sonnet.

### LLM-authored SQL is treated as untrusted input

The rule-based validator in `guardrails/validator.py` is the most
important piece of safety code in the repo. It's ~60 lines that check
"starts with `SELECT`/`WITH`", no blocklist keywords (after stripping
string literals so queries *about* school dropouts still work), no
statement chaining, no comment injection. The
validator is untouched by the LLM chain at runtime, so prompt injection
in either the user message or in any row retrieved from Snowflake
cannot steer it. This is the reason we reject SQL statically before
executing it, rather than asking the LLM to self-check.

### Self-correcting SQL retry loop

The Census dataset has enough sub-table variation (e.g. SNAP is in
`B22010`, not the more obvious `B22001`) that the LLM occasionally
generates plausible-but-nonexistent column references. Rather than
fail the whole answer, the `execution_node` catches any SQL-
compilation-class error (invalid identifier, syntax error, ambiguous
column, "does not exist") and invokes a one-shot retry: it looks up
the *actual* column list for every referenced table from
`INFORMATION_SCHEMA.COLUMNS` and re-prompts Sonnet with the real
columns plus the original Snowflake error. Almost always resolves the
hallucination. Costs ~5–10 s extra on the rare retry path; keeps
worst-case latency well under the 60 s budget.

### Schema summary rather than full schema dump

The Census DB has hundreds of tables and tens of thousands of columns.
Dumping `INFORMATION_SCHEMA.COLUMNS` verbatim would blow past context
windows and cost. The `schema_loader` produces a ~3 KB summary: the
table-code legend, discovered years, the metadata-table names, small
samples of field descriptions, FIPS-join decomposition pattern
(`SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = STATE_FIPS`), and an enumeration
of all distinct `STATE` values (so the LLM knows PR is in scope).
Cached per-process with `lru_cache`.

### Four-sentinel output contract

Early experiments had the generator quietly hallucinate when the data
couldn't support a question, or blindly default to US-level when the
question lacked a geography. The fix was an explicit output contract:
Sonnet emits either SQL, or one of four sentinels that each route to a
dedicated synthesis path:

- `CANNOT_ANSWER: <reason>`: dataset genuinely doesn't cover it.
  Synthesis appends a helpful offer of the closest available alternative
  when one exists ("I only have 2019/2020, not 2010, but I could
  answer the 2019 → 2020 change instead").
- `NEED_CLARIFICATION: <question>`: question is too ambiguous to answer
  with a reasonable default ("What's the population?" with no geography).
- `SMALL_TALK: <reply>`: conversational closers, thanks, greetings.
  Skips SQL entirely; synthesis just emits the reply.
- SQL: runs normally.

This pattern turned "hallucination" from the default failure mode into
a first-class response the user can act on.

### Statistical correctness on aggregation

ACS median columns (`B19013e1` for income, `B25077e1` for house value,
`B01002e1` for age, `B25064e1` for rent) are topcoded: income caps at
$250,001, so a naive `MAX` over block groups hits the ceiling for every
large state and produces meaningless rankings. Plain `AVG` across
block-group medians systematically overstates, because wealthy areas
have more block groups per household.

The SQL-generation prompt explicitly requires a **household-weighted
average**: `SUM(median_col * weight_col) / SUM(weight_col)`, with the
right weight column per B-table family (e.g. `B19001e1` for income
medians, `B25001e1` for value/rent medians, `B01001e1` for age
medians). This took DC's "highest median household income" answer
from a misleading $107,536 (plain AVG) to $105,171 (weighted), which
matches the published ACS figure within rounding.

### Substitution disclosure

When the user asks for a metric we don't have but we can answer with a
close substitute (e.g. "average income" when only the median is available,
"longer time trend" when only 2 years exist), the synthesis prompt
requires the answer to open with an explicit disclosure sentence
("*I don't have average household income in this dataset, but I can
share the median…*"). Synthesis also sees the full conversation
history, so the disclosure fires correctly on follow-ups where the
original "average" question is in an earlier turn and the current user
turn is just "ca".

The SQL-generation prompt reinforces this: output column aliases must
name the true metric being computed, not the word the user used. So
when the user asks for "average" but the generator returns median,
the column alias is `median_household_income`, which then flows
through to synthesis as ground truth for the disclosure.

### Warm conversational register

The synthesis prompt ends every answer with a friendly offer of a
follow-up ("*Would you like to see how California compares…*") rather
than a flat user-perspective question. The offer is also constrained:
it must be something the dataset can actually deliver. Offers of
"longer-term trends" or "forecasts" or "crime rates" are explicitly
forbidden so the agent never sets up a follow-up it can't honor.

### Error handling as graph routing, not exceptions

Every node can set `state["error"]` and return normally. The graph
routes error states to the `synthesize` node, which picks the right
user-facing template. The synthesis node distinguishes:

- `snowflake_error`: true connection trouble
- `column_unavailable`: retry failed with a column-name issue
- `query_error`: retry failed with some other SQL bug
- `unsafe_sql`: validator rejected the LLM's SQL
- `execution_error` / `llm_unavailable`: unexpected runtime failures

Each maps to a distinct user-facing message. Nothing can crash the UI;
the worst case is the generic "something went wrong" message.

### UX polish

- **Typewriter streaming** via `st.write_stream()`. Answers render
  word-by-word on a 12ms delay, which perceptibly lowers "dead air"
  on a 6-second response even though total wall-clock time is the
  same.
- **Fade-in animations** on new chat bubbles (0.28s CSS keyframe) and a
  pulsing dot on the currently-running graph node, so the user sees
  progress during the pipeline rather than a blank spinner.
- **Auto-charts** when the result shape justifies one: top-10
  horizontal bar chart for category-value rankings, line chart for
  year-indexed series, grouped bars for state × year comparisons.
  Conservatively silent (no chart) on single-value answers and on
  results where all values are equal.
- **Theme pinned** in `.streamlit/config.toml` (`base = "light"`,
  Snowflake-blue `primaryColor`), so the UI is identical locally,
  on Streamlit Cloud, and across user browser themes. No more
  "white text on white sidebar" when someone's OS is in dark mode.

## What I would improve with more time

- **Server-side row-count / cost ceiling.** Right now `max_rows=1000` is
  a client cap. A `WHERE`-less `SELECT *` on a foot-traffic table would
  still spin Snowflake up unnecessarily. A `LIMIT`-injection check in the
  validator, or an `EXPLAIN`-based cost estimate, would close this.

- **Result caching.** Census tables are static per-year. A simple
  `(sql) -> rows` cache with a 24-hour TTL would make repeat questions
  near-instant. Trivial to add; I skipped it because it introduces a
  second place to look when rows seem stale.

- **Derivable-metrics lookup.** The agent declines to compute ratios it
  *could* compute from available columns (e.g. unemployment rate from
  B23025's `unemployed / in_labor_force`). A small dictionary
  mapping high-level metric names → (numerator column, denominator
  column) would let it answer these without risking a hallucinated
  formula.

- **Vector search over `METADATA_CBG_FIELD_DESCRIPTIONS`.** The dataset
  has thousands of field descriptions. Today the prompt includes a
  small *sample*, good enough for common questions like "population"
  or "median income," but weaker on long-tail queries like
  "self-employment rate among women with a graduate degree." A vector
  index over that metadata table would let us fetch the 10 most
  relevant field descriptions per query and slot them into the prompt.

- **Structured follow-up memory.** Today the agent passes raw message
  history. A lightweight summary ("last topic: CA population 2020,
  19.5M; last filter: household income") would be more robust to long
  conversations than relying on Sonnet to re-read every prior turn.

- **Multi-step plans for hard questions.** For "compare counties across
  two states with three different metrics," a plan → retrieve-N-times →
  aggregate loop would beat a single monolithic SQL statement. A
  `ReAct`-style iteration node would be a natural extension of the
  current graph.

- **Unit test for the compiled graph end-to-end** (with both LLM and
  Snowflake mocked), verifying the routing for each of the terminal
  paths. Today the nodes and router functions are tested individually,
  which catches most regressions but leaves graph-wiring bugs
  uncovered.

- **LLM-as-judge regression tests.** For production this would catch
  silent quality regressions when prompts are edited. For a take-home
  it was overkill; a deterministic golden-query harness with
  structural assertions (contains the right state abbrev, value within
  a plausible range, SQL includes the expected pattern) would be the
  right next step.

## Tradeoffs in "unanswerable" handling

The `CANNOT_ANSWER` path offers a partial alternative when one exists.
For example, asked "how did California's population change from 2010
to 2020?", the agent replies *"I only have 2019 and 2020 data, not
2010, but I could answer the 2019 to 2020 change instead."* This is more
helpful than a clean refusal, but it introduces a judgment call the LLM
has to make correctly: **is the substitute actually close enough to be
useful, or does it meaningfully misrepresent what the user asked?** A
2019 vs 2020 delta is not really a "decade of change" story, and a user
who accepts the offer without thinking gets an answer that's
technically responsive but substantively narrower than they wanted.

The same tradeoff exists on the metric-substitution path (median-for-
average, employment-status-count-for-unemployment-rate). The current
design leans helpful-over-strict, counting on the mandatory disclosure
sentence ("I don't have X, but I can share Y instead…") to let the
user decide whether the substitute is acceptable. If this ever fronted
a regulated or high-stakes context, the safer default would be "refuse
clean and name what's missing" without auto-offering alternatives.

## Edge cases identified but not fully addressed

- **Under-claiming on computable ratios.** Asked *"what's the current
  unemployment rate in Seattle?"*, the agent declines and offers raw
  employment statistics for King County instead. The ACS B23025 table
  actually does have `in_labor_force` and `unemployed` counts, so a
  rate could be computed as `unemployed / in_labor_force`. The agent
  chose the safe path. I left this as-is: over-caution is preferable
  to the failure mode of confidently computing a ratio from a
  misremembered table code. A targeted prompt nudge could teach it
  specifically about B23025's structure, but that's a narrow fix with
  a long tail of similar cases (poverty rate from B17, homeownership
  rate from B25, etc.), better addressed by a small
  "derivable-metrics" lookup than by expanding the system prompt.

- **Geography ambiguity.** "Kansas City" is in both KS and MO. The
  agent will produce *a* plausible answer, but no disambiguation
  question. A small name-collision table would be enough to trigger
  clarification.

- **Year-vs-table-coverage drift.** Not every ACS table appears in
  every year of the Census dataset. The generator can choose a year
  that doesn't have the table code it needs. Today this produces a
  row-count-zero response, handled gracefully, but it would be
  better to catch this during SQL generation by teaching the schema
  summary about per-year coverage.

- **Long-tail ACS codes.** Some B-tables have hundreds of sub-columns
  (e.g. B01001 has separate columns per age-bracket × sex). The
  synthesis node gets all of them and has to decide which are most
  relevant. Works well most of the time, but a user asking "how many
  women aged 25–29" is relying on Sonnet to pick the right column out
  of the result dict. A dedicated "code → column" mini-retriever
  would be more reliable.

- **Rate-limit and cold-start behavior.** Anthropic 529s and Snowflake
  cold warehouse spin-up both manifest as slow first requests. We
  retry nothing today for the LLM call; a single retry with
  exponential backoff on the Anthropic call would smooth this out at
  negligible cost.

- **Prompt-injection via query results.** A malicious row in a
  third-party dataset could in principle try to steer the synthesis
  LLM. The Census data is a trusted public source, but a defense-in-
  depth measure (e.g. wrapping row content in `<DATA>` tags and
  re-instructing the model to treat it as data-not-instructions) would
  be worth adding if this fronted a less-trusted source.

## Testing strategy

### What's covered today

- **`test_guardrails.py`**: the SQL validator's contract is the
  highest-leverage thing to test, since a bug here has security
  consequences. I covered valid `SELECT`/`WITH` forms, every blocked
  DDL/DML keyword, statement chaining, comment injection, and the
  critical false-positive case (a blocked keyword inside a string
  literal).

- **`test_schema_loader.py`**: introspection of empty table lists,
  latest-year picking, year/code extraction, bundle → string rendering
  (including the <3000-char size ceiling), and `lru_cache` hit-once
  behavior.

- **`test_nodes.py`**: each node with its LLM or Snowflake dependency
  mocked: guardrail pass/block/malformed/exception paths; SQL
  generation markdown stripping, `CANNOT_ANSWER` passthrough, LLM
  failure; validation safe/unsafe/sentinel paths; execution success,
  `SnowflakeError`, and unexpected-exception paths; synthesis
  off-topic / cannot-answer / connection-error / empty-results (the
  anti-hallucination contract) / rows-passed-verbatim; and routing
  functions for every branch.

All tests run without live credentials or network access. 58 tests
total, runtime ~1.3 s.

### Live performance measurement

`scripts/latency_check.py` walks 10 representative queries through the
compiled graph against real Snowflake + Anthropic, and reports p50 /
p95 / max latency and the routed path. Current numbers:

- **p50**: ~5 s
- **p95**: ~9 s
- **max**: ~9 s
- **0 of 10** exceed the 60 s requirement in the brief.

The slowest path is "text answer requiring Snowflake + two Sonnet
calls" (generation + synthesis). Guardrail-blocked and
cannot-answer paths return in 1–3 s.

### What I'd add next

- End-to-end graph tests that drive the compiled `StateGraph` with
  mocked LLM+SF and assert the terminal state for each leaf path.
- A property-style fuzz test for the SQL validator with random
  mixing of literal content, comments, and keywords.
- Golden-output regression tests for the synthesis prompt on a fixed
  set of (question, rows) pairs, mostly to catch silent regressions
  when the prompts are edited.
- A small set of live smoke tests that run against the real Snowflake
  dataset in CI on a schedule, so schema drift (e.g. a new year of
  data) is detected before a user finds it.

## Deployment

The app is hosted on Streamlit Community Cloud at
<https://census-agent-assignment.streamlit.app/>. Secrets (Snowflake
credentials, Anthropic API key) live only in Streamlit Cloud's Secrets
panel; the repo contains only placeholders. The theme is pinned in
`.streamlit/config.toml` so the UI renders identically on cloud as it
does locally. Redeploys are automatic on every `git push` to `main`.
