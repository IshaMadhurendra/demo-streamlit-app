# Reflection

## Development process

I started from the data, not the UI. Before writing any node code I
spent time understanding the Census schema — the `{YEAR}_CBG_{CODE}`
pattern, the role of the three metadata tables (`FIPS_CODES`,
`FIELD_DESCRIPTIONS`, `GEOGRAPHIC_DATA`), and the fact that table
names starting with a digit force double-quoting in every generated
SQL. Those details ended up being the single biggest driver of prompt
design: once the LLM knows *how* to reference these tables correctly,
everything else falls out of standard SQL reasoning.

Module order was deliberate:

1. **`db/`** first — until Snowflake access and schema introspection
   worked, nothing downstream could be validated end-to-end.
2. **`guardrails/validator.py`** next — pure Python, fastest to test,
   and a concrete contract ("no write statement ever reaches Snowflake")
   the rest of the system can depend on.
3. **`agent/nodes.py` + `graph.py`** — assembled once the pieces they
   compose were already trusted.
4. **`app.py`** last — a thin shell around the graph. This keeps the UI
   layer boring on purpose; the interesting behavior lives in the graph.

## Key architectural decisions

### Two-model split (Haiku + Sonnet)

The guardrail classifier runs on every message but only needs to emit one
of two labels. Sonnet for that task would be wasteful. Putting Haiku at
the front of the graph costs ~cents and ~1 s per off-topic interaction
instead of the 5–10 s a full Sonnet round-trip would take — meaningful
both for cost and for user perception of snappiness.

The two places where judgment actually matters — authoring SQL and
grounding a natural-language answer in results — use Sonnet.

### LLM-authored SQL is treated as untrusted input

The rule-based validator in `guardrails/validator.py` is the most
important piece of safety code in the repo. It isn't clever; it's
~60 lines that check "starts with `SELECT`/`WITH`", no blocklist
keywords (after stripping string literals so queries *about* school
dropouts still work), no statement chaining, no comment injection. The
validator is untouched by the LLM chain at runtime — prompt injection
in either the user message or in any row retrieved from Snowflake
cannot steer it. This is the reason we reject SQL statically before
executing it, rather than asking the LLM to self-check.

### Schema summary rather than full schema dump

The Census DB has hundreds of tables and tens of thousands of columns.
Dumping `INFORMATION_SCHEMA.COLUMNS` verbatim would blow past context
windows and cost. The `schema_loader` produces a ~3 KB summary:
the table-code legend, discovered years, the metadata-table names, and
small samples of field descriptions and FIPS rows so the LLM sees the
shape of the data. Cached per-process with `lru_cache`.

### The "CANNOT_ANSWER" contract

Early experiments had the generator quietly hallucinate table names when
the data couldn't support a question. Adding an explicit "if you can't
do it, say `CANNOT_ANSWER: <reason>`" output convention, plus a
synthesis-node branch that renders it as a polite user-facing message,
cut the hallucination rate more than any other prompt tweak.

### Error handling as graph routing, not exceptions

Every node can set `state["error"]` and return normally. The graph
routes error states to the `synthesize` node, which picks the right
user-facing template. Nothing can crash the UI — the worst case is the
generic "something went wrong, try rephrasing" message.

## What I would improve with more time

- **Server-side row-count / cost ceiling.** Right now `max_rows=1000` is
  a client cap. A `WHERE`-less `SELECT *` on a foot-traffic table would
  still spin Snowflake up unnecessarily. A `LIMIT`-injection check in the
  validator, or an `EXPLAIN`-based cost estimate, would close this.

- **Result caching.** Census tables are static per-year. A simple
  `(sql) -> rows` cache with a 24-hour TTL would make repeat questions
  near-instant. Trivial to add; I skipped it because it introduces a
  second place to look when rows seem stale.

- **Vector search over `METADATA_CBG_FIELD_DESCRIPTIONS`.** The dataset
  has thousands of field descriptions. Today the prompt includes a
  small *sample* — good enough for common questions like "population"
  or "median income," but weaker on long-tail queries like
  "self-employment rate among women with a graduate degree." A vector
  index over that metadata table would let us fetch the 10 most
  relevant field descriptions per query and slot them into the prompt.

- **Structured follow-up memory.** Today the agent passes raw message
  history. A lightweight summary ("last topic: CA population 2020,
  19.5M; last filter: household income") would be more robust to long
  conversations than relying on Sonnet to re-read every prior turn.

- **More nuanced guardrail.** Today "pass" vs "block" is binary. A
  third class — "ambiguous, clarify first" — would let us explicitly
  route short prompts like "the population" through a clarification
  node instead of hoping the generator asks for help.

- **Multi-step plans for hard questions.** For "compare counties
  across two states," a plan→retrieve-N-times→aggregate loop would
  beat a single monolithic SQL statement. A `ReAct`-style iteration
  node would be a natural extension of the current graph.

- **Unit test for the compiled graph end-to-end** (with both LLM and
  Snowflake mocked), verifying the routing for each of the six
  terminal paths. Today the nodes and router functions are tested
  individually, which catches most regressions but leaves graph-wiring
  bugs uncovered.

## Tradeoffs in "unanswerable" handling

The `CANNOT_ANSWER` path now offers a partial alternative when one
exists — e.g. asked "how did California's population change from 2010
to 2020?", the agent replies *"I only have 2019 and 2020 data, not
2010 — but I could answer the 2019 to 2020 change instead."* This is
more helpful than a clean refusal, but it introduces a judgment call
the LLM has to make correctly: **is the substitute actually close
enough to be useful, or does it meaningfully misrepresent what the
user asked?** A 2019 vs 2020 delta is not really a "decade of change"
story, and a user who accepts the offer without thinking gets an
answer that's technically responsive but substantively narrower than
they wanted.

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
  rate from B25, etc.) — better addressed by a small
  "derivable-metrics" lookup than by expanding the system prompt.

- **Geography ambiguity.** "Kansas City" is in both KS and MO. The
  agent will produce *a* plausible answer, but no disambiguation
  question. A small name-collision table would be enough to trigger
  clarification.

- **Year-vs-table-coverage drift.** Not every ACS table appears in
  every year of the Census dataset. The generator can choose a year
  that doesn't have the table code it needs. Today this produces a
  row-count-zero response, handled gracefully — but it would be
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
  retry nothing today; a single retry with exponential backoff on the
  Anthropic call would smooth this out at negligible cost.

- **Prompt-injection via query results.** A malicious row in a
  third-party dataset could in principle try to steer the synthesis
  LLM. The Census data is a trusted public source, but a defense-in-
  depth measure (e.g. wrapping row content in `<DATA>` tags and
  re-instructing the model to treat it as data-not-instructions) would
  be worth adding if this fronted a less-trusted source.

## Testing strategy

### What's covered today

- **`test_guardrails.py`** — the SQL validator's contract is the
  highest-leverage thing to test, since a bug here has security
  consequences. I covered valid `SELECT`/`WITH` forms, every blocked
  DDL/DML keyword, statement chaining, comment injection, and the
  critical false-positive case (a blocked keyword inside a string
  literal).

- **`test_schema_loader.py`** — introspection of empty table lists,
  latest-year picking, year/code extraction, bundle → string rendering
  (including the <3000-char size ceiling), and `lru_cache` hit-once
  behavior.

- **`test_nodes.py`** — each node with its LLM or Snowflake dependency
  mocked: guardrail pass/block/malformed/exception paths; SQL
  generation markdown stripping, `CANNOT_ANSWER` passthrough, LLM
  failure; validation safe/unsafe/sentinel paths; execution success,
  `SnowflakeError`, and unexpected-exception paths; synthesis
  off-topic / cannot-answer / connection-error / empty-results (the
  anti-hallucination contract) / rows-passed-verbatim; and routing
  functions for every branch.

All tests run without live credentials or network access.

### What I'd add next

- End-to-end graph tests that drive the compiled `StateGraph` with
  mocked LLM+SF and assert the terminal state for each of the six
  leaf paths.
- A property-style fuzz test for the SQL validator with random
  mixing of literal content, comments, and keywords.
- Golden-output regression tests for the synthesis prompt on a fixed
  set of (question, rows) pairs, using Anthropic's caching so reruns
  are cheap — mostly to catch silent regressions when the prompts
  are edited.
- A small set of live smoke tests that run against the real Snowflake
  dataset in CI on a schedule, so schema drift (e.g. a new year of
  data) is detected before a user finds it.
