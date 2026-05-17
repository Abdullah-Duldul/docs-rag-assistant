# Evaluation Harness

This directory contains the corpus, golden questions, and methodology for
measuring two core properties of the RAG system:

1. **Retrieval hit-rate** — does the vector store return the right chunks?
2. **Answer hit-rate** — does Claude produce a correct, cited answer from those chunks?

These are reported separately so failures can be diagnosed: a retrieval miss
is a chunking or embedding problem; an answer miss with a retrieval hit is a
prompt problem.

---

## Corpus

Seven pages from the Anthropic API documentation, fetched 2026-05-18 and
committed as static Markdown files. The eval does **not** re-fetch at test
time — results are reproducible without network access after initial setup.

| File | Source URL |
|------|-----------|
| `models-overview.md` | /docs/about-claude/models/overview |
| `getting-started.md` | /api/getting-started |
| `use-xml-tags.md` | /docs/build-with-claude/prompt-engineering/use-xml-tags |
| `vision.md` | /docs/build-with-claude/vision |
| `messages.md` | /api/messages |
| `rate-limits.md` | /api/rate-limits |
| `tool-use-overview.md` | /docs/build-with-claude/tool-use/overview |

---

## Golden Questions

Ten questions in `golden_questions.yaml`: 7 answerable from the corpus,
3 designed to trigger the anti-hallucination fallback.

### Answerable (7)

| Q# | Question | Expected substring | Source file |
|----|----------|--------------------|-------------|
| 1 | What HTTP status code does the Anthropic API return when you exceed the rate limit? | `429` | rate-limits.md |
| 2 | What HTTP header must be included in all Anthropic API requests for authentication? | `x-api-key` | getting-started.md |
| 3 | What does Anthropic say XML tags help Claude do when processing complex prompts? | `unambiguously` | use-xml-tags.md |
| 4 | What encoding type does the Claude vision API support for passing images inline in a message? | `base64` | vision.md |
| 5 | What parameter in the Messages API is used to provide a system prompt to Claude? | `system` | messages.md |
| 6 | What stop_reason does the API return when Claude wants to use a client tool? | `tool_use` | tool-use-overview.md |
| 7 | Which Claude model has the largest context window currently available, and how large is it? | `1M tokens` | models-overview.md |

### Unanswerable (3)

| Q# | Question | Why unanswerable |
|----|----------|-----------------|
| 8 | What is Anthropic's stock ticker symbol? | Anthropic is a private company; no ticker in corpus |
| 9 | What is the API endpoint for generating images with Claude? | Anthropic has no image generation API; tests OpenAI knowledge bleed |
| 10 | What is Anthropic's office address in San Francisco? | Office addresses not in API documentation |

---

## Methodology

### Hit-rate definition

A **retrieval HIT** requires both conditions simultaneously (conjunction, not OR):
1. `expected_source_file` is among the filenames of top-5 retrieved chunks
2. `expected_substring` appears (case-insensitive) in at least one retrieved chunk text

This is intentionally strict. Source file correct + wrong text chunk indicates
a real chunking problem; it should register as a miss, not a partial hit.

### Answer hit definition

A **answer HIT** for answerable questions: `expected_substring` appears
case-insensitively in Claude's response AND Claude did not output the
anti-hallucination fallback.

A **answer HIT** for unanswerable questions: Claude's response contains the
exact fallback string:

> I don't have information about that in the indexed documents.

### Thresholds

```python
# tests/test_eval.py
# TODO(step-7): raise to 0.80 after tuning chunk size / top-k / prompt
RETRIEVAL_HIT_RATE_THRESHOLD = 0.70
ANSWER_HIT_RATE_THRESHOLD    = 0.70
```

The fallback tests use a hard 100% threshold — any unjustified refusal
or hallucination is a test failure regardless of the numeric thresholds.

### Why two separate metrics?

The Monte Carlo testing protocol (see `prompting/testing-prompts.md` in the
Nick KB) requires a pre-defined "good enough" bar measured across many runs.
Retrieval and answer are separate bars so that:
- A retrieval drop (embedding drift, chunk-size change) is immediately visible
- A prompt regression (answer quality drops while retrieval holds) is separately visible

---

## Measured Metrics (baseline, 2026-05-18)

| Metric | Score | Threshold |
|--------|-------|-----------|
| Retrieval hit-rate | **100% (7/7)** | ≥ 70% (→ 80% in step 7) |
| Answer hit-rate | **100% (7/7)** | ≥ 70% (→ 80% in step 7) |
| Anti-hallucination | **100% (3/3)** | 100% (hard) |

Model: `claude-haiku-4-5-20251001`

---

## Cost

~$0.03 per full eval run (30k input tokens + 365 output tokens at Haiku 4.5
pricing: $1/MTok input, $5/MTok output). Safe to run on every PR.

---

## Running the eval

```bash
# CLI (ingests corpus, runs 10 questions, prints table)
uv run rag eval

# pytest (requires ANTHROPIC_API_KEY in .env or environment)
uv run pytest -m eval -v

# Override threshold
uv run rag eval --threshold 0.80
```
