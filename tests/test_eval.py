"""Evaluation tests: retrieval hit-rate and answer accuracy against the golden set."""

import os
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent.parent / "eval" / "golden_questions.yaml"
CORPUS_PATH = Path(__file__).parent.parent / "eval" / "corpus"

# TODO(step-7): raise to 0.80 after tuning chunk size / top-k / prompt
RETRIEVAL_HIT_RATE_THRESHOLD = 0.70
# TODO(step-7): raise to 0.80 after tuning chunk size / top-k / prompt
ANSWER_HIT_RATE_THRESHOLD = 0.70


@pytest.fixture(scope="session")
def eval_store(tmp_path_factory):
    """Ingest the eval corpus once per session into a dedicated tmp store."""
    from rag_assistant.ingest import ingest_path
    from rag_assistant.store import ChunkStore

    db_path = tmp_path_factory.mktemp("eval_chroma")
    store = ChunkStore(db_path=db_path)
    ingest_path(CORPUS_PATH, store)
    return store


@pytest.fixture(scope="session")
def anthropic_client():
    """Create an Anthropic client; skip if ANTHROPIC_API_KEY is not set."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping eval tests")
    return anthropic.Anthropic(api_key=api_key)


@pytest.fixture(scope="session")
def eval_report(eval_store, anthropic_client):
    """Run the full evaluation once; all tests share the result."""
    from rag_assistant.eval_harness import load_golden_set, run_full_eval

    golden = load_golden_set(GOLDEN_PATH)
    return run_full_eval(golden, eval_store, client=anthropic_client)


@pytest.mark.slow
@pytest.mark.eval
def test_retrieval_hit_rate_meets_threshold(eval_report):
    assert eval_report.retrieval_hit_rate >= RETRIEVAL_HIT_RATE_THRESHOLD, (
        f"Retrieval hit-rate {eval_report.retrieval_hit_rate:.1%} "
        f"< threshold {RETRIEVAL_HIT_RATE_THRESHOLD:.1%}"
    )


@pytest.mark.slow
@pytest.mark.eval
def test_answer_hit_rate_meets_threshold(eval_report):
    assert eval_report.answer_hit_rate >= ANSWER_HIT_RATE_THRESHOLD, (
        f"Answer hit-rate {eval_report.answer_hit_rate:.1%} "
        f"< threshold {ANSWER_HIT_RATE_THRESHOLD:.1%}"
    )


@pytest.mark.slow
@pytest.mark.eval
def test_no_fallback_on_answerable_questions(eval_report):
    """System must never refuse on questions that are answerable from the corpus."""
    answerable = [r for r in eval_report.per_question if not r.question.expected_fallback]
    failures = [r for r in answerable if r.answer.was_fallback]
    assert not failures, (
        f"Unjustified fallback on {len(failures)} answerable question(s):\n"
        + "\n".join(f"  - {r.question.question}" for r in failures)
    )


@pytest.mark.slow
@pytest.mark.eval
def test_fallback_on_unanswerable_questions(eval_report):
    """System must always refuse on questions not in the corpus (anti-hallucination)."""
    unanswerable = [r for r in eval_report.per_question if r.question.expected_fallback]
    failures = [r for r in unanswerable if not r.answer.was_fallback]
    assert not failures, (
        f"Hallucination detected on {len(failures)} unanswerable question(s):\n"
        + "\n".join(
            f"  - Q: {r.question.question}\n    A: {r.answer.answer_text[:120]}"
            for r in failures
        )
    )
