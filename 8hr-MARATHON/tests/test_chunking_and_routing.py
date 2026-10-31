import json
from pathlib import Path

import pytest

from app.ingestion.chunking.splitter import chunk_text
from app.agents.nodes.planner import is_conversational

PROCESSED = Path(__file__).resolve().parents[1] / "processed_data"


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Pods scale horizontally.") == ["Pods scale horizontally."]


def test_single_newline_document_is_split():
    # The old splitter only split on "\n\n" and returned this as one 60K chunk.
    text = "\n".join(f"Line {i}: the kubelet restarts failed containers." for i in range(1200))
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    assert len(chunks) > 30
    assert max(len(c) for c in chunks) <= 1500 + 200 + 1


def test_unbreakable_token_is_hard_cut():
    chunks = chunk_text("x" * 5000, chunk_size=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000] * 5


def test_overlap_carries_previous_tail():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    for prev, cur in zip(chunks, chunks[1:]):
        first_word = cur.split(" ")[0]
        assert first_word in prev


def test_no_content_lost():
    text = "\n\n".join(f"Paragraph {i} " + "alpha beta " * 80 for i in range(40))
    joined = " ".join(chunk_text(text, chunk_size=800, overlap=0))
    for i in range(40):
        assert f"Paragraph {i} " in joined


@pytest.mark.skipif(not PROCESSED.exists(), reason="no processed_data snapshot")
def test_real_documents_respect_size_cap():
    # Re-chunk the text saved by the old ingestion run (which held 58K+ char chunks).
    for f in PROCESSED.glob("*/*.json"):
        old = json.loads(f.read_text(encoding="utf-8"))["chunks"]
        new = chunk_text("\n\n".join(old))
        assert max(len(c) for c in new) <= 1701, f.name


@pytest.mark.parametrize("raw", ["CONVERSATIONAL", "CONVERSATIONAL.", "'CONVERSATIONAL'",
                                 "conversational", "  `Conversational`  "])
def test_planner_conversational_variants(raw):
    assert is_conversational(raw)


@pytest.mark.parametrize("raw", ["kubernetes cronjob concurrencyPolicy",
                                 "Is CONVERSATIONAL memory supported in LangGraph?"])
def test_planner_search_queries_route_to_retriever(raw):
    assert not is_conversational(raw)
