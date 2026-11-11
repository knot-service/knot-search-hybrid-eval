from unittest import TestCase

from build_dataset import SourceDoc
from phase1 import Hit
from phase3 import CONTEXT_BUDGET, assemble, assemble_worker_context
from phase4_retrieval import prefer_source_text


def document(doc_id: int, body: str) -> SourceDoc:
    return SourceDoc(id=doc_id, meetingId=doc_id, recordedAt="2026-11-11T09:00:00+09:00",
                     title="", topic="", summary=body[:10], body=body)


def character_offsets(text: str, **_options) -> dict:
    return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class WorkerContextTest(TestCase):
    def test_uses_short_passages_unused_budget_for_earlier_child(self):
        docs = {i: document(i, "가" * length) for i, length in enumerate((350, 320, 320, 20, 10), 1)}
        hits = tuple(Hit(documentId=i, field="body", start=0, end=len(doc.body)) for i, doc in docs.items())
        before = assemble(hits, docs, character_offsets, expand=True, balanced=True)
        after = assemble_worker_context(hits, docs, character_offsets, {})

        self.assertLess(before[0].hit.end, 310)
        self.assertEqual(hits, tuple(context.hit for context in after))
        self.assertLessEqual(sum(context.tokenCount for context in after), CONTEXT_BUDGET)

    def test_shortens_weak_passage_without_dropping_its_meeting(self):
        docs = {1: document(1, "가" * 320), 2: document(2, "나" * 320)}
        hits = tuple(Hit(documentId=i, field="body", start=0, end=320) for i in docs)

        contexts = assemble_worker_context(hits, docs, character_offsets, {hits[1]: 96})

        self.assertEqual([1, 2], [context.hit.documentId for context in contexts])
        self.assertEqual([320, 96], [context.tokenCount for context in contexts])

    def test_prefers_near_tied_body_without_changing_document_order(self):
        summary = Hit(documentId=1, field="summary", start=0, end=10)
        body = Hit(documentId=1, field="body", start=0, end=320)
        other = Hit(documentId=2, field="body", start=0, end=320)
        ranked = (summary, other, body)

        self.assertEqual((body, other), prefer_source_text(
            (summary, other), ranked, {summary: 2.0, other: 1.7, body: 1.2},
        ))
        self.assertEqual((summary, other), prefer_source_text(
            (summary, other), ranked, {summary: 2.0, other: 1.7, body: 0.8},
        ))
