from threading import Lock
from unittest import TestCase

from build_dataset import SourceDoc
from evidence_expansion import expand_evidence
from phase1 import Hit
from phase3 import CONTEXT_BUDGET, Context, assemble, assemble_worker_context, complete_excerpt_boundaries
from phase4_retrieval import RankRequest, RetrievalSettings, Retriever, prefer_source_text
from same_document_evidence import extend_same_document


def document(doc_id: int, body: str) -> SourceDoc:
    return SourceDoc(id=doc_id, meetingId=doc_id, recordedAt="2026-11-11T09:00:00+09:00",
                     title="", topic="", summary=body[:10], body=body)


def character_offsets(text: str, **_options) -> dict:
    return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class WorkerContextTest(TestCase):
    def test_same_document_extension_keeps_original_excerpt_and_fits_spare_budget(self):
        original = "첫 회의 내용"
        relevant = "결제 승인 기준은 영수증 확인이다."
        body = original + "\n" + relevant + "\n별도 회의의 운영 일정을 길게 기록한다."
        source = {1: document(1, body), 2: document(2, relevant)}
        hit = Hit(documentId=1, field="body", start=0, end=len(original))
        passages = (Context(hit=hit, tokenCount=len(original)),)

        def tokenizer(text, **_options):
            return {"input_ids": list(text)}

        expected = original + "\n" + relevant
        expanded = extend_same_document(
            "결제 승인 기준은?", passages, source, tokenizer, budget=50, max_added=len(relevant) + 1,
        )

        self.assertEqual((Context(hit=hit.model_copy(update={"end": len(expected)}), tokenCount=len(expected)),),
                         expanded)

    def test_expansion_keeps_original_evidence_and_only_adds_from_selected_meetings(self):
        source = {doc: document(doc, "가" * 60).model_copy(update={"meetingId": meeting})
                  for doc, meeting in ((1, 10), (2, 10), (3, 10), (4, 20))}

        def tokenizer(text, **options):
            return {"input_ids": list(text), **character_offsets(text)}

        def hit(doc, start, end):
            return Hit(documentId=doc, field="body", start=start, end=end)

        primary = (Context(hit=hit(1, 0, 30), tokenCount=30),)
        support = (Context(hit=hit(2, 0, 30), tokenCount=30),)
        candidates = (hit(4, 0, 40), hit(1, 20, 50), hit(3, 0, 40))
        expanded, extra = expand_evidence(
            "근거", primary, support, candidates, source, tokenizer,
            budget=120, excerpt_tokens=40, supports_per_meeting=2,
        )
        self.assertEqual((Context(hit=hit(1, 0, 50), tokenCount=50),), expanded)
        self.assertEqual(support, extra[:1])
        self.assertEqual([2, 3], [item.hit.documentId for item in extra])
        self.assertEqual(120, sum(item.tokenCount for item in (*expanded, *extra)))

    def test_supporting_documents_try_each_meeting_before_a_second_document(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {
            i: document(i, "결제 취소 승인 기준").model_copy(update={"meetingId": meeting})
            for i, meeting in enumerate((1, 2, 1, 1, 2, 3), 1)
        }
        first, second, alternative, another, next_meeting, outside = (
            Hit(documentId=i, field="body", start=0, end=len(doc.body))
            for i, doc in retriever.source.items()
        )
        ranked = (outside, another, alternative, next_meeting, first, second)
        candidates = (alternative, another, next_meeting, outside, first, second)
        relevance = dict(zip(ranked, (10.0, 6.0, 5.5, 3.0, 2.0, 1.0), strict=True))

        selected = retriever.supporting_documents(
            "근거", (first, second), ranked, candidates, relevance,
            RetrievalSettings(supporting_rrf_bonus=5.0), per_meeting=2,
        )

        self.assertEqual((alternative, next_meeting, another), selected)

    def test_support_uses_the_shared_budget_and_does_not_replace_representatives(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {
            1: document(1, "가" * 70),
            2: document(2, "나" * 30).model_copy(update={"meetingId": 1}),
            3: document(3, "다" * 30),
            4: document(4, "라" * 30),
        }
        retriever.tokenizer = character_offsets
        hits = tuple(Hit(documentId=i, field="body", start=0, end=len(doc.body))
                     for i, doc in retriever.source.items() if i != 4)

        class Reranker:
            def predict(self, pairs, **options):
                return (5.0, 4.0, 3.0)

        settings = RetrievalSettings(
            context_budget=100, supporting_excerpt_tokens=20, supporting_reserve_tokens=20,
        )
        response = retriever.with_context(
            RankRequest(workspaceId=1, question="근거", allowedDocumentIds=(1, 2, 3), contextMode="synthesis"),
            hits, Reranker(), Lock(), Lock(), settings,
        )

        self.assertEqual([1, 3], [passage.documentId for passage in response.passages])
        self.assertEqual([2], [passage.documentId for passage in response.supportingPassages])
        self.assertEqual(100, sum(p.tokenCount for p in (*response.passages, *response.supportingPassages)))

    def test_extra_rrf_preserves_primary_and_first_support_passages(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {
            i: document(i, letter * 20).model_copy(update={"meetingId": 1})
            for i, letter in enumerate("가나다라", 1)
        }
        retriever.tokenizer = character_offsets
        candidates = tuple(Hit(documentId=i, field="body", start=0, end=20) for i in (4, 1, 3, 2))

        class Reranker:
            def predict(self, pairs, **options):
                return (5.5, 10.0, 6.0, 9.0)

        request = RankRequest(workspaceId=1, question="근거", allowedDocumentIds=(1, 2, 3, 4),
                              contextMode="synthesis")
        settings = RetrievalSettings(context_budget=60, supporting_excerpt_tokens=20,
                                     supporting_extra_per_meeting=True)
        before = retriever.with_context(request, candidates, Reranker(), Lock(), Lock(), settings)
        after = retriever.with_context(request, candidates, Reranker(), Lock(), Lock(),
                                       settings.model_copy(update={"extra_support_rrf_bonus": 5.0}))

        self.assertEqual(before.passages, after.passages)
        self.assertEqual(before.supportingPassages[0], after.supportingPassages[0])
        self.assertEqual([2, 3], [passage.documentId for passage in before.supportingPassages])
        self.assertEqual([2, 4], [passage.documentId for passage in after.supportingPassages])
        self.assertEqual(60, sum(p.tokenCount for p in (*after.passages, *after.supportingPassages)))

    def test_representative_can_change_document_without_changing_the_selected_meetings(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {
            1: document(1, "시스템 운영 지표 검토"),
            2: document(2, "결제 취소 승인 기준").model_copy(update={"meetingId": 1}),
            3: document(3, "분기 예산 검토"),
            4: document(4, "결제 취소 승인 기준"),
        }
        hits = tuple(Hit(documentId=i, field="body", start=0, end=len(doc.body))
                     for i, doc in retriever.source.items())
        first, alternate, second, outside = hits
        picked = retriever.choose_representatives(
            "결제 취소 승인 기준", (first, second), hits,
            {first: 5.0, alternate: 4.8, second: 5.0, outside: 10.0},
            RetrievalSettings(representative_lexical_bonus=3.0),
        )

        self.assertEqual((alternate, second), picked)

    def test_a_wider_source_window_preserves_one_contiguous_passage(self):
        body = "가" * 20 + "\n\n" + "나" * 20
        hit = Hit(documentId=1, field="body", start=0, end=20)
        extent = hit.model_copy(update={"end": len(body)})
        contexts = assemble_worker_context(
            (hit,), {1: document(1, body)}, character_offsets, {},
            budget=len(body), extents={hit: extent},
        )

        self.assertEqual((extent,), tuple(context.hit for context in contexts))
        self.assertEqual(len(body), contexts[0].tokenCount)

    def test_spare_slots_add_at_most_one_document_from_each_selected_meeting(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {i: document(i, "가" * 40) for i in range(1, 7)}
        for i, meeting in ((2, 1), (3, 1), (5, 4)):
            retriever.source[i] = retriever.source[i].model_copy(update={"meetingId": meeting})
        hits = {i: Hit(documentId=i, field="body", start=0, end=40) for i in retriever.source}

        selected = retriever.supplement((hits[1], hits[4]),
                                        tuple(hits[i] for i in (6, 1, 2, 3, 4, 5)), "document", {})

        self.assertEqual((hits[1], hits[4], hits[2], hits[5]), selected)

    def test_spare_slot_prefers_relevant_new_information_over_a_duplicate(self):
        retriever = Retriever.__new__(Retriever)
        duplicate = "환불 요청 접수 후 결제 승인 번호를 확인한다."
        complementary = "카드 취소는 영업일 기준 삼일 이내 처리한다."
        retriever.source = {
            i: document(i, text).model_copy(update={"meetingId": 1 if i < 4 else i})
            for i, text in enumerate((duplicate, duplicate, complementary, "다른 회의 문서"), 1)
        }
        hits = tuple(Hit(documentId=i, field="body", start=0, end=len(doc.body))
                     for i, doc in retriever.source.items())
        representative, copied, additional, outside = hits

        selected = retriever.supplement(
            (representative,), hits, "both",
            {representative: 8.0, copied: 7.5, additional: 7.0, outside: 9.0},
            overlap_penalty=2.0,
        )

        self.assertEqual((representative, additional), selected)

    def test_reuses_slots_only_after_other_selected_meetings_get_a_supplement(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {
            i: document(i, "결제 취소 환불 조건 확인").model_copy(update={"meetingId": meeting})
            for i, meeting in enumerate((1, 1, 1, 2, 2, 3), 1)
        }
        hits = tuple(Hit(documentId=i, field="body", start=0, end=len(doc.body))
                     for i, doc in retriever.source.items())
        first, first_extra, first_last, second, second_extra, outside = hits

        selected = retriever.supplement(
            (first, second), (outside, first_extra, first_last, second_extra), "both", {},
            reuse_spare_slots=True,
        )

        self.assertEqual((first, second, first_extra, second_extra, first_last), selected)

    def test_focused_excerpt_can_keep_a_matching_sentence_after_the_prefix(self):
        body = "가" * 50 + "\n환불 기준은 영업일 7일이다."
        docs = {1: document(1, body)}
        hit = Hit(documentId=1, field="body", start=0, end=len(body))
        contexts = assemble_worker_context((hit,), docs, character_offsets, {},
                                           budget=26, focus="환불 기준 알려줘")

        excerpt = body[contexts[0].hit.start:contexts[0].hit.end]
        self.assertIn("환불 기준은 영업일 7일이다.", excerpt)
        self.assertEqual(26, contexts[0].tokenCount)

    def test_finishes_a_clipped_sentence_without_exceeding_the_shared_budget(self):
        source = {1: document(1, "결정 안 함. 다음 회의에서 확정.")}
        hit = Hit(documentId=1, field="body", start=0, end=6)
        original = Context(hit=hit, tokenCount=6)

        def tokenizer(text, **options):
            return {"input_ids": list(text)}

        contexts, supports = complete_excerpt_boundaries((original,), (), source, tokenizer, 6, 24)
        self.assertEqual((original,), contexts)
        self.assertEqual((), supports)
        contexts, _ = complete_excerpt_boundaries((original,), (), source, tokenizer, 7, 24)
        self.assertEqual((0, 7, 7), (contexts[0].hit.start, contexts[0].hit.end, contexts[0].tokenCount))

    def test_fusion_supplement_keeps_original_candidates_and_adds_an_underrepresented_meeting(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {
            1: document(1, "가"), 2: document(2, "나").model_copy(update={"meetingId": 1}),
            3: document(3, "다").model_copy(update={"meetingId": 1}), 4: document(4, "라"),
        }
        hits = tuple(Hit(documentId=i, field="body", start=0, end=1) for i in (1, 2, 3, 4))
        settings = RetrievalSettings(fusion_depth=2, supplemental_candidates=1)
        self.assertEqual((hits[0], hits[1], hits[3]), retriever.fuse_candidates(hits, hits, settings))

    def test_fusion_fallback_extends_an_overlapping_excerpt_without_replacing_it(self):
        retriever = Retriever.__new__(Retriever)
        retriever.source = {1: document(1, "0123456789" * 10)}

        def tokenizer(text, **options):
            return {"input_ids": list(text), **character_offsets(text)}

        retriever.tokenizer = tokenizer
        original = Context(hit=Hit(documentId=1, field="body", start=0, end=40), tokenCount=40)
        candidate = Hit(documentId=1, field="body", start=30, end=90)
        contexts, supports = retriever.use_fusion_fallback(
            "근거", (original,), (), (candidate,), {candidate: 1.0}, RetrievalSettings(context_budget=72),
        )
        self.assertEqual((0, 72, 72), (contexts[0].hit.start, contexts[0].hit.end, contexts[0].tokenCount))
        self.assertEqual((), supports)

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
