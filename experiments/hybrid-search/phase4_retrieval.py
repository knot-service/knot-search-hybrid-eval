"""Loopback-only scoped retrieval and evidence selection for the Spring Worker."""
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, datetime
from hashlib import sha256
import os
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable, Literal

from psycopg import Connection
from zoneinfo import ZoneInfo

import torch
from fastapi import FastAPI, Request
from pydantic import Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from starlette.datastructures import State
from starlette.middleware.trustedhost import TrustedHostMiddleware
from transformers import PreTrainedTokenizerFast

from build_dataset import Frozen, SourceDoc, documents
from database import connect, vector_literal
from experiment import MODEL, REVISION, tokens
from evidence_expansion import compact_overlaps, expand_evidence
from live_index import LiveIndex, live_documents, source_versions
from meeting_evidence import reserve_meeting_evidence
from phase1 import Hit, chunks, prepare
from phase2 import DEPTH, fuse, ms
from phase3 import (
    CONTEXT_BUDGET, CONTEXT_COUNT, RERANKER, RERANKER_REVISION, Context,
    adjacent_paragraphs, assemble_worker_context, complete_excerpt_boundaries, material,
)
from same_document_evidence import extend_same_document, select_short_passages

BODY_SCORE_MARGIN = 1.0
WEAK_SCORE_GAP = 6.25
WEAK_PASSAGE_TOKENS = 96
KOREA = ZoneInfo("Asia/Seoul")
MAX_EXPANSION_BUDGET = 6144


class RetrievalSettings(Frozen):
    supplement: Literal["none", "document", "chunk", "both"] = "none"
    supplement_overlap_penalty: float = 0.0
    reuse_spare_slots: bool = False
    representative_body_bonus: float = 0.0
    representative_lexical_bonus: float = 0.0
    representative_rrf_bonus: float = 0.0
    representatives_only_when_full: bool = True
    expand_source_window: bool = False
    supporting_excerpt_tokens: int = 0
    supporting_reserve_tokens: int = 0
    supporting_lexical_bonus: float = 0.0
    supporting_rrf_bonus: float = 0.0
    supporting_extra_per_meeting: bool = False
    supporting_shared_slots: bool = False
    extra_support_rrf_bonus: float | None = None
    extra_support_reserve_tokens: int = Field(default=0, ge=0)
    supporting_in_questions: bool = False
    keyword_depth: int = DEPTH
    vector_depth: int = DEPTH
    fusion_depth: int | None = DEPTH
    fusion_reserve: int = 0
    context_budget: int = CONTEXT_BUDGET
    weak_passage_tokens: int | None = WEAK_PASSAGE_TOKENS
    sentence_focus: bool = False
    complementary_bonus: float = 0
    spare_evidence: Literal["none", "chunk", "document", "both"] = "none"
    boundary_tokens: int = 0
    supplemental_candidates: int = 0
    late_fusion: bool = False
    expansion_budget: int = 0
    expansion_excerpt_tokens: int = 160
    expansion_slots: int = 5
    expansion_supports_per_meeting: int = 2
    expansion_compact: bool = False
    expansion_order: Literal["rank", "documents"] = "rank"
    same_document_tokens: int = Field(default=0, ge=0, le=768)
    short_passage_reserve: int = Field(default=0, ge=0, le=512)
    short_passages_spare_only: bool = False
    rerank_supplemental: bool = False
    discovery_depth: int = Field(default=0, ge=0, le=160)
    discovery_rerank_limit: int = Field(default=48, ge=1, le=80)
    discovery_compact: bool = True
    discovery_window_rerank: bool = False
    discovery_uncovered_only: bool = False
    discovery_excerpt_tokens: int = Field(default=96, ge=96, le=320)
    discovery_existing_first: bool = False
    meeting_evidence_first: bool = False
    meeting_evidence_topic_status: bool = False
    meeting_evidence_reclaim: int = Field(default=0, ge=0, le=256)


BASELINE = RetrievalSettings()
SPARE_SLOT_SETTINGS = RetrievalSettings(
    supplement="both",
    supplement_overlap_penalty=1.0,
    keyword_depth=40,
    context_budget=2048,
    sentence_focus=True,
)
EXPANDED_REPRESENTATIVE_SETTINGS = SPARE_SLOT_SETTINGS.model_copy(update={
    "representative_body_bonus": 1.0,
    "representative_lexical_bonus": 24.0,
    "representative_rrf_bonus": 1.0,
    "representatives_only_when_full": False,
    "expand_source_window": True,
    "context_budget": 3072,
})
SUPPORTING_DOCUMENT_SETTINGS = SPARE_SLOT_SETTINGS.model_copy(update={
    "supporting_excerpt_tokens": 220,
    "supporting_lexical_bonus": 8.0,
    "supporting_in_questions": True,
})
SUPPORTING_SELECTION_SETTINGS = SUPPORTING_DOCUMENT_SETTINGS.model_copy(update={
    "supporting_extra_per_meeting": True,
})
PROTECTED_SUPPORT_SETTINGS = SUPPORTING_SELECTION_SETTINGS.model_copy(update={"extra_support_rrf_bonus": 5.0})
EVIDENCE_SELECTION_SETTINGS = PROTECTED_SUPPORT_SETTINGS.model_copy(update={
    "complementary_bonus": 8.0,
    "spare_evidence": "chunk",
    "boundary_tokens": 24,
    "supplemental_candidates": 4,
    "late_fusion": True,
})

EVIDENCE_EXPANSION_SETTINGS = EVIDENCE_SELECTION_SETTINGS.model_copy(update={
    "expansion_budget": 3072,
    "expansion_excerpt_tokens": 220,
    "expansion_slots": 7,
    "expansion_supports_per_meeting": 3,
})


class RankRequest(Frozen):
    workspaceId: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=4000)
    allowedDocumentIds: tuple[int, ...] = Field(max_length=10_000)
    contextMode: Literal["none", "question", "synthesis"] = "none"
    recentMeetings: int = Field(default=0, ge=0, le=10)
    preferredDocumentId: int | None = Field(default=None, gt=0)
    inferredComparisonDays: tuple[date, ...] = Field(default=(), max_length=2)


class Passage(Hit):
    quote: str
    tokenCount: int


class RankResponse(Frozen):
    documentIds: tuple[int, ...]
    passages: tuple[Passage, ...] = ()
    supportingPassages: tuple[Passage, ...] = ()
    shortPassages: tuple[Passage, ...] = ()
    discoveryPassages: tuple[Passage, ...] = ()
    rerankMs: float = 0
    contextMs: float = 0


class Retriever:
    def __init__(self, model: SentenceTransformer, workspace_id: int, *,
                 live_source: tuple[SourceDoc, ...] | None = None):
        source = documents(workspace_id) if live_source is None else live_source
        if not source:
            raise RuntimeError(f"No snapshot for workspace {workspace_id}")
        tokenizer = model.tokenizer
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise RuntimeError("The embedding tokenizer must return character offsets.")
        units = tuple(child for doc in source for child in chunks(doc, "fixed", tokenizer))
        self.source = {doc.id: doc for doc in source}
        self.tokenizer = tokenizer
        self.hits = tuple(Hit(documentId=child.documentId, field=child.field,
                              start=child.start, end=child.end) for child in units)
        self.keyword = BM25Okapi([tokens(child.text) for child in units])
        self.workspace_id = workspace_id
        if live_source is not None:
            return
        with connect() as db:
            existing = {Hit(documentId=doc, field=field, start=left, end=right)
                        for doc, field, left, right in db.execute(
                            """select document_id,field,start_offset,end_offset from workspace1_chunks
                               where workspace_id=%s and mode='fixed'""", (workspace_id,),
                        )}
            if existing != set(self.hits):
                prepare(db, source, model, workspace_id=workspace_id, modes=("fixed",))

    def rank(self, question: str, embedding: str, allowed_ids: tuple[int, ...],
             settings: RetrievalSettings = BASELINE,
             preferred_document_id: int | None = None, *,
             db: Connection | None = None) -> tuple[Hit, ...]:
        if preferred_document_id in allowed_ids:
            allowed_ids = (preferred_document_id,)
        keyword, vector = (self.rank_sources(question, embedding, allowed_ids, settings) if db is None else
                           self.rank_sources(question, embedding, allowed_ids, settings, db=db))
        return self.fuse_candidates(keyword, vector, settings)

    def fuse_candidates(self, keyword: tuple[Hit, ...], vector: tuple[Hit, ...],
                        settings: RetrievalSettings) -> tuple[Hit, ...]:
        fused = fuse(keyword, vector, settings.fusion_depth, settings.fusion_reserve)
        if not settings.supplemental_candidates or settings.fusion_depth is None:
            return fused
        meeting_counts = Counter(self.source[hit.documentId].meetingId for hit in fused)
        documents = {hit.documentId for hit in fused}
        extras = []
        for hit in fuse(keyword, vector, None):
            meeting = self.source[hit.documentId].meetingId
            if settings.late_fusion:
                if hit in fused or hit.field != "body":
                    continue
            elif meeting_counts[meeting] > 1 or hit.documentId in documents:
                continue
            if hit not in fused:
                extras.append(hit)
                documents.add(hit.documentId)
                meeting_counts[meeting] += 1
                if len(extras) == settings.supplemental_candidates:
                    break
        return (*fused, *extras)

    def rank_sources(self, question: str, embedding: str, allowed_ids: tuple[int, ...],
                     settings: RetrievalSettings, *, db: Connection | None = None
                     ) -> tuple[tuple[Hit, ...], tuple[Hit, ...]]:
        allowed = set(allowed_ids) & self.source.keys()
        params = (self.workspace_id, list(allowed), embedding, settings.vector_depth)
        if db is None:
            with connect() as snapshot_db:
                rows = snapshot_db.execute("""
                    SELECT document_id, field, start_offset, end_offset
                    FROM workspace1_chunks WHERE workspace_id=%s AND mode='fixed' AND document_id=any(%s)
                    ORDER BY embedding <=> %s::vector,document_id,field,start_offset LIMIT %s
                    """, params).fetchall()
        else:
            rows = db.execute("""
                SELECT document_id, field, start_offset, end_offset
                FROM hybrid_document_index_chunks WHERE workspace_id=%s AND document_id=any(%s)
                ORDER BY embedding <=> %s::vector,document_id,field,start_offset LIMIT %s
                """, params).fetchall()
        vector = tuple(Hit(documentId=doc, field=field, start=left, end=right) for doc, field, left, right in rows)
        scores = self.keyword.get_scores(tokens(question))
        matching = (i for i, hit in enumerate(self.hits) if hit.documentId in allowed and scores[i] > 0)
        keyword = tuple(self.hits[i] for i in sorted(matching, key=lambda i: (-scores[i], i))[:settings.keyword_depth])
        return keyword, vector

    def with_context(self, request: RankRequest, candidates: tuple[Hit, ...],
                     reranker: CrossEncoder, rerank_lock: Lock, tokenizer_lock: Lock,
                     settings: RetrievalSettings = BASELINE, *,
                     on_selected: Callable[[tuple[Hit, ...]], None] | None = None,
                     on_supporting: Callable[[tuple[Hit, ...], tuple[Hit, ...], int], None] | None = None) -> RankResponse:
        document_ids = tuple(dict.fromkeys(hit.documentId for hit in candidates))
        if request.contextMode == "none" or not candidates:
            return RankResponse(documentIds=document_ids)

        core_candidates = candidates[:settings.fusion_depth] if settings.late_fusion else candidates
        first_pass = candidates if settings.rerank_supplemental else core_candidates
        pairs = [(request.question, f"{self.source[hit.documentId].title}\n"
                  f"{material(self.source[hit.documentId], hit)[hit.start:hit.end]}") for hit in first_pass]
        started = perf_counter()
        with rerank_lock:
            scores = reranker.predict(pairs, batch_size=10, show_progress_bar=False)
        relevance = dict(zip(first_pass, map(float, scores), strict=True))
        ranked = tuple(sorted(first_pass, key=lambda hit: -relevance[hit]))
        rerank_ms = ms(started)
        started = perf_counter()
        synthesis = request.contextMode == "synthesis"
        selected = self.by_document(ranked) if not synthesis or request.inferredComparisonDays else ()
        if synthesis and request.inferredComparisonDays:
            covered_days = {datetime.fromisoformat(self.source[hit.documentId].recordedAt).astimezone(KOREA).date()
                            for hit in selected}
            synthesis = not set(request.inferredComparisonDays).issubset(covered_days)
        if synthesis:
            selected = self.by_meeting(ranked, request.recentMeetings)
        selected = prefer_source_text(selected, ranked, relevance)
        if synthesis and (
            not settings.representatives_only_when_full or len(selected) == CONTEXT_COUNT
        ) and (settings.representative_body_bonus or settings.representative_lexical_bonus
               or settings.representative_rrf_bonus):
            selected = self.choose_representatives(request.question, selected, first_pass, relevance, settings)
        if synthesis and settings.supplement != "none":
            selected = self.supplement(selected, ranked, settings.supplement, relevance,
                                       overlap_penalty=settings.supplement_overlap_penalty,
                                       reuse_spare_slots=settings.reuse_spare_slots)
        if on_selected is not None:
            on_selected(selected)
        best = relevance[ranked[0]]
        limits = {hit: settings.weak_passage_tokens for hit in selected
                  if settings.weak_passage_tokens is not None and best >= 0 and best - relevance[hit] > WEAK_SCORE_GAP}
        extents = {}
        if settings.expand_source_window:
            for hit in selected:
                siblings = tuple(candidate for candidate in ranked if candidate.documentId == hit.documentId
                                 and candidate.field == hit.field)
                if len(siblings) > 1:
                    combined = hit.model_copy(update={
                        "start": min(item.start for item in siblings),
                        "end": max(item.end for item in siblings),
                    })
                    extents[hit] = adjacent_paragraphs(self.source[hit.documentId], combined)
        with tokenizer_lock:
            focus = request.question if settings.sentence_focus else ""
            contexts = assemble_worker_context(
                selected, self.source, self.tokenizer, limits, budget=settings.context_budget,
                focus=focus, extents=extents,
            )
            supports = ()
            if settings.supporting_excerpt_tokens and (
                request.contextMode == "synthesis" or settings.supporting_in_questions
            ):
                primary_tokens = sum(item.tokenCount for item in contexts)
                remaining = settings.context_budget - primary_tokens
                support_budget = remaining + settings.supporting_reserve_tokens
                primary_hits = tuple(item.hit for item in contexts)
                extras = ()
                if support_budget:
                    extras = self.supporting_documents(
                        request.question, primary_hits, ranked, first_pass, relevance, settings,
                        per_meeting=2 if settings.supporting_shared_slots else 1,
                    )
                    extra_settings = settings
                    if settings.extra_support_rrf_bonus is not None:
                        extra_settings = settings.model_copy(update={"supporting_rrf_bonus": settings.extra_support_rrf_bonus})
                    can_add_support = settings.supporting_extra_per_meeting and not settings.supporting_shared_slots
                    reserved = 0
                    if can_add_support and settings.extra_support_reserve_tokens:
                        reserve_candidates = self.supporting_documents(
                            request.question, (*primary_hits, *extras), ranked, first_pass, relevance, extra_settings,
                        )[:CONTEXT_COUNT - len(extras)]
                        reserved = min(len(reserve_candidates) * settings.extra_support_reserve_tokens, support_budget // 3)
                    supports = assemble_worker_context(
                        extras, self.source, self.tokenizer,
                        dict.fromkeys(extras, settings.supporting_excerpt_tokens),
                        budget=support_budget - reserved, focus=request.question,
                    )
                    extra_tokens = sum(item.tokenCount for item in supports)
                    spare = support_budget - extra_tokens
                    if can_add_support and spare and len(supports) < CONTEXT_COUNT:
                        extras_after = self.supporting_documents(
                            request.question, (*primary_hits, *(item.hit for item in supports)),
                            ranked, first_pass, relevance, extra_settings,
                        )[:CONTEXT_COUNT - len(supports)]
                        supports += assemble_worker_context(
                            extras_after, self.source, self.tokenizer,
                            dict.fromkeys(extras_after, settings.supporting_excerpt_tokens),
                            budget=spare, focus=request.question,
                        )
                        extra_tokens = sum(item.tokenCount for item in supports)
                    if extra_tokens > remaining:
                        contexts = assemble_worker_context(
                            selected, self.source, self.tokenizer, limits,
                            budget=settings.context_budget - extra_tokens, focus=focus, extents=extents,
                        )
                if on_supporting is not None:
                    on_supporting(primary_hits, extras, support_budget)
            if settings.spare_evidence != "none":
                contexts, supports = self.add_spare_evidence(
                    request.question, contexts, supports, ranked, relevance, settings,
                )
            if settings.late_fusion:
                fallback = candidates[len(core_candidates):]
                fallback_scores = {hit: -index for index, hit in enumerate(fallback)}
                contexts, supports = self.use_fusion_fallback(
                    request.question, contexts, supports, fallback, fallback_scores, settings,
                )
            if settings.boundary_tokens:
                contexts, supports = complete_excerpt_boundaries(
                    contexts, supports, self.source, self.tokenizer, settings.context_budget, settings.boundary_tokens,
                )
            short = ()
            if settings.expansion_budget:
                reserve = 0 if settings.short_passages_spare_only else settings.short_passage_reserve
                contexts, supports = expand_evidence(
                    request.question, contexts, supports, candidates, self.source, self.tokenizer,
                    budget=settings.expansion_budget - reserve,
                    excerpt_tokens=settings.expansion_excerpt_tokens,
                    passages_per_kind=settings.expansion_slots,
                    supports_per_meeting=settings.expansion_supports_per_meeting,
                    compact=settings.expansion_compact, order=settings.expansion_order,
                )
                if reserve:
                    short = select_short_passages(
                        request.question, (*contexts, *supports), self.source, self.tokenizer,
                        budget=reserve,
                    )
                if settings.same_document_tokens:
                    excerpts = extend_same_document(
                        request.question, (*contexts, *supports), self.source, self.tokenizer,
                        budget=settings.expansion_budget - sum(item.tokenCount for item in short),
                        max_added=settings.same_document_tokens,
                    )
                    contexts, supports = excerpts[:len(contexts)], excerpts[len(contexts):]
                if settings.short_passages_spare_only and settings.short_passage_reserve:
                    available = settings.expansion_budget - sum(item.tokenCount for item in (*contexts, *supports))
                    if available > 0:
                        short = select_short_passages(
                            request.question, (*contexts, *supports), self.source, self.tokenizer,
                            budget=min(available, settings.short_passage_reserve),
                        )
        return RankResponse(documentIds=document_ids, passages=self.passages(contexts),
                            supportingPassages=self.passages(supports),
                            shortPassages=self.passages(short),
                            rerankMs=rerank_ms, contextMs=ms(started))

    def discover(self, question: str, keyword: tuple[Hit, ...], vector: tuple[Hit, ...],
                 candidates: tuple[Hit, ...], baseline: RankResponse, reranker: CrossEncoder,
                 rerank_lock: Lock, tokenizer_lock: Lock, settings: RetrievalSettings) -> RankResponse:
        """Rerank additional body candidates after fixing the original selection."""
        if not baseline.passages:
            return baseline
        existing = set(candidates)
        contexts = tuple(
            tuple(Context(hit=Hit.model_validate(p.model_dump()), tokenCount=p.tokenCount) for p in passages)
            for passages in (baseline.passages, baseline.supportingPassages, baseline.shortPassages)
        )
        with tokenizer_lock:
            if settings.discovery_compact:
                contexts = tuple(compact_overlaps(group, self.source, self.tokenizer) for group in contexts)
            remaining = settings.expansion_budget - sum(item.tokenCount for group in contexts for item in group)
        if remaining < 24:
            return baseline
        extra = tuple(hit for hit in fuse(keyword, vector, None) if hit.field == "body" and hit not in existing)
        extra = extra[:settings.discovery_rerank_limit]
        if not extra:
            return baseline
        original = tuple(item.hit for group in contexts for item in group)
        started = perf_counter()
        pairs = [(question, f"{self.source[hit.documentId].title}\n"
                  f"{material(self.source[hit.documentId], hit)[hit.start:hit.end]}") for hit in extra]
        with rerank_lock:
            scores = reranker.predict(pairs, batch_size=10, show_progress_bar=False)
        ranked = tuple(hit for hit, _ in sorted(zip(extra, scores, strict=True), key=lambda pair: -pair[1]))
        window_rerank = settings.discovery_window_rerank and (
            not settings.discovery_existing_first or remaining < settings.discovery_excerpt_tokens
        )
        if window_rerank:
            with tokenizer_lock:
                windows = self.discovery_windows(ranked[:12], original if settings.discovery_uncovered_only else ())
            if windows:
                with rerank_lock:
                    scores = reranker.predict([
                        (question, f"{self.source[hit.documentId].title}\n"
                         f"{material(self.source[hit.documentId], hit)[hit.start:hit.end]}") for hit in windows
                    ], batch_size=10, show_progress_bar=False)
                ranked = tuple(hit for hit, _ in sorted(zip(windows, scores, strict=True), key=lambda pair: -pair[1]))
        if settings.discovery_existing_first and not window_rerank:
            selected_documents = {hit.documentId for hit in original}
            ranked = tuple(sorted(ranked, key=lambda hit: hit.documentId not in selected_documents))
        rerank_ms = ms(started)
        started = perf_counter()
        discovered: list[Context] = []
        added_documents: set[int] = set()
        spare = remaining
        with tokenizer_lock:
            for hit in ranked:
                if spare < 24 or len(discovered) == 4:
                    break
                if hit.documentId in added_documents:
                    continue
                new = assemble_worker_context((hit,), self.source, self.tokenizer, {},
                                              budget=min(spare, settings.discovery_excerpt_tokens), focus=question,
                                              extents={hit: hit} if window_rerank else None)
                if not new:
                    continue
                piece = new[0].hit
                if any(old.documentId == piece.documentId and old.field == piece.field
                       and old.start <= piece.start and old.end >= piece.end for old in original):
                    continue
                discovered.extend(new)
                spare -= new[0].tokenCount
                added_documents.add(hit.documentId)
        if not discovered:
            return baseline.model_copy(update={"rerankMs": baseline.rerankMs + rerank_ms})
        primary, supports, short = contexts
        return baseline.model_copy(update={
            "documentIds": tuple(dict.fromkeys((*baseline.documentIds, *(item.hit.documentId for item in discovered)))),
            "passages": self.passages(primary), "supportingPassages": self.passages(supports),
            "shortPassages": self.passages(short), "discoveryPassages": self.passages(tuple(discovered)),
            "rerankMs": baseline.rerankMs + rerank_ms, "contextMs": baseline.contextMs + ms(started),
        })

    def discovery_windows(self, candidates: tuple[Hit, ...], covered: tuple[Hit, ...]) -> tuple[Hit, ...]:
        windows = []
        for hit in candidates:
            cursor = hit.start
            gaps = []
            for item in sorted((p for p in covered if p.documentId == hit.documentId and p.field == hit.field
                                and p.start < hit.end and p.end > hit.start), key=lambda p: p.start):
                if cursor < item.start:
                    gaps.append((cursor, item.start))
                cursor = max(cursor, item.end)
            if cursor < hit.end:
                gaps.append((cursor, hit.end))
            for left, right in gaps:
                offsets = self.tokenizer(material(self.source[hit.documentId], hit)[left:right],
                                         add_special_tokens=False, return_offsets_mapping=True,
                                         verbose=False)["offset_mapping"]
                for start in range(0, len(offsets), 32):
                    end = min(start + 96, len(offsets))
                    windows.append(hit.model_copy(update={
                        "start": left + offsets[start][0], "end": left + offsets[end - 1][1],
                    }))
                    if end == len(offsets):
                        break
        return tuple(dict.fromkeys(windows))

    def use_fusion_fallback(self, question: str, contexts: tuple[Context, ...], supports: tuple[Context, ...],
                            candidates: tuple[Hit, ...], relevance: dict[Hit, float],
                            settings: RetrievalSettings) -> tuple[tuple[Context, ...], tuple[Context, ...]]:
        existing = (*contexts, *supports)
        spare = min(220, settings.context_budget - sum(item.tokenCount for item in existing))
        if spare < 32:
            return contexts, supports
        for hit in sorted(candidates, key=relevance.__getitem__, reverse=True):
            for index, item in enumerate(existing):
                original = item.hit
                if (hit.documentId != original.documentId or hit.field != original.field
                        or not original.start <= hit.start < original.end < hit.end):
                    continue
                text = material(self.source[hit.documentId], hit)
                tail = self.tokenizer(text[original.end:hit.end], add_special_tokens=False,
                                      return_offsets_mapping=True, verbose=False)["offset_mapping"]
                if not tail:
                    continue
                end = original.end + tail[min(spare, len(tail)) - 1][1]
                count = len(self.tokenizer(text[original.start:end], add_special_tokens=False,
                                           verbose=False)["input_ids"])
                if count > item.tokenCount + spare:
                    continue
                extended = Context(hit=original.model_copy(update={"end": end}),
                                   tokenCount=max(count, item.tokenCount))
                updated = (*existing[:index], extended, *existing[index + 1:])
                return updated[:len(contexts)], updated[len(contexts):]
        return self.add_spare_evidence(question, contexts, supports, candidates, relevance,
                                       settings.model_copy(update={"spare_evidence": "both"}))

    def add_spare_evidence(self, question: str, contexts: tuple[Context, ...], supports: tuple[Context, ...],
                           ranked: tuple[Hit, ...], relevance: dict[Hit, float],
                           settings: RetrievalSettings) -> tuple[tuple[Context, ...], tuple[Context, ...]]:
        remaining = settings.context_budget - sum(item.tokenCount for item in (*contexts, *supports))
        if remaining < 32 or not contexts:
            return contexts, supports
        primary = {item.hit.documentId for item in contexts}
        meetings = {self.source[doc].meetingId for doc in primary}
        support_docs = {item.hit.documentId for item in supports}
        counts = Counter(self.source[doc].meetingId for doc in support_docs)
        existing = tuple(item.hit for item in (*contexts, *supports))
        eligible = []
        for hit in ranked:
            doc = hit.documentId
            meeting = self.source[doc].meetingId
            same_doc = doc in primary and settings.spare_evidence in ("chunk", "both") \
                and len(contexts) < CONTEXT_COUNT
            other_doc = doc not in primary and doc not in support_docs and meeting in meetings \
                and counts[meeting] < 2 and settings.spare_evidence in ("document", "both") \
                and len(supports) < CONTEXT_COUNT
            if hit.field == "body" and (same_doc or other_doc) and not any(
                hit.documentId == old.documentId and hit.field == old.field
                and hit.start < old.end and old.start < hit.end for old in existing
            ):
                eligible.append(hit)
        if not eligible:
            return contexts, supports
        novelty = self.uncovered_query_terms(question, existing, tuple(eligible))
        hit = max(eligible, key=lambda h: relevance[h] + 8 * novelty[h])
        added = assemble_worker_context(
            (hit,), self.source, self.tokenizer, {}, budget=min(remaining, 220), focus=question,
        )
        if hit.documentId in primary:
            return (*contexts, *added), supports
        return contexts, (*supports, *added)

    def uncovered_query_terms(self, question: str, selected: tuple[Hit, ...],
                              candidates: tuple[Hit, ...]) -> dict[Hit, float]:
        query = term_bigrams(question)

        def matches(hit: Hit) -> set[str]:
            source = self.source[hit.documentId]
            return query & term_bigrams(material(source, hit)[hit.start:hit.end])

        seen: dict[int, set[str]] = {}
        for hit in selected:
            seen.setdefault(self.source[hit.documentId].meetingId, set()).update(matches(hit))
        return {hit: len(matches(hit) - seen.get(self.source[hit.documentId].meetingId, set())) / max(1, len(query))
                for hit in candidates}

    def passages(self, contexts: tuple[Context, ...]) -> tuple[Passage, ...]:
        return tuple(Passage(**item.hit.model_dump(), tokenCount=item.tokenCount,
                             quote=material(self.source[item.hit.documentId], item.hit)[item.hit.start:item.hit.end])
                     for item in contexts)

    def supporting_documents(
        self, question: str, primary: tuple[Hit, ...], ranked: tuple[Hit, ...],
        candidates: tuple[Hit, ...], relevance: dict[Hit, float], settings: RetrievalSettings,
        *, per_meeting: int = 1,
    ) -> tuple[Hit, ...]:
        documents = {hit.documentId for hit in primary}
        meetings = {self.source[doc].meetingId for doc in documents}
        eligible = tuple(hit for hit in ranked if hit.documentId not in documents
                         and self.source[hit.documentId].meetingId in meetings)
        if not eligible:
            return ()
        scores = self.supporting_scores(question, primary, eligible, candidates, relevance, settings)
        novelty = (self.uncovered_query_terms(question, primary, eligible) if settings.complementary_bonus else {})
        chosen: dict[int, list[Hit]] = {}
        used_documents = set()
        for hit in sorted(eligible, key=lambda item: sum(scores[item]) + settings.complementary_bonus * novelty.get(item, 0),
                          reverse=True):
            if hit.documentId in used_documents:
                continue
            by_meeting = chosen.setdefault(self.source[hit.documentId].meetingId, [])
            if len(by_meeting) < per_meeting:
                by_meeting.append(hit)
                used_documents.add(hit.documentId)
        return tuple(hits[index] for index in range(per_meeting) for hits in chosen.values()
                     if index < len(hits))[:CONTEXT_COUNT]

    def supporting_scores(
        self, question: str, primary: tuple[Hit, ...], eligible: tuple[Hit, ...], candidates: tuple[Hit, ...],
        relevance: dict[Hit, float], settings: RetrievalSettings,
    ) -> dict[Hit, tuple[float, float, float, float]]:
        existing = [passage_terms(self.source[hit.documentId], hit) for hit in primary]
        lexical = (self.representative_lexical_scores(question, eligible)
                   if settings.supporting_lexical_bonus else {})
        rrf = {hit: 3 / (3 + index) for index, hit in enumerate(candidates)}
        scores = {}
        for hit in eligible:
            terms = passage_terms(self.source[hit.documentId], hit)
            overlap = max(passage_overlap(terms, terms_in_primary) for terms_in_primary in existing)
            scores[hit] = (relevance[hit], -overlap, settings.supporting_lexical_bonus * lexical.get(hit, 0.0),
                           settings.supporting_rrf_bonus * rrf[hit])
        return scores

    @staticmethod
    def by_document(ranked: tuple[Hit, ...]) -> tuple[Hit, ...]:
        by_document: dict[int, Hit] = {}
        for hit in ranked:
            by_document.setdefault(hit.documentId, hit)
        selected = tuple(by_document.values())[:CONTEXT_COUNT - 1]
        others = (hit for hit in ranked if hit not in selected)
        return (*selected, *others)[:CONTEXT_COUNT]

    def by_meeting(self, ranked: tuple[Hit, ...], recent: int) -> tuple[Hit, ...]:
        by_meeting: dict[int, Hit] = {}
        for hit in ranked:
            by_meeting.setdefault(self.source[hit.documentId].meetingId, hit)
        meetings = list(by_meeting.values())

        def recorded(hit: Hit) -> datetime:
            return datetime.fromisoformat(self.source[hit.documentId].recordedAt)

        if recent:
            meetings.sort(key=recorded, reverse=True)
            meetings = meetings[:recent]
        return tuple(meetings[:CONTEXT_COUNT])

    def choose_representatives(
        self, question: str, representatives: tuple[Hit, ...], candidates: tuple[Hit, ...],
        relevance: dict[Hit, float], settings: RetrievalSettings,
    ) -> tuple[Hit, ...]:
        options: dict[int, list[Hit]] = {}
        for hit in candidates:
            options.setdefault(self.source[hit.documentId].meetingId, []).append(hit)
        lexical = (self.representative_lexical_scores(question, candidates)
                   if settings.representative_lexical_bonus else {})
        rrf_weight = {hit: 3 / (3 + index) for index, hit in enumerate(candidates)}

        def priority(hit: Hit) -> float:
            return (relevance[hit] + settings.representative_body_bonus * (hit.field == "body")
                    + settings.representative_lexical_bonus * lexical.get(hit, 0.0)
                    + settings.representative_rrf_bonus * rrf_weight[hit])

        return tuple(max((representative, *options[self.source[representative.documentId].meetingId]),
                         key=priority) for representative in representatives)

    def representative_lexical_scores(self, question: str, ranked: tuple[Hit, ...]) -> dict[Hit, float]:
        query = term_bigrams(question)
        if not query:
            return {}
        matches = {}
        for hit in ranked:
            doc = self.source[hit.documentId]
            window = adjacent_paragraphs(doc, hit)
            title = query & term_bigrams(doc.title)
            excerpt = query & term_bigrams(material(doc, window)[window.start:window.end])
            matches[hit] = title, excerpt
        weights = {term: 1 / (1 + sum(term in title or term in excerpt for title, excerpt in matches.values()))
                   for term in query}
        total = sum(weights.values())
        return {hit: (sum(weights[term] for term in title) + sum(weights[term] for term in excerpt)) / total
                for hit, (title, excerpt) in matches.items()}

    def supplement(self, representatives: tuple[Hit, ...], ranked: tuple[Hit, ...],
                   mode: Literal["document", "chunk", "both"], relevance: dict[Hit, float], *,
                   overlap_penalty: float = 0.0, reuse_spare_slots: bool = False) -> tuple[Hit, ...]:
        if len(representatives) >= CONTEXT_COUNT:
            return representatives
        by_meeting = {self.source[hit.documentId].meetingId: hit for hit in representatives}
        eligible = []
        for hit in ranked:
            meeting = self.source[hit.documentId].meetingId
            representative = by_meeting.get(meeting)
            if representative is None:
                continue
            other_document = hit.documentId != representative.documentId
            distinct_chunk = (hit.field == representative.field
                              and (hit.end <= representative.start or representative.end <= hit.start))
            if ((other_document and mode in ("document", "both"))
                    or (not other_document and distinct_chunk and mode in ("chunk", "both"))):
                eligible.append((meeting, hit))
        selected = list(representatives)
        windows = {}
        if overlap_penalty and eligible:
            windows = {hit: passage_terms(self.source[hit.documentId], hit)
                       for hit in (*representatives, *(hit for _, hit in eligible))}

        supplemented_meetings = set()
        while eligible and len(selected) < CONTEXT_COUNT:
            choices = [item for item in eligible if item[0] not in supplemented_meetings]
            if not choices and not reuse_spare_slots:
                break
            choices = choices or eligible
            meeting, hit = choices[0]
            if windows:
                meeting, hit = max(choices, key=lambda item: relevance[item[1]] - overlap_penalty * max(
                    passage_overlap(windows[item[1]], windows[chosen]) for chosen in selected
                ))
            selected.append(hit)
            supplemented_meetings.add(meeting)
            eligible = [(other_meeting, other_hit) for other_meeting, other_hit in eligible
                        if not (
                            other_hit.documentId == hit.documentId and other_hit.field == hit.field
                            and other_hit.start < hit.end and hit.start < other_hit.end
                        )]
        return tuple(selected)


def pack_meeting_evidence(question: str, result: RankResponse, retriever: Retriever,
                          settings: RetrievalSettings) -> RankResponse:
    """Reallocate the synthesis budget without changing the document or meeting selection."""
    started = perf_counter()
    groups = (result.passages, result.supportingPassages, result.discoveryPassages)
    flat = tuple(
        Context(hit=Hit.model_validate(passage.model_dump()), tokenCount=passage.tokenCount)
        for group in groups for passage in group
    )
    short = tuple(
        Context(hit=Hit.model_validate(passage.model_dump()), tokenCount=passage.tokenCount)
        for passage in result.shortPassages
    )
    packed = reserve_meeting_evidence(
        question, flat, short, retriever.source, retriever.tokenizer,
        budget=settings.expansion_budget, reclaim=settings.meeting_evidence_reclaim,
        require_topic_status=settings.meeting_evidence_topic_status,
    )
    primary_end = len(result.passages)
    supporting_end = primary_end + len(result.supportingPassages)
    return result.model_copy(update={
        "passages": retriever.passages(packed.passages[:primary_end]),
        "supportingPassages": retriever.passages(packed.passages[primary_end:supporting_end]),
        "discoveryPassages": retriever.passages(packed.passages[supporting_end:]),
        "shortPassages": retriever.passages(packed.short),
        "contextMs": result.contextMs + ms(started),
    })


def term_bigrams(text: str) -> set[str]:
    return {term[index:index + 2] for term in tokens(text) for index in range(len(term) - 1)}


def passage_terms(document: SourceDoc, hit: Hit) -> set[tuple[str, ...]]:
    window = adjacent_paragraphs(document, hit)
    terms = tokens(material(document, window)[window.start:window.end])
    if len(terms) < 3:
        return {tuple(terms)} if terms else set()
    return set(zip(terms, terms[1:], terms[2:]))


def passage_overlap(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    return len(left & right) / max(1, min(len(left), len(right)))


def prefer_source_text(selected: tuple[Hit, ...], ranked: tuple[Hit, ...],
                       scores: dict[Hit, float]) -> tuple[Hit, ...]:
    body_by_document: dict[int, Hit] = {}
    for hit in ranked:
        if hit.field == "body":
            body_by_document.setdefault(hit.documentId, hit)
    passages = []
    used = set(selected)
    for hit in selected:
        body = body_by_document.get(hit.documentId)
        if hit.field == "summary" and body is not None and body not in used \
                and scores[body] + BODY_SCORE_MARGIN >= scores[hit]:
            used.add(body)
            passages.append(body)
        else:
            passages.append(hit)
    return tuple(passages)


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(4)
    app.state.worker_settings = SPARE_SLOT_SETTINGS
    if os.environ.get("KNOT_HYBRID_EVIDENCE_EXPANSION") == "true":
        minimum_budget = EVIDENCE_EXPANSION_SETTINGS.expansion_budget
        budget = int(os.environ.get("KNOT_SEARCH_HYBRID_EVIDENCE_EXPANSION_BUDGET", minimum_budget))
        if not minimum_budget <= budget <= MAX_EXPANSION_BUDGET:
            raise ValueError(f"Hybrid evidence expansion budget must be between {minimum_budget} and {MAX_EXPANSION_BUDGET}")
        app.state.worker_settings = RetrievalSettings.model_validate({
            **EVIDENCE_EXPANSION_SETTINGS.model_dump(), "expansion_budget": budget,
            "same_document_tokens": int(os.environ.get("KNOT_HYBRID_SAME_DOCUMENT_EVIDENCE_TOKENS", "0")),
            "short_passage_reserve": int(os.environ.get("KNOT_HYBRID_SHORT_PASSAGE_RESERVE", "0")),
            "short_passages_spare_only": os.environ.get("KNOT_HYBRID_SHORT_PASSAGES_SPARE_ONLY") == "true",
            "rerank_supplemental": os.environ.get("KNOT_HYBRID_RERANK_SUPPLEMENTAL") == "true",
            "discovery_depth": int(os.environ.get("KNOT_HYBRID_DISCOVERY_DEPTH", "0")),
            "discovery_rerank_limit": int(os.environ.get("KNOT_HYBRID_DISCOVERY_RERANK_LIMIT", "48")),
            "discovery_compact": os.environ.get("KNOT_HYBRID_DISCOVERY_COMPACT", "true") == "true",
            "discovery_window_rerank": os.environ.get("KNOT_HYBRID_DISCOVERY_WINDOW_RERANK") == "true",
            "discovery_uncovered_only": os.environ.get("KNOT_HYBRID_DISCOVERY_UNCOVERED_ONLY") == "true",
            "discovery_excerpt_tokens": int(os.environ.get("KNOT_HYBRID_DISCOVERY_EXCERPT_TOKENS", "96")),
            "discovery_existing_first": os.environ.get("KNOT_HYBRID_DISCOVERY_EXISTING_FIRST") == "true",
            "meeting_evidence_first": os.environ.get("KNOT_HYBRID_MEETING_EVIDENCE_FIRST") == "true",
            "meeting_evidence_topic_status": os.environ.get("KNOT_HYBRID_MEETING_EVIDENCE_TOPIC_STATUS") == "true",
            "meeting_evidence_reclaim": int(os.environ.get("KNOT_HYBRID_MEETING_EVIDENCE_RECLAIM", "0")),
        })
    elif os.environ.get("KNOT_HYBRID_EVIDENCE_SELECTION") == "true":
        app.state.worker_settings = EVIDENCE_SELECTION_SETTINGS
    elif os.environ.get("KNOT_HYBRID_PROTECTED_SUPPORT") == "true":
        app.state.worker_settings = PROTECTED_SUPPORT_SETTINGS
    elif os.environ.get("KNOT_HYBRID_SUPPORTING_SELECTION") == "true":
        app.state.worker_settings = SUPPORTING_SELECTION_SETTINGS
    elif os.environ.get("KNOT_HYBRID_SUPPORTING_CONTEXT") == "true":
        app.state.worker_settings = SUPPORTING_DOCUMENT_SETTINGS
    elif os.environ.get("KNOT_HYBRID_EXPAND_CONTEXT") == "true":
        app.state.worker_settings = EXPANDED_REPRESENTATIVE_SETTINGS
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    dsn = os.environ.get("KNOT_HYBRID_LIVE_DSN")
    live = None
    if dsn:
        workspace_ids = tuple(int(value) for value in os.environ.get("KNOT_HYBRID_LIVE_WORKSPACES", "").split(",") if value)
        if not workspace_ids or any(value <= 0 for value in workspace_ids):
            raise RuntimeError("Set KNOT_HYBRID_LIVE_WORKSPACES for the live index.")
        live = LiveIndex(dsn, workspace_ids)
        with live.connect() as db:
            db.execute("SELECT 1 FROM hybrid_document_index_jobs LIMIT 1")
        app.state.retrievers = {}
        app.state.live_cache = {}
        app.state.live_cache_lock = Lock()
    else:
        app.state.retrievers = {workspace: Retriever(model, workspace) for workspace in (1, 458)}
    app.state.live_index = live
    model.encode(["query: 워밍업"], normalize_embeddings=True, show_progress_bar=False)
    app.state.model = model
    app.state.embed_lock = Lock()
    app.state.reranker = CrossEncoder(RERANKER, revision=RERANKER_REVISION, max_length=384,
                                      device="cpu", trust_remote_code=False)
    app.state.reranker.predict([("워밍업", "검색 예시")], show_progress_bar=False)
    app.state.rerank_lock = Lock()
    trace_path = os.environ.get("KNOT_ACCURACY_TRACE")
    if trace_path:
        app.state.accuracy_trace = Path(trace_path)
        app.state.accuracy_trace.touch(mode=0o600, exist_ok=False)
        app.state.accuracy_trace_lock = Lock()
    else:
        app.state.accuracy_trace = None
    stop = Event()
    worker = None
    if live:
        worker = Thread(target=live.run, args=(stop, model, app.state.embed_lock, model.tokenizer), daemon=True)
        worker.start()
    try:
        yield
    finally:
        stop.set()
        if worker:
            worker.join(timeout=5)


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.post("/rank", response_model=RankResponse)
def rank(data: RankRequest, request: Request) -> RankResponse:
    trace = request.app.state.accuracy_trace
    if trace is not None:
        row = data.model_copy(update={"question": sha256(data.question.encode()).hexdigest()})
        with request.app.state.accuracy_trace_lock, trace.open("a") as output:
            output.write(row.model_dump_json() + "\n")
    if not data.allowedDocumentIds:
        return RankResponse(documentIds=())
    with request.app.state.embed_lock:
        query = request.app.state.model.encode([f"query: {data.question}"], normalize_embeddings=True,
                                               show_progress_bar=False)[0]
    embedding = vector_literal(query.tolist())
    state = request.app.state
    live = state.live_index
    if live is None:
        retriever = state.retrievers.get(data.workspaceId)
        return rank_with(data, state, retriever, embedding) if retriever else RankResponse(documentIds=())
    if data.workspaceId not in live.workspaces:
        return RankResponse(documentIds=())
    # One snapshot covers the version check, BM25 source, and vector reads.
    with live.connect() as db:
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        versions = source_versions(db, data.workspaceId)
        if not versions:
            return RankResponse(documentIds=())
        with state.live_cache_lock:
            cached = state.live_cache.get(data.workspaceId)
            if cached is None or cached[0] != versions:
                with state.embed_lock:
                    retriever = Retriever(state.model, data.workspaceId, live_source=live_documents(db, data.workspaceId))
                state.live_cache[data.workspaceId] = (versions, retriever)
            else:
                retriever = cached[1]
        return rank_with(data, state, retriever, embedding, db=db)


def rank_with(data: RankRequest, state: State, retriever: Retriever, embedding: str, *,
              db: Connection | None = None) -> RankResponse:
    settings = BASELINE if data.contextMode == "none" else state.worker_settings
    if settings.discovery_depth and data.contextMode == "synthesis":
        allowed = (data.preferredDocumentId,) if data.preferredDocumentId in data.allowedDocumentIds else data.allowedDocumentIds
        wide = settings.model_copy(update={"keyword_depth": max(settings.discovery_depth, settings.keyword_depth),
                                           "vector_depth": max(settings.discovery_depth, settings.vector_depth)})
        keyword, vector = retriever.rank_sources(data.question, embedding, allowed, wide, db=db)
        candidates = retriever.fuse_candidates(keyword[:settings.keyword_depth], vector[:settings.vector_depth], settings)
        baseline = retriever.with_context(data, candidates, state.reranker, state.rerank_lock, state.embed_lock, settings)
        result = retriever.discover(data.question, keyword, vector, candidates, baseline, state.reranker,
                                    state.rerank_lock, state.embed_lock, settings)
        if settings.meeting_evidence_first:
            with state.embed_lock:
                return pack_meeting_evidence(data.question, result, retriever, settings)
        return result
    candidates = retriever.rank(data.question, embedding, data.allowedDocumentIds, settings,
                                data.preferredDocumentId, db=db)
    return retriever.with_context(data, candidates, state.reranker, state.rerank_lock, state.embed_lock, settings)
