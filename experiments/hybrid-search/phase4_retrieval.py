"""Loopback-only scoped retrieval and evidence selection for the Spring Worker."""
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter
from threading import Lock
from typing import Callable, Literal

import torch
from fastapi import FastAPI, Request
from pydantic import Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from starlette.middleware.trustedhost import TrustedHostMiddleware
from transformers import PreTrainedTokenizerFast

from build_dataset import Frozen, SourceDoc, documents
from database import connect, vector_literal
from experiment import MODEL, REVISION, tokens
from phase1 import Hit, chunks, prepare
from phase2 import DEPTH, fuse, ms
from phase3 import (
    CONTEXT_BUDGET, CONTEXT_COUNT, RERANKER, RERANKER_REVISION, Context,
    adjacent_paragraphs, assemble_worker_context, material,
)

BODY_SCORE_MARGIN = 1.0
WEAK_SCORE_GAP = 6.25
WEAK_PASSAGE_TOKENS = 96


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
    supporting_in_questions: bool = False
    keyword_depth: int = DEPTH
    vector_depth: int = DEPTH
    fusion_depth: int | None = DEPTH
    fusion_reserve: int = 0
    context_budget: int = CONTEXT_BUDGET
    weak_passage_tokens: int | None = WEAK_PASSAGE_TOKENS
    sentence_focus: bool = False


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


class RankRequest(Frozen):
    workspaceId: Literal[1, 458]
    question: str = Field(min_length=1, max_length=4000)
    allowedDocumentIds: tuple[int, ...] = Field(max_length=10_000)
    contextMode: Literal["none", "question", "synthesis"] = "none"
    recentMeetings: int = Field(default=0, ge=0, le=10)


class Passage(Hit):
    quote: str
    tokenCount: int


class RankResponse(Frozen):
    documentIds: tuple[int, ...]
    passages: tuple[Passage, ...] = ()
    supportingPassages: tuple[Passage, ...] = ()
    rerankMs: float = 0
    contextMs: float = 0


class Retriever:
    def __init__(self, model: SentenceTransformer, workspace_id: int):
        source = documents(workspace_id)
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
        with connect() as db:
            existing = {Hit(documentId=doc, field=field, start=left, end=right)
                        for doc, field, left, right in db.execute(
                            """select document_id,field,start_offset,end_offset from workspace1_chunks
                               where workspace_id=%s and mode='fixed'""", (workspace_id,),
                        )}
            if existing != set(self.hits):
                prepare(db, source, model, workspace_id=workspace_id, modes=("fixed",))

    def rank(self, question: str, embedding: str, allowed_ids: tuple[int, ...],
             settings: RetrievalSettings = BASELINE) -> tuple[Hit, ...]:
        keyword, vector = self.rank_sources(question, embedding, allowed_ids, settings)
        return fuse(keyword, vector, settings.fusion_depth, settings.fusion_reserve)

    def rank_sources(self, question: str, embedding: str, allowed_ids: tuple[int, ...],
                     settings: RetrievalSettings) -> tuple[tuple[Hit, ...], tuple[Hit, ...]]:
        allowed = set(allowed_ids)
        with connect() as db:
            vector = tuple(Hit(documentId=doc, field=field, start=left, end=right)
                           for doc, field, left, right in db.execute(
                """select document_id,field,start_offset,end_offset
                   from workspace1_chunks where workspace_id=%s and mode='fixed' and document_id=any(%s)
                   order by embedding <=> %s::vector,document_id,field,start_offset limit %s""",
                (self.workspace_id, list(allowed), embedding, settings.vector_depth),
            ))
        scores = self.keyword.get_scores(tokens(question))
        matching = (i for i, hit in enumerate(self.hits) if hit.documentId in allowed and scores[i] > 0)
        keyword = tuple(self.hits[i] for i in sorted(matching, key=lambda i: (-scores[i], i))[:settings.keyword_depth])
        return keyword, vector

    def with_context(self, request: RankRequest, candidates: tuple[Hit, ...],
                     reranker: CrossEncoder, rerank_lock: Lock, tokenizer_lock: Lock,
                     settings: RetrievalSettings = BASELINE, *,
                     on_selected: Callable[[tuple[Hit, ...]], None] | None = None) -> RankResponse:
        document_ids = tuple(dict.fromkeys(hit.documentId for hit in candidates))
        if request.contextMode == "none" or not candidates:
            return RankResponse(documentIds=document_ids)

        pairs = [(request.question, f"{self.source[hit.documentId].title}\n"
                  f"{material(self.source[hit.documentId], hit)[hit.start:hit.end]}") for hit in candidates]
        started = perf_counter()
        with rerank_lock:
            scores = reranker.predict(pairs, batch_size=10, show_progress_bar=False)
        relevance = dict(zip(candidates, map(float, scores), strict=True))
        ranked = tuple(sorted(candidates, key=lambda hit: -relevance[hit]))
        rerank_ms = ms(started)
        started = perf_counter()
        if request.contextMode == "synthesis":
            selected = self.by_meeting(ranked, request.recentMeetings)
        else:
            selected = self.by_document(ranked)
        selected = prefer_source_text(selected, ranked, relevance)
        if request.contextMode == "synthesis" and (
            not settings.representatives_only_when_full or len(selected) == CONTEXT_COUNT
        ) and (settings.representative_body_bonus or settings.representative_lexical_bonus
               or settings.representative_rrf_bonus):
            selected = self.choose_representatives(request.question, selected, candidates, relevance, settings)
        if request.contextMode == "synthesis" and settings.supplement != "none":
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
                if support_budget:
                    primary_hits = tuple(item.hit for item in contexts)
                    extras = self.supporting_documents(request.question, primary_hits, ranked, relevance, settings)
                    supports = assemble_worker_context(
                        extras, self.source, self.tokenizer,
                        dict.fromkeys(extras, settings.supporting_excerpt_tokens),
                        budget=support_budget, focus=request.question,
                    )
                    extra_tokens = sum(item.tokenCount for item in supports)
                    if extra_tokens > remaining:
                        contexts = assemble_worker_context(
                            selected, self.source, self.tokenizer, limits,
                            budget=settings.context_budget - extra_tokens, focus=focus, extents=extents,
                        )
        return RankResponse(documentIds=document_ids, passages=self.passages(contexts),
                            supportingPassages=self.passages(supports),
                            rerankMs=rerank_ms, contextMs=ms(started))

    def passages(self, contexts: tuple[Context, ...]) -> tuple[Passage, ...]:
        return tuple(Passage(**item.hit.model_dump(), tokenCount=item.tokenCount,
                             quote=material(self.source[item.hit.documentId], item.hit)[item.hit.start:item.hit.end])
                     for item in contexts)

    def supporting_documents(
        self, question: str, primary: tuple[Hit, ...], ranked: tuple[Hit, ...],
        relevance: dict[Hit, float], settings: RetrievalSettings,
    ) -> tuple[Hit, ...]:
        documents = {hit.documentId for hit in primary}
        meetings = {self.source[doc].meetingId for doc in documents}
        eligible = tuple(hit for hit in ranked if hit.documentId not in documents
                         and self.source[hit.documentId].meetingId in meetings)
        if not eligible:
            return ()
        existing = [passage_terms(self.source[hit.documentId], hit) for hit in primary]
        lexical = (self.representative_lexical_scores(question, eligible)
                   if settings.supporting_lexical_bonus else {})
        scores = {}
        for hit in eligible:
            terms = passage_terms(self.source[hit.documentId], hit)
            overlap = max(passage_overlap(terms, terms_in_primary) for terms_in_primary in existing)
            scores[hit] = relevance[hit] - overlap + settings.supporting_lexical_bonus * lexical.get(hit, 0.0)
        chosen = {}
        for hit in sorted(eligible, key=scores.__getitem__, reverse=True):
            chosen.setdefault(self.source[hit.documentId].meetingId, hit)
        return tuple(chosen.values())

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
    if os.environ.get("KNOT_HYBRID_SUPPORTING_CONTEXT") == "true":
        app.state.worker_settings = SUPPORTING_DOCUMENT_SETTINGS
    elif os.environ.get("KNOT_HYBRID_EXPAND_CONTEXT") == "true":
        app.state.worker_settings = EXPANDED_REPRESENTATIVE_SETTINGS
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    app.state.retrievers = {workspace: Retriever(model, workspace) for workspace in (1, 458)}
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
    yield


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
    retriever = request.app.state.retrievers[data.workspaceId]
    settings = BASELINE if data.contextMode == "none" else request.app.state.worker_settings
    candidates = retriever.rank(data.question, vector_literal(query.tolist()), data.allowedDocumentIds, settings)
    return retriever.with_context(data, candidates, request.app.state.reranker, request.app.state.rerank_lock,
                                  request.app.state.embed_lock, settings)
