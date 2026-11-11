"""Loopback-only scoped retrieval and evidence selection for the Spring Worker."""
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from threading import Lock
from typing import Literal

import torch
from fastapi import FastAPI, Request
from pydantic import Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from starlette.middleware.trustedhost import TrustedHostMiddleware
from transformers import PreTrainedTokenizerFast

from build_dataset import Frozen, documents
from database import connect, vector_literal
from experiment import MODEL, REVISION, tokens
from phase1 import Hit, chunks, prepare
from phase2 import DEPTH, fuse, ms
from phase3 import CONTEXT_COUNT, RERANKER, RERANKER_REVISION, assemble_worker_context, material

BODY_SCORE_MARGIN = 1.0
WEAK_SCORE_GAP = 6.25
WEAK_PASSAGE_TOKENS = 96


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

    def rank(self, question: str, embedding: str, allowed_ids: tuple[int, ...]) -> tuple[Hit, ...]:
        allowed = set(allowed_ids)
        with connect() as db:
            vector = tuple(Hit(documentId=doc, field=field, start=left, end=right)
                           for doc, field, left, right in db.execute(
                """select document_id,field,start_offset,end_offset
                   from workspace1_chunks where workspace_id=%s and mode='fixed' and document_id=any(%s)
                   order by embedding <=> %s::vector,document_id,field,start_offset limit %s""",
                (self.workspace_id, list(allowed), embedding, DEPTH),
            ))
        scores = self.keyword.get_scores(tokens(question))
        matching = (i for i, hit in enumerate(self.hits) if hit.documentId in allowed and scores[i] > 0)
        keyword = tuple(self.hits[i] for i in sorted(matching, key=lambda i: (-scores[i], i))[:DEPTH])
        return fuse(keyword, vector)

    def with_context(self, request: RankRequest, candidates: tuple[Hit, ...],
                     reranker: CrossEncoder, rerank_lock: Lock, tokenizer_lock: Lock) -> RankResponse:
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
        best = relevance[ranked[0]]
        limits = {hit: WEAK_PASSAGE_TOKENS for hit in selected
                  if best >= 0 and best - relevance[hit] > WEAK_SCORE_GAP}
        with tokenizer_lock:
            contexts = assemble_worker_context(selected, self.source, self.tokenizer, limits)
        passages = tuple(Passage(**item.hit.model_dump(), tokenCount=item.tokenCount,
                                 quote=material(self.source[item.hit.documentId], item.hit)[item.hit.start:item.hit.end])
                         for item in contexts)
        return RankResponse(documentIds=document_ids, passages=passages,
                            rerankMs=rerank_ms, contextMs=ms(started))

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
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    app.state.retrievers = {workspace: Retriever(model, workspace) for workspace in (1, 458)}
    model.encode(["query: 워밍업"], normalize_embeddings=True, show_progress_bar=False)
    app.state.model = model
    app.state.embed_lock = Lock()
    app.state.reranker = CrossEncoder(RERANKER, revision=RERANKER_REVISION, max_length=384,
                                      device="cpu", trust_remote_code=False)
    app.state.reranker.predict([("워밍업", "검색 예시")], show_progress_bar=False)
    app.state.rerank_lock = Lock()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.post("/rank", response_model=RankResponse)
def rank(data: RankRequest, request: Request) -> RankResponse:
    if not data.allowedDocumentIds:
        return RankResponse(documentIds=())
    with request.app.state.embed_lock:
        query = request.app.state.model.encode([f"query: {data.question}"], normalize_embeddings=True,
                                               show_progress_bar=False)[0]
    retriever = request.app.state.retrievers[data.workspaceId]
    candidates = retriever.rank(data.question, vector_literal(query.tolist()), data.allowedDocumentIds)
    return retriever.with_context(data, candidates, request.app.state.reranker, request.app.state.rerank_lock,
                                  request.app.state.embed_lock)
