"""Loopback-only retrieval adapter for the Spring Boot phase 4 experiment."""
from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

import torch
from fastapi import FastAPI, Request
from pydantic import Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from starlette.middleware.trustedhost import TrustedHostMiddleware
from transformers import PreTrainedTokenizerFast

from build_dataset import Frozen, documents
from database import connect, vector_literal
from experiment import MODEL, REVISION, tokens
from phase1 import Hit, chunks, prepare
from phase2 import DEPTH, fuse


class RankRequest(Frozen):
    workspaceId: Literal[1, 458]
    question: str = Field(min_length=1, max_length=4000)
    allowedDocumentIds: tuple[int, ...] = Field(max_length=10_000)


class RankResponse(Frozen):
    documentIds: tuple[int, ...]


class Retriever:
    def __init__(self, model: SentenceTransformer, workspace_id: int):
        source = documents(workspace_id)
        if not source:
            raise RuntimeError(f"No snapshot for workspace {workspace_id}")
        tokenizer = model.tokenizer
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise RuntimeError("The embedding tokenizer must return character offsets.")
        units = tuple(child for doc in source for child in chunks(doc, "fixed", tokenizer))
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

    def rank(self, question: str, embedding: str, allowed_ids: tuple[int, ...]) -> RankResponse:
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
        return RankResponse(documentIds=tuple(dict.fromkeys(hit.documentId for hit in fuse(keyword, vector))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    app.state.retrievers = {workspace: Retriever(model, workspace) for workspace in (1, 458)}
    model.encode(["query: 워밍업"], normalize_embeddings=True, show_progress_bar=False)
    app.state.model = model
    app.state.embed_lock = Lock()
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
    return retriever.rank(data.question, vector_literal(query.tolist()), data.allowedDocumentIds)
