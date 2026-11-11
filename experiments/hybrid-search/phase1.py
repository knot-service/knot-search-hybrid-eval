"""Compare exact vector retrieval for document, fixed-window, and Markdown child indexes."""
from datetime import datetime
from hashlib import sha256
import json
from math import log2
import re
from statistics import mean
import sys
from time import perf_counter
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from psycopg import Connection

from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast
import torch

from build_dataset import Evidence, Frozen, History, SourceDoc, documents
from database import LOCAL, connect, vector_literal
from experiment import MODEL, REVISION

IndexMode = Literal["document", "fixed", "sections"]
SourceField = Literal["body", "summary"]
MODES: tuple[IndexMode, ...] = ("document", "fixed", "sections")
CHILD_TOKENS = 320
OVERLAP_TOKENS = 48
HEADING = re.compile(r"(?m)^#{1,6}[ \t]+\S.*$")
SCHEMA = """
create table if not exists workspace1_chunks (
    workspace_id integer not null,
    mode text not null,
    document_id integer not null,
    field text not null,
    start_offset integer not null,
    end_offset integer not null,
    embedding vector(384) not null,
    primary key(workspace_id,mode,document_id,field,start_offset),
    foreign key(workspace_id,document_id) references documents(workspace_id,id) on delete cascade
);
"""


class Child(Frozen):
    documentId: int
    field: SourceField
    start: int
    end: int
    text: str


class EvidenceSpan(Evidence):
    startOffset: int
    endOffset: int


class DraftCase(Frozen):
    id: str
    variantLabel: str = "기본"
    question: str
    expectedType: str
    expectedEvidenceIds: tuple[int, ...]
    evidence: tuple[EvidenceSpan, ...]
    history: tuple[History, ...]


class Draft(Frozen):
    sourceHash: str
    version: str
    cases: tuple[DraftCase, ...]


class Hit(Frozen):
    documentId: int
    field: SourceField
    start: int
    end: int


class Result(Frozen):
    caseId: str
    variantLabel: str
    mode: IndexMode
    documentRecallAt10: float
    ndcgAt5: float
    evidenceSpanRecallAt10: float | None
    embeddingMs: float
    searchMs: float
    rankedDocumentIds: tuple[int, ...]
    rankedChunks: tuple[Hit, ...]


def spans(text: str, tokenizer: PreTrainedTokenizerFast, *, sections: bool) -> tuple[tuple[int, int], ...]:
    boundaries = {0, len(text)}
    if sections:
        boundaries.update(match.start() for match in HEADING.finditer(text))
    points = sorted(boundaries)
    result: list[tuple[int, int]] = []
    for start, end in zip(points, points[1:]):
        offsets: list[tuple[int, int]] = tokenizer(
            text[start:end], add_special_tokens=False, return_offsets_mapping=True, verbose=False,
        )["offset_mapping"]
        cursor = 0
        while cursor < len(offsets):
            window = offsets[cursor:cursor + CHILD_TOKENS]
            result.append((start + window[0][0], start + window[-1][1]))
            if cursor + CHILD_TOKENS >= len(offsets):
                break
            cursor += CHILD_TOKENS - OVERLAP_TOKENS
    return tuple(result)


def chunks(doc: SourceDoc, mode: IndexMode, tokenizer: PreTrainedTokenizerFast) -> tuple[Child, ...]:
    if mode == "document":
        return (Child(documentId=doc.id, field="summary", start=0, end=len(doc.summary),
                      text=f"{doc.title}\n{doc.topic}\n{doc.summary or doc.body[:750]}"[:1200]),)
    children = [Child(documentId=doc.id, field="body", start=start, end=end,
                      text=f"{doc.title}\n{doc.body[start:end]}")
                for start, end in spans(doc.body, tokenizer, sections=mode == "sections")]
    if doc.summary:
        children.append(Child(documentId=doc.id, field="summary", start=0, end=len(doc.summary),
                              text=f"{doc.title}\n{doc.summary}"))
    return tuple(children)


def prepare(
    db: Connection, docs: tuple[SourceDoc, ...], model: SentenceTransformer,
    *, workspace_id: int = 1, modes: tuple[IndexMode, ...] = MODES,
) -> dict[str, int]:
    tokenizer = model.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise SystemExit("The embedding tokenizer must return character offsets.")
    by_mode = {mode: tuple(child for doc in docs for child in chunks(doc, mode, tokenizer)) for mode in modes}
    with db.transaction(), db.cursor() as cursor:
        cursor.execute(SCHEMA)
        cursor.execute("delete from workspace1_chunks where workspace_id=%s and mode=any(%s)",
                       (workspace_id, list(modes)))
        for mode, children in by_mode.items():
            vectors = model.encode(
                [f"passage: {child.text}" for child in children], batch_size=32,
                normalize_embeddings=True, show_progress_bar=False,
            )
            cursor.executemany(
                "insert into workspace1_chunks values (%s,%s,%s,%s,%s,%s,%s::vector)",
                [(workspace_id, mode, child.documentId, child.field, child.start, child.end, vector_literal(vector.tolist()))
                 for child, vector in zip(children, vectors, strict=True)],
            )
    return {mode: len(children) for mode, children in by_mode.items()}

def rank_quality(case: DraftCase, ranked: tuple[int, ...], hits: tuple[Hit, ...]) -> tuple[float, float, float]:
    expected = set(case.expectedEvidenceIds)
    dcg = sum(1 / log2(rank + 2) for rank, doc_id in enumerate(ranked[:5]) if doc_id in expected)
    ideal = sum(1 / log2(rank + 2) for rank in range(min(len(expected), 5)))
    matched = [any(
        hit.documentId == evidence.documentId and hit.field == evidence.field
        and hit.start <= evidence.startOffset and hit.end >= evidence.endOffset
        for hit in hits
    ) for evidence in case.evidence]
    return len(expected.intersection(ranked[:10])) / len(expected), dcg / ideal, mean(matched)



def score(
    db: Connection, mode: IndexMode, case: DraftCase, vector: str, embedding_ms: float,
) -> Result:
    start = perf_counter()
    records = db.execute(
        """select document_id,field,start_offset,end_offset
           from workspace1_chunks where workspace_id=1 and mode=%s
           order by embedding <=> %s::vector,document_id,field,start_offset""", (mode, vector),
    ).fetchall()
    ranked = tuple(dict.fromkeys(row[0] for row in records))[:10]
    hits = tuple(Hit(documentId=doc_id, field=field, start=left, end=right)
                 for doc_id, field, left, right in records[:10])
    search_ms = (perf_counter() - start) * 1000
    recall, ndcg, span = rank_quality(case, ranked, hits)
    return Result(
        caseId=case.id, variantLabel=case.variantLabel, mode=mode, documentRecallAt10=recall,
        ndcgAt5=ndcg, evidenceSpanRecallAt10=span if mode != "document" else None,
        embeddingMs=embedding_ms, searchMs=search_ms, rankedDocumentIds=ranked, rankedChunks=hits,
    )


def summarize(rows: list[Result]) -> dict:
    metrics = {}
    for mode in MODES:
        runs = [row for row in rows if row.mode == mode]
        span_scores = [row.evidenceSpanRecallAt10 for row in runs if row.evidenceSpanRecallAt10 is not None]
        metrics[mode] = {
            "documentRecallAt10": mean(row.documentRecallAt10 for row in runs),
            "ndcgAt5": mean(row.ndcgAt5 for row in runs),
            "evidenceSpanRecallAt10": mean(span_scores) if span_scores else None,
            "queryEmbeddingMs": dict(zip(("p50", "p95"), np.percentile([r.embeddingMs for r in runs], (50, 95)).tolist(), strict=True)),
            "searchMs": dict(zip(("p50", "p95"), np.percentile([r.searchMs for r in runs], (50, 95)).tolist(), strict=True)),
        }
    return metrics


def main() -> None:
    docs = documents()
    if not docs:
        raise SystemExit("Import Workspace 1 before preparing the phase 1 index.")
    source_hash = sha256(json.dumps([doc.model_dump() for doc in docs], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    with connect() as db:
        index_counts = prepare(db, docs, model)
        print(f"indexed workspace=1 source={source_hash} chunks={index_counts}", flush=True)
        if "--index-only" in sys.argv:
            return
        draft = Draft.model_validate_json((LOCAL / "workspace1-evaluation-draft.json").read_bytes())
        if draft.sourceHash != source_hash:
            raise SystemExit("The draft and indexed documents come from different source snapshots.")
        eligible = [case for case in draft.cases
                    if case.expectedType == "answer" and not case.history and case.expectedEvidenceIds]
        if not eligible:
            raise SystemExit("No standalone answer cases are eligible for retrieval scoring.")
        model.encode(["query: 워밍업"], normalize_embeddings=True, show_progress_bar=False)
        rows: list[Result] = []
        for case in eligible:
            start = perf_counter()
            embedding = model.encode([f"query: {case.question}"], normalize_embeddings=True,
                                     show_progress_bar=False)[0]
            embedding_ms = (perf_counter() - start) * 1000
            vector = vector_literal(embedding.tolist())
            rows.extend(score(db, mode, case, vector, embedding_ms) for mode in MODES)
    metrics = summarize(rows)
    variant_counts = {label: sum(case.variantLabel == label for case in eligible)
                      for label in dict.fromkeys(case.variantLabel for case in eligible)}
    variant_metrics = {label: summarize([row for row in rows if row.variantLabel == label])
                       for label in variant_counts}
    report = {"status":"observed_on_ai_draft","sourceHash":source_hash,"datasetVersion":draft.version,
              "createdAt":datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "caseCount":len(eligible),
              "generationCalls":0, "indexCounts":index_counts, "metrics":metrics,
              "variantCounts":variant_counts,"variantMetrics":variant_metrics,
              "results":[row.model_dump() for row in rows]}
    path = LOCAL / "workspace1-phase1.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)
    print(json.dumps({"cases":len(eligible),"metrics":metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
