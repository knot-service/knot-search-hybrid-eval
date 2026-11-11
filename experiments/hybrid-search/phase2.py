"""Measure lexical, vector, and RRF ranking on the same fixed-length Child corpus."""
from datetime import datetime
from hashlib import sha256
import json
from statistics import mean
from time import perf_counter
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
from psycopg import Connection
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast
import torch

from build_dataset import Frozen, documents
from database import LOCAL, connect, vector_literal
from experiment import MODEL, REVISION, tokens
from phase1 import Draft, DraftCase, Hit, chunks, rank_quality

DEPTH: Final = 20
RRF_K: Final = 60
MODES: Final = ("keyword", "vector", "rrf", "rrf_diversified")


class Timing(Frozen):
    embeddingMs: float
    vectorSearchMs: float
    keywordSearchMs: float
    fusionMs: float
    diversityMs: float


class Ranking(Frozen):
    documentRecallAt10: float
    ndcgAt5: float
    evidenceSpanRecallAt10: float
    totalSearchMs: float
    distinctDocumentsInTop10: int
    rankedDocumentIds: tuple[int, ...]
    rankedChunks: tuple[Hit, ...]


class Observation(Frozen):
    caseId: str
    variantLabel: str
    timing: Timing
    modes: dict[str, Ranking]


def ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def diversify(ranking: tuple[Hit, ...]) -> tuple[Hit, ...]:
    first_by_document: dict[int, Hit] = {}
    remaining = []
    for hit in ranking:
        if hit.documentId in first_by_document:
            remaining.append(hit)
        else:
            first_by_document[hit.documentId] = hit
    return (*first_by_document.values(), *remaining)


def measure(case: DraftCase, hits: tuple[Hit, ...], elapsed_ms: float) -> Ranking:
    ranked_documents = tuple(dict.fromkeys(hit.documentId for hit in hits))[:10]
    top_chunks = hits[:10]
    recall, ndcg, spans = rank_quality(case, ranked_documents, top_chunks)
    return Ranking(
        documentRecallAt10=recall, ndcgAt5=ndcg, evidenceSpanRecallAt10=spans,
        totalSearchMs=elapsed_ms, distinctDocumentsInTop10=len({hit.documentId for hit in top_chunks}),
        rankedDocumentIds=ranked_documents, rankedChunks=top_chunks,
    )


def compare(
    db: Connection, case: DraftCase, model: SentenceTransformer, lexical: BM25Okapi,
    children: tuple[Hit, ...],
) -> Observation:
    start = perf_counter()
    query = model.encode([f"query: {case.question}"], normalize_embeddings=True, show_progress_bar=False)[0]
    embedding_ms = ms(start)

    start = perf_counter()
    vector = tuple(Hit(documentId=doc, field=field, start=left, end=right)
                   for doc, field, left, right in db.execute(
        """select document_id,field,start_offset,end_offset
           from workspace1_chunks where workspace_id=1 and mode='fixed'
           order by embedding <=> %s::vector,document_id,field,start_offset limit %s""",
        (vector_literal(query.tolist()), DEPTH),
    ))
    vector_ms = ms(start)

    start = perf_counter()
    scores = lexical.get_scores(tokens(case.question))
    keyword = tuple(children[index] for index in sorted(
        (index for index, score in enumerate(scores) if score > 0),
        key=lambda index: (-scores[index], index),
    )[:DEPTH])
    keyword_ms = ms(start)

    start = perf_counter()
    fused: dict[Hit, float] = {}
    for hits in (keyword, vector):
        for rank, hit in enumerate(hits, 1):
            fused[hit] = fused.get(hit, 0.0) + 1 / (RRF_K + rank)
    rrf = tuple(sorted(fused, key=lambda hit: (-fused[hit], hit.documentId, hit.field, hit.start)))[:DEPTH]
    fusion_ms = ms(start)

    start = perf_counter()
    diverse = diversify(rrf)
    diversity_ms = ms(start)
    vector_total = embedding_ms + vector_ms
    rrf_total = vector_total + keyword_ms + fusion_ms
    return Observation(
        caseId=case.id, variantLabel=case.variantLabel,
        timing=Timing(embeddingMs=embedding_ms, vectorSearchMs=vector_ms, keywordSearchMs=keyword_ms,
                      fusionMs=fusion_ms, diversityMs=diversity_ms),
        modes={"keyword": measure(case, keyword, keyword_ms),
               "vector": measure(case, vector, vector_total),
               "rrf": measure(case, rrf, rrf_total),
               "rrf_diversified": measure(case, diverse, rrf_total + diversity_ms)},
    )


def percentiles(values: list[float]) -> dict[str, float]:
    p50, p95 = np.percentile(values, (50, 95)).tolist()
    return {"p50": p50, "p95": p95}


def metrics(rows: list[Observation]) -> dict:
    result = {}
    for mode in MODES:
        runs = [row.modes[mode] for row in rows]
        result[mode] = {
            "documentRecallAt10": mean(row.documentRecallAt10 for row in runs),
            "ndcgAt5": mean(row.ndcgAt5 for row in runs),
            "evidenceSpanRecallAt10": mean(row.evidenceSpanRecallAt10 for row in runs),
            "distinctDocumentsInTop10": mean(row.distinctDocumentsInTop10 for row in runs),
            "searchMs": percentiles([row.totalSearchMs for row in runs]),
        }
    return result


def main() -> None:
    docs = documents()
    draft = Draft.model_validate_json((LOCAL / "workspace1-evaluation-draft.json").read_bytes())
    source = sha256(json.dumps([doc.model_dump() for doc in docs], sort_keys=True,
                               ensure_ascii=False).encode()).hexdigest()
    if draft.sourceHash != source:
        raise SystemExit("The Workspace 1 evaluation set and source documents differ.")
    cases = [case for case in draft.cases if case.expectedType == "answer" and not case.history
             and case.expectedEvidenceIds]
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    tokenizer = model.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise SystemExit("The embedding tokenizer must return character offsets.")
    units = tuple(child for doc in docs for child in chunks(doc, "fixed", tokenizer))
    children = tuple(Hit(documentId=child.documentId, field=child.field, start=child.start, end=child.end)
                     for child in units)
    lexical = BM25Okapi([tokens(child.text) for child in units])
    model.encode(["query: 워밍업"], normalize_embeddings=True, show_progress_bar=False)
    with connect() as db:
        indexed = {Hit(documentId=doc, field=field, start=left, end=right)
                   for doc, field, left, right in db.execute(
                       "select document_id,field,start_offset,end_offset from workspace1_chunks "
                       "where workspace_id=1 and mode='fixed'",
                   )}
        if indexed != set(children):
            raise SystemExit("Prepare the matching phase 1 fixed-length index first.")
        results = [compare(db, case, model, lexical, children) for case in cases]

    labels = tuple(dict.fromkeys(case.variantLabel for case in cases))
    report = {
        "status": "observed_on_ai_draft", "sourceHash": source, "datasetVersion": draft.version,
        "createdAt": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "caseCount": len(cases), "candidateDepth": DEPTH, "rrfK": RRF_K, "generationCalls": 0,
        "metrics": metrics(results),
        "variantCounts": {label: sum(case.variantLabel == label for case in cases) for label in labels},
        "variantMetrics": {label: metrics([row for row in results if row.variantLabel == label]) for label in labels},
        "stageTimes": {stage: percentiles([getattr(row.timing, stage) for row in results])
                       for stage in Timing.model_fields},
        "results": [row.model_dump() for row in results],
    }
    path = LOCAL / "workspace1-phase2.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)
    print(json.dumps({"cases": len(cases), "metrics": report["metrics"], "stageTimes": report["stageTimes"]}, indent=2))


if __name__ == "__main__":
    main()
