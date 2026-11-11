"""Compare a multilingual cross-encoder and bounded Parent context on fixed RRF candidates."""
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
import json
import re
from statistics import mean
from time import perf_counter
from typing import Final
from zoneinfo import ZoneInfo

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import PreTrainedTokenizerFast
import torch

from build_dataset import Frozen, SourceDoc, documents
from database import LOCAL
from experiment import MODEL, REVISION, tokens
from phase1 import Draft, DraftCase, Hit, rank_quality
from phase2 import ms, percentiles

RERANKER: Final = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION: Final = "1427fd652930e4ba29e8149678df786c240d8825"
CONTEXT_BUDGET: Final = 1024
WINDOW: Final = 240
CONTEXT_COUNT: Final = 5
SAMPLE_COUNTS: Final = {"documents": 14, "meeting": 8, "cross": 8}


class LabeledCase(DraftCase):
    sourceJob: str


class Phase3Dataset(Draft):
    cases: tuple[LabeledCase, ...]


class CandidateList(Frozen):
    rankedChunks: tuple[Hit, ...]


class CandidateRow(Frozen):
    caseId: str
    modes: dict[str, CandidateList]


class Phase2Report(Frozen):
    sourceHash: str
    datasetVersion: str
    results: tuple[CandidateRow, ...]


class Context(Frozen):
    hit: Hit
    tokenCount: int


class Quality(Frozen):
    ndcgAt5: float
    evidenceAt5: float
    evidenceInContext: float
    contextTokens: int
    contextMs: float


class Result(Frozen):
    caseId: str
    rerankMs: float
    baseline: Quality
    reranked: Quality
    expanded: Quality


def sample(cases: tuple[LabeledCase, ...], candidates: dict[str, CandidateRow]) -> tuple[LabeledCase, ...]:
    grouped: dict[str, list[LabeledCase]] = defaultdict(list)
    for case in cases:
        if case.id in candidates and case.variantLabel in ("기본", "구어체", "혼합형"):
            grouped[case.id.rsplit("-", 1)[0]].append(case)
    selected = []
    for cohort, count in SAMPLE_COUNTS.items():
        eligible = [name for name, group in grouped.items() if len(group) == 3 and group[0].sourceJob.startswith(cohort)]
        ordered = sorted(eligible, key=lambda name: sha256(f"phase3:{name}".encode()).digest())
        selected.extend(case for name in ordered[:count] for case in grouped[name])
    return tuple(selected)


def material(doc: SourceDoc, hit: Hit) -> str:
    return doc.body if hit.field == "body" else doc.summary


def adjacent_paragraphs(doc: SourceDoc, hit: Hit) -> Hit:
    text = material(doc, hit)
    if hit.field == "summary":
        return hit
    previous = text.rfind("\n\n", max(0, hit.start - WINDOW), hit.start)
    following = text.find("\n\n", hit.end, min(len(text), hit.end + WINDOW))
    return hit.model_copy(update={
        "start": previous + 2 if previous >= 0 else hit.start,
        "end": following if following >= 0 else hit.end,
    })


def assemble(
    hits: tuple[Hit, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, *, expand: bool, balanced: bool = False,
) -> tuple[Context, ...]:
    contexts: list[Context] = []
    remaining = CONTEXT_BUDGET
    selected = hits[:CONTEXT_COUNT]
    for index, hit in enumerate(selected):
        doc = source[hit.documentId]
        extent = adjacent_paragraphs(doc, hit) if expand else hit
        if any(item.hit.documentId == extent.documentId and item.hit.field == extent.field
               and item.hit.start <= extent.start and item.hit.end >= extent.end for item in contexts):
            continue
        text = material(doc, extent)
        offsets: list[tuple[int, int]] = tokenizer(
            text[extent.start:extent.end], add_special_tokens=False, return_offsets_mapping=True, verbose=False,
        )["offset_mapping"]
        budget = remaining
        if balanced:
            budget //= len(selected) - index
            # Give each meeting a share of the budget. Keep its matching child ahead of extra context.
            child_start = next((i for i, (_, end) in enumerate(offsets)
                                if extent.start + end > hit.start), 0)
            if len(offsets) > budget and child_start > 0:
                skip = min(child_start, len(offsets) - budget)
                offsets = offsets[skip:]
        taken = min(budget, len(offsets))
        if taken == 0:
            continue
        if balanced or taken < len(offsets):
            start = extent.start + offsets[0][0] if balanced else extent.start
            extent = extent.model_copy(update={"start": start, "end": extent.start + offsets[taken - 1][1]})
        contexts.append(Context(hit=extent, tokenCount=taken))
        remaining -= taken
        if remaining == 0:
            break
    return tuple(contexts)


def assemble_worker_context(
    hits: tuple[Hit, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, limits: dict[Hit, int], *,
    budget: int = CONTEXT_BUDGET, focus: str = "",
) -> tuple[Context, ...]:
    windows = []
    for hit in hits[:CONTEXT_COUNT]:
        doc = source[hit.documentId]
        extent = adjacent_paragraphs(doc, hit)
        offsets = tokenizer(
            material(doc, extent)[extent.start:extent.end], add_special_tokens=False,
            return_offsets_mapping=True, verbose=False,
        )["offset_mapping"]
        child_start = next((i for i, (_, end) in enumerate(offsets)
                            if extent.start + end > hit.start), len(offsets))
        child_end = next((i for i, (start, _) in enumerate(offsets)
                          if extent.start + start >= hit.end), len(offsets))
        limit = limits.get(hit, budget)
        windows.append((extent, offsets, child_start, min(child_end - child_start, limit), min(len(offsets), limit)))

    # Spend the shared budget on matching children before expanding any Parent.
    budgets = [0] * len(windows)
    remaining = budget
    for limits_by_window in ([window[3] for window in windows], [window[4] for window in windows]):
        while remaining and (open_windows := [i for i, limit in enumerate(limits_by_window) if budgets[i] < limit]):
            share = max(1, remaining // len(open_windows))
            for index in open_windows:
                added = min(share, limits_by_window[index] - budgets[index], remaining)
                budgets[index] += added
                remaining -= added

    contexts = []
    for (extent, offsets, child_start, _, _), taken in zip(windows, budgets, strict=True):
        if not taken:
            continue
        first = min(child_start, len(offsets) - taken)
        if focus and taken < len(offsets):
            text = material(source[extent.documentId], extent)[extent.start:extent.end]
            first = focused_start(text, offsets, taken, focus, first)
        passage = extent.model_copy(update={
            "start": extent.start + offsets[first][0],
            "end": extent.start + offsets[first + taken - 1][1],
        })
        if not any(item.hit.documentId == passage.documentId and item.hit.field == passage.field
                   and item.hit.start <= passage.start and item.hit.end >= passage.end for item in contexts):
            contexts.append(Context(hit=passage, tokenCount=taken))
    return tuple(contexts)


def focused_start(text: str, offsets: list[tuple[int, int]], taken: int, question: str, fallback: int) -> int:
    terms = set(tokens(question))
    boundaries = [0, *(m.end() for m in re.finditer(r"\n+|[.!?。！？](?=\s|$)", text)), len(text)]
    best = (0, 0)
    anchor = fallback
    for start, end in zip(boundaries, boundaries[1:]):
        overlap = len(terms.intersection(tokens(text[start:end])))
        if not overlap:
            continue
        left = next((i for i, (_, token_end) in enumerate(offsets) if token_end > start), len(offsets))
        right = next((i for i, (token_start, _) in enumerate(offsets) if token_start >= end), len(offsets))
        score = (overlap, -abs(left - fallback))
        if left < right and score > best:
            best = score
            anchor = min(max(0, right - taken, left - max(0, taken - (right - left)) // 2),
                         left, len(offsets) - taken)
    return anchor


def quality(
    case: DraftCase, hits: tuple[Hit, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, *, expand: bool,
) -> Quality:
    start = perf_counter()
    context = assemble(hits, source, tokenizer, expand=expand)
    context_ms = ms(start)
    ranked = tuple(dict.fromkeys(hit.documentId for hit in hits))
    _, ndcg, span_at_5 = rank_quality(case, ranked, hits[:CONTEXT_COUNT])
    _, _, in_context = rank_quality(case, ranked, tuple(item.hit for item in context))
    return Quality(ndcgAt5=ndcg, evidenceAt5=span_at_5, evidenceInContext=in_context,
                   contextTokens=sum(item.tokenCount for item in context), contextMs=context_ms)


def main() -> None:
    docs = documents()
    by_id = {doc.id: doc for doc in docs}
    draft = Phase3Dataset.model_validate_json((LOCAL / "workspace1-evaluation-draft.json").read_bytes())
    previous = Phase2Report.model_validate_json((LOCAL / "workspace1-phase2.json").read_bytes())
    source_hash = sha256(json.dumps([doc.model_dump() for doc in docs], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()
    if draft.sourceHash != source_hash or previous.sourceHash != source_hash or draft.version != previous.datasetVersion:
        raise SystemExit("The phase 2 candidates and the source dataset must match.")
    candidates = {item.caseId: item for item in previous.results}
    selected = sample(draft.cases, candidates)
    if len(selected) != 3 * sum(SAMPLE_COUNTS.values()):
        raise SystemExit("Not enough matched phase 2 cases for the fixed sample.")

    torch.set_num_threads(4)
    encoder = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    tokenizer = encoder.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise SystemExit("The context tokenizer must return character offsets.")
    reranker = CrossEncoder(RERANKER, revision=RERANKER_REVISION, max_length=384,
                            device="cpu", trust_remote_code=False)
    reranker.predict([("워밍업", "검색 예시")], show_progress_bar=False)

    results = []
    for case in selected:
        before = candidates[case.id].modes["rrf"].rankedChunks
        pairs = [(case.question, f"{by_id[hit.documentId].title}\n"
                  f"{material(by_id[hit.documentId], hit)[hit.start:hit.end]}") for hit in before]
        started = perf_counter()
        scores = reranker.predict(pairs, batch_size=10, show_progress_bar=False)
        after = tuple(hit for hit, _ in sorted(zip(before, scores, strict=True),
                                               key=lambda entry: -float(entry[1])))
        rerank_ms = ms(started)
        results.append(Result(
            caseId=case.id, rerankMs=rerank_ms,
            baseline=quality(case, before, by_id, tokenizer, expand=False),
            reranked=quality(case, after, by_id, tokenizer, expand=False),
            expanded=quality(case, after, by_id, tokenizer, expand=True),
        ))
        if len(results) % 15 == 0:
            print(f"completed={len(results)} total={len(selected)}", flush=True)

    metrics = {}
    for mode in ("baseline", "reranked", "expanded"):
        rows = [getattr(row, mode) for row in results]
        metrics[mode] = {
            "ndcgAt5": mean(row.ndcgAt5 for row in rows),
            "evidenceAt5": mean(row.evidenceAt5 for row in rows),
            "evidenceInContext": mean(row.evidenceInContext for row in rows),
            "contextTokens": mean(row.contextTokens for row in rows),
            "contextMs": percentiles([row.contextMs for row in rows]),
        }
    report = {
        "status": "observed_on_ai_draft", "sourceHash": source_hash, "datasetVersion": draft.version,
        "createdAt": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "caseCount": len(selected),
        "generationCalls": 0, "reranker": RERANKER, "rerankerRevision": RERANKER_REVISION,
        "contextTokenizer": MODEL, "contextBudget": CONTEXT_BUDGET, "metrics": metrics,
        "rerankMs": percentiles([row.rerankMs for row in results]),
        "results": [row.model_dump() for row in results],
    }
    path = LOCAL / "workspace1-phase3.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)
    print(json.dumps({"cases": len(selected), "metrics": metrics, "rerankMs": report["rerankMs"]}, indent=2))


if __name__ == "__main__":
    main()
