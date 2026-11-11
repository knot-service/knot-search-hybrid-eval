"""Compare wider retrieval with the original representative documents held fixed."""
import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from accuracy_eval import FIND, SYNTHESIS, SYNTHETIC_SPANS, Replay, read, span_match, synthetic_match
from evidence_expansion_eval import positions, span_key
import phase4_retrieval as retrieval
from phase2 import fuse
from phase3 import adjacent_paragraphs


VARIANTS = {
    "depth60": (60, 32, False, False, False),
    "depth120": (120, 64, False, False, False),
    "depth120-compact": (120, 64, True, False, False),
    "depth120-windows": (120, 64, True, True, False),
    "depth120-uncovered": (120, 64, True, True, True),
    "depth120-existing-first": (120, 64, True, False, False),
    "depth120-adaptive": (120, 64, True, True, True),
}


async def run(root: Path, targeted: bool, variant: str | None) -> None:
    audit = read(root / "same-document-evaluator" / "remaining-108.json")
    reference = next(trial for trial in audit["trials"] if trial["name"] == "4096 same document +512")
    recovered = set(map(tuple, reference["gained"]))
    targets = {tuple(row[:-1]): row[-1] for row in audit["losses"]
               if row[-1] in ("search", "fusion") and tuple(row[:-1]) not in recovered}
    assert len(targets) == 14
    baseline_settings = retrieval.EVIDENCE_EXPANSION_SETTINGS.model_copy(update={
        "expansion_budget": 4096, "same_document_tokens": 512,
        "short_passage_reserve": 256, "short_passages_spare_only": True,
    })
    baseline_spans = {1: set(), 458: set()}
    variants = {variant: VARIANTS[variant]} if variant else VARIANTS
    spans = {name: {1: set(), 458: set()} for name in variants}
    reach = {name: set() for name in variants}
    meeting_coverage = {name: Counter() for name in variants}
    target_cases = {key[0] for key in targets}
    async with retrieval.lifespan(retrieval.app):
        replay = Replay(root, retrieval.app.state)
        requests, recorded = replay.requests("short-spare-full")
        for workspace in replay.workspaces:
            wid = workspace["id"]
            if targeted and wid != 1:
                continue
            retriever = replay.state.retrievers[wid]
            for case in workspace["cases"]:
                cid, question = case["id"], case["question"]
                if targeted and cid not in target_cases:
                    continue
                row = recorded[wid, cid]
                if FIND in (row["expectedRoute"], row["actualRoute"]):
                    continue
                request = requests[wid, sha256(question.encode()).hexdigest()].model_copy(update={"question": question})
                keyword, vector = replay.rank_sources(request, 120, 120)
                core = retriever.fuse_candidates(keyword[:baseline_settings.keyword_depth],
                                                vector[:baseline_settings.vector_depth], baseline_settings)
                baseline = retriever.with_context(request, core, replay.reranker, replay.lock, replay.lock, baseline_settings)

                def all_passages(result):
                    return [p.model_dump() for p in (*result.passages, *result.supportingPassages,
                                                     *result.shortPassages, *result.discoveryPassages)]

                def matches(passages):
                    if wid == 1:
                        return {span_key(cid, s) for s in replay.draft[cid]["evidence"] if span_match(passages, s)}
                    return {(cid, doc, index) for index, (doc, quotes) in enumerate(SYNTHETIC_SPANS.get(cid, ()))
                            if synthetic_match(passages, doc, quotes)}

                original = all_passages(baseline)
                assert positions(original) == positions(row["passages"]), cid
                baseline_spans[wid].update(matches(original))
                for name, (depth, limit, compact, window_rerank, uncovered) in variants.items():
                    wide_keyword, wide_vector = keyword[:depth], vector[:depth]
                    existing_first = name in ("depth120-existing-first", "depth120-adaptive")
                    config = baseline_settings.model_copy(update={
                        "discovery_rerank_limit": limit, "discovery_compact": compact,
                        "discovery_window_rerank": window_rerank,
                        "discovery_uncovered_only": uncovered,
                        "discovery_existing_first": existing_first,
                        "discovery_excerpt_tokens": 320 if existing_first else 96,
                    })
                    result = baseline
                    if request.contextMode == "synthesis":
                        result = retriever.discover(question, wide_keyword, wide_vector, core, baseline,
                                                    replay.reranker, replay.lock, replay.lock, config)
                    passages = all_passages(result)
                    assert sum(p["tokenCount"] for p in passages) <= 4096
                    assert {p.documentId for p in result.passages} == {p.documentId for p in baseline.passages}
                    spans[name][wid].update(matches(passages))
                    if row["expectedRoute"] == SYNTHESIS:
                        meetings = {retriever.source[p["documentId"]].meetingId for p in passages}
                        meeting_coverage[name][wid] += {retriever.source[d].meetingId for d in row["expectedIds"]} <= meetings
                    if wid == 1 and cid in target_cases:
                        extra = [h for h in fuse(wide_keyword, wide_vector, None) if h.field == "body" and h not in core][:limit]
                        windows = [adjacent_paragraphs(retriever.source[h.documentId], h) for h in extra]
                        reach[name].update(key for key in targets if key[0] == cid and any(
                            h.documentId == key[1] and h.field == key[2] and h.start <= key[3] and h.end >= key[4]
                            for h in windows
                        ))
    if not targeted:
        assert len(baseline_spans[1]) == 487
    rows = []
    for name, current in spans.items():
        gained, lost = current[1] - baseline_spans[1], baseline_spans[1] - current[1]
        row = dict(name=name, included=len(current[1]), gained=sorted(gained), lost=sorted(lost),
                   targetReached=len(reach[name]), targetRecovered=len(gained & targets.keys()),
                   meetings=dict(meeting_coverage[name]), w458=len(current[458]))
        rows.append(row)
        print(name, len(current[1]), len(gained), len(lost), row["targetReached"], row["targetRecovered"], flush=True)
    filename = "discovery-targeted" if targeted else "discovery-full"
    if variant:
        filename = f"{filename}-{variant}"
    (root / f"{filename}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    asyncio.run(run(args.root, args.targeted, args.variant))
