"""Replay captured Spring scopes to compare reserved excerpts and supplemental reranking."""
import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import phase4_retrieval as retrieval
from accuracy_eval import FIND, SYNTHESIS, SYNTHETIC_SPANS, Replay, read, span_match, synthetic_match
from evidence_expansion_eval import positions, span_key


VARIANTS = {
    "baseline": {},
    "short 128": {"short_passage_reserve": 128},
    "short 256": {"short_passage_reserve": 256},
    "short 384": {"short_passage_reserve": 384},
    "short spare only": {"short_passage_reserve": 256, "short_passages_spare_only": True},
    "rerank supplemental": {"rerank_supplemental": True},
    "short 256 and rerank": {"short_passage_reserve": 256, "rerank_supplemental": True},
}


async def run(root: Path) -> None:
    cohort = "same-document-full-512"
    audit = read(root / "same-document-evaluator" / "remaining-108.json")
    reference = next(trial for trial in audit["trials"] if trial["name"] == "4096 same document +512")
    recovered = set(map(tuple, reference["gained"]))
    stages = {tuple(row[:-1]): row[-1] for row in audit["losses"] if tuple(row[:-1]) not in recovered}
    settings = {name: retrieval.EVIDENCE_EXPANSION_SETTINGS.model_copy(update={
        "expansion_budget": 4096, "same_document_tokens": 512, **changes,
    }) for name, changes in VARIANTS.items()}
    matches = {name: {1: set(), 458: set()} for name in settings}
    coverage = {name: Counter() for name in settings}
    tokens = {name: Counter() for name in settings}
    async with retrieval.lifespan(retrieval.app):
        replay = Replay(root, retrieval.app.state)
        requests, observations = replay.requests(cohort)
        for workspace in replay.workspaces:
            wid = workspace["id"]
            retriever = replay.state.retrievers[wid]
            for case in workspace["cases"]:
                cid = case["id"]
                row = observations[wid, cid]
                if FIND in (row["expectedRoute"], row["actualRoute"]):
                    continue
                question = case["question"]
                request = requests[wid, sha256(question.encode()).hexdigest()].model_copy(update={"question": question})
                sources = replay.rank_sources(request, settings["baseline"].keyword_depth,
                                              settings["baseline"].vector_depth)
                candidates = retriever.fuse_candidates(*sources, settings["baseline"])
                for name, config in settings.items():
                    result = retriever.with_context(request, candidates, replay.reranker, replay.lock, replay.lock, config)
                    passages = [p.model_dump() for p in (*result.passages, *result.supportingPassages, *result.shortPassages)]
                    if name == "baseline":
                        assert positions(passages) == positions(row["passages"]), cid
                    used = sum(p["tokenCount"] for p in passages)
                    assert used <= 4096
                    tokens[name][wid] += used
                    if row["expectedRoute"] == SYNTHESIS:
                        meeting_ids = {retriever.source[p["documentId"]].meetingId for p in passages}
                        coverage[name][wid] += {retriever.source[d].meetingId for d in row["expectedIds"]} <= meeting_ids
                    if wid == 1:
                        matches[name][wid].update(
                            span_key(cid, s) for s in replay.draft[cid]["evidence"] if span_match(passages, s)
                        )
                    else:
                        matches[name][wid].update(
                            (cid, doc, index) for index, (doc, quotes) in enumerate(SYNTHETIC_SPANS.get(cid, ()))
                            if synthetic_match(passages, doc, quotes)
                        )
    baseline = matches["baseline"][1]
    assert len(baseline) == 486 and len(stages) == 40
    rows = []
    for name, result in matches.items():
        gained = result[1] - baseline
        lost = baseline - result[1]
        rows.append(dict(name=name, included=len(result[1]), gained=sorted(gained), lost=sorted(lost),
                         recovered=Counter(stages[key] for key in gained), meetings=dict(coverage[name]),
                         w458=len(result[458]), tokens=dict(tokens[name])))
        print(name, len(result[1]), len(gained), len(lost), dict(coverage[name]), len(result[458]), flush=True)
    (root / "short-passage-eval.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    asyncio.run(run(parser.parse_args().root))
