"""Audit the 418-span Spring replay and compare expansions against the same excerpts."""
import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import phase4_retrieval as retrieval
from accuracy_eval import FIND, SYNTHESIS, SYNTHETIC_SPANS, Replay, span_match, synthetic_match
from evidence_expansion import expand_evidence
from phase3 import adjacent_paragraphs

VARIANTS = {
    "control": {},
    "2048 compact": dict(budget=2048, excerpt_tokens=160, compact=True),
    "2560 existing limits": dict(budget=2560, excerpt_tokens=160),
    "3072 existing limits": dict(budget=3072, excerpt_tokens=160),
    "3072 6 slots": dict(budget=3072, excerpt_tokens=160, passages_per_kind=6, supports_per_meeting=3),
    "3072 6 slots 220": dict(budget=3072, excerpt_tokens=220, passages_per_kind=6, supports_per_meeting=3),
    "3072 7 slots 120": dict(budget=3072, excerpt_tokens=120, passages_per_kind=7, supports_per_meeting=3),
    "3072 7 slots 160": dict(budget=3072, excerpt_tokens=160, passages_per_kind=7, supports_per_meeting=3),
    "3072 7 slots 220": dict(budget=3072, excerpt_tokens=220, passages_per_kind=7, supports_per_meeting=3),
    "3072 7 slots documents": dict(budget=3072, excerpt_tokens=160, passages_per_kind=7,
                                    supports_per_meeting=3, order="documents"),
    "4096 7 slots 220": dict(budget=4096, excerpt_tokens=220, passages_per_kind=7, supports_per_meeting=3),
}


def positions(passages):
    return sorted((p["documentId"], p["field"], p["start"], p["end"]) for p in passages)


def span_key(case, span):
    return case, span["documentId"], span["field"], span["startOffset"], span["endOffset"]


def audit_case(replay, workspace, case, request, observation):
    settings = retrieval.EVIDENCE_SELECTION_SETTINGS
    retriever = replay.state.retrievers[workspace]
    question = case["question"]
    request = request.model_copy(update={"question": question})
    keyword, vector = replay.rank_sources(request, settings.keyword_depth, settings.vector_depth)
    candidates = retriever.fuse_candidates(keyword, vector, settings)

    def windows(hits):
        return [adjacent_paragraphs(retriever.source[hit.documentId], hit).model_dump() for hit in hits]

    selected = []
    assemble = retrieval.assemble_worker_context

    def record_assembly(hits, *args, **kwargs):
        selected.extend(windows(hits))
        return assemble(hits, *args, **kwargs)

    with patch.object(retrieval, "assemble_worker_context", record_assembly):
        baseline = retriever.with_context(request, candidates, replay.reranker, replay.lock, replay.lock, settings)
    actual = [p.model_dump() for p in (*baseline.passages, *baseline.supportingPassages)]
    assert positions(actual) == positions(observation["passages"]), case["id"]

    def contexts(items):
        return tuple(retrieval.Context(hit=retrieval.Hit.model_validate(p.model_dump()), tokenCount=p.tokenCount)
                     for p in items)

    primary, support = contexts(baseline.passages), contexts(baseline.supportingPassages)
    passages = {"control": actual}
    for name, config in VARIANTS.items():
        if config:
            expanded, extra = expand_evidence(
                question, primary, support, candidates, retriever.source, retriever.tokenizer, **config,
            )
            passages[name] = [p.model_dump() for p in retriever.passages((*expanded, *extra))]
    return passages, (("search", windows(keyword + vector)), ("fusion", windows(candidates)), ("selection", selected))


async def run(root):
    async with retrieval.lifespan(retrieval.app):
        replay = Replay(root, retrieval.app.state)
        requests, observations = replay.requests("four-stage-verified-final")
        matches = {name: {1: set(), 458: set()} for name in VARIANTS}
        coverage = {name: Counter() for name in VARIANTS}
        tokens = {name: Counter() for name in VARIANTS}
        losses = {}
        for workspace in replay.workspaces:
            wid = workspace["id"]
            meetings = {doc["id"]: doc["meetingId"] for doc in workspace["documents"]}
            for case in workspace["cases"]:
                cid = case["id"]
                row = observations[wid, cid]
                if FIND in (row["actualRoute"], row["expectedRoute"]):
                    continue
                request = requests[wid, sha256(case["question"].encode()).hexdigest()]
                outputs, stages = audit_case(replay, wid, case, request, row)
                if wid == 1:
                    for span in replay.draft[cid]["evidence"]:
                        if not span_match(outputs["control"], span):
                            losses[span_key(cid, span)] = next(
                                (name for name, hits in stages if not span_match(hits, span)), "excerpt",
                            )
                for name, passages in outputs.items():
                    tokens[name][wid] += sum(p["tokenCount"] for p in passages)
                    if row["expectedRoute"] == SYNTHESIS:
                        covered = {meetings[p["documentId"]] for p in passages}
                        coverage[name][wid] += {meetings[d] for d in row["expectedIds"]} <= covered
                    if wid == 1:
                        matches[name][wid].update(span_key(cid, s) for s in replay.draft[cid]["evidence"]
                                                  if span_match(passages, s))
                    else:
                        matches[name][wid].update((cid, doc, index) for index, (doc, alternatives)
                                                  in enumerate(SYNTHETIC_SPANS.get(cid, ()))
                                                  if synthetic_match(passages, doc, alternatives))
        assert len(matches["control"][1]) == 418 and len(losses) == 108
        trials = []
        for name, settings in VARIANTS.items():
            gained = matches[name][1] - matches["control"][1]
            lost = matches["control"][1] - matches[name][1]
            trials.append(dict(name=name, settings=settings, included=len(matches[name][1]), gained=sorted(gained),
                               lost=sorted(lost), recovered=Counter(losses.get(key, "other") for key in gained),
                               meetings=dict(coverage[name]), w458=len(matches[name][458]), tokens=dict(tokens[name])))
        stage_counts = Counter(losses.values())
        output = dict(lossStages=stage_counts, losses=[[*key, stage] for key, stage in sorted(losses.items())],
                      trials=trials)
        (root / "remaining-108.json").write_text(json.dumps(output, indent=2) + "\n")
        print(stage_counts, flush=True)
        for row in trials:
            print(row["name"], row["included"], len(row["gained"]), len(row["lost"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    asyncio.run(run(parser.parse_args().root))
