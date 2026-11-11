"""Replay captured Spring scopes and compare retrieval settings on fixed evidence."""
import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from phase4_retrieval import BASELINE, WORKER_SETTINGS, RankRequest, RetrievalSettings, app, lifespan
from phase5_eval import FIND, SYNTHESIS
from quality1_eval import SYNTHETIC_SPANS, span_match, synthetic_match
from database import LOCAL, vector_literal
from phase2 import fuse, percentiles
from phase3 import adjacent_paragraphs


def read(path):
    return json.loads(path.read_bytes())


class CachedReranker:
    def __init__(self, model):
        self.model = model
        self.scores = {}

    def predict(self, pairs, **options):
        missing = list(dict.fromkeys(pair for pair in pairs if pair not in self.scores))
        if missing:
            self.scores.update(zip(missing, self.model.predict(missing, **options), strict=True))
        return [self.scores[pair] for pair in pairs]


class Replay:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        self.reranker = CachedReranker(state.reranker)
        self.draft = {case["id"]: case for case in read(LOCAL / "workspace1-evaluation-draft.json")["cases"]}
        self.workspaces = read(root / "baseline" / "phase5-fixtures.json")["workspaces"]
        self.lock = Lock()

    @lru_cache(maxsize=None)
    def requests(self, cohort):
        source = self.root / cohort
        requests = {}
        for line in (source / "requests.jsonl").read_text().splitlines():
            row = RankRequest.model_validate_json(line)
            key = row.workspaceId, row.question
            if key in requests and requests[key] != row:
                raise ValueError(f"Different scopes for the same query in {cohort}")
            requests[key] = row
        return requests, {(r["workspaceId"], r["caseId"]): r for r in read(source / "retrieval-report.json")}

    @lru_cache(maxsize=None)
    def embedding(self, question):
        query = self.state.model.encode([f"query: {question}"], normalize_embeddings=True, show_progress_bar=False)[0]
        return vector_literal(query.tolist())

    @lru_cache(maxsize=None)
    def rank_sources(self, request, keyword_depth, vector_depth):
        return self.state.retrievers[request.workspaceId].rank_sources(
            request.question, self.embedding(request.question), request.allowedDocumentIds,
            BASELINE.model_copy(update=dict(keyword_depth=keyword_depth, vector_depth=vector_depth)),
        )

    def run(self, settings, cohort, *, compare_spring=False, selection_misses=None):
        requests, observations = self.requests(cohort)
        stats, matched = {}, {}
        for ws in self.workspaces:
            wid = ws["id"]
            documents = {d["id"]: d["meetingId"] for d in ws["documents"]}
            covered, count, synthesis_count, meetings_covered = set(), 0, 0, 0
            sizes = defaultdict(list)
            stage_counts = dict(allowed=0, search=0, fusion=0, selection=0)
            for case in ws["cases"]:
                row = observations[wid, case["id"]]
                if row["actualRoute"] == FIND or case["route"] == FIND:
                    continue
                request = requests.get((wid, sha256(case["question"].encode()).hexdigest()))
                passages = []
                stages = {}
                if request:
                    request = request.model_copy(update={"question": case["question"]})
                    keyword, vector = self.rank_sources(request, settings.keyword_depth, settings.vector_depth)
                    candidates = fuse(keyword, vector, settings.fusion_depth, settings.fusion_reserve)
                    selected = []
                    retriever = self.state.retrievers[wid]
                    result = retriever.with_context(
                        request, candidates, self.reranker,
                        self.lock, self.lock, settings, on_selected=selected.extend,
                    )
                    if wid == 1:
                        stages = {
                            name: [adjacent_paragraphs(retriever.source[hit.documentId], hit).model_dump()
                                   for hit in hits]
                            for name, hits in (("search", keyword + vector), ("fusion", candidates),
                                               ("selection", selected))
                        }
                    passages = [p.model_dump() for p in result.passages]
                    sizes["rerankCandidates"].append(len(candidates))
                elif row["retrievalNanos"]:
                    raise ValueError(f"Missing captured scope for {case['id']}")
                if compare_spring:
                    assert sorted(passages, key=lambda p: (p["documentId"], p["field"], p["start"], p["end"])) == sorted(
                        row["passages"], key=lambda p: (p["documentId"], p["field"], p["start"], p["end"])), case["id"]
                sizes["tokens"].append(sum(p["tokenCount"] for p in passages))
                if row["expectedRoute"] == SYNTHESIS:
                    synthesis_count += 1
                    meetings_covered += {documents[d] for d in row["expectedIds"]} <= {
                        documents[p["documentId"]] for p in passages}
                if wid == 1:
                    for span in self.draft[case["id"]]["evidence"]:
                        count += 1
                        if request:
                            stage_counts["allowed"] += span["documentId"] in request.allowedDocumentIds
                        stage_hits = {name: span_match(hits, span) for name, hits in stages.items()}
                        for name, matched_stage in stage_hits.items():
                            stage_counts[name] += matched_stage
                        if selection_misses is not None and stage_hits.get("fusion") and not stage_hits["selection"]:
                            document = span["documentId"]
                            meeting = documents[document]
                            selected_documents = {hit.documentId for hit in selected}
                            selected_meetings = {documents[doc] for doc in selected_documents}
                            reason = ("same_document_other_chunk" if document in selected_documents
                                      else "same_meeting_other_document" if meeting in selected_meetings
                                      else "meeting_not_selected")
                            selection_misses.append({
                                "case": case["id"], "document": document, "field": span["field"],
                                "start": span["startOffset"], "end": span["endOffset"], "reason": reason,
                                "mode": request.contextMode, "selectedPassages": len(selected),
                                "selectedMeetings": len(selected_meetings),
                            })
                        if span_match(passages, span):
                            covered.add((case["id"], span["documentId"], span["field"],
                                         span["startOffset"], span["endOffset"]))
                else:
                    for i, (doc, alternatives) in enumerate(SYNTHETIC_SPANS.get(case["id"], ())):
                        count += 1
                        if synthetic_match(passages, doc, alternatives):
                            covered.add((case["id"], doc, i))
            stats[wid] = {
                "matched": len(covered), "total": count, "meetings": meetings_covered,
                "synthesisCases": synthesis_count,
                **({"stages": stage_counts} if wid == 1 else {}),
                **{name: percentiles(values) for name, values in sizes.items()},
            }
            matched[wid] = covered
        return stats, matched


def compare(before, after):
    return {wid: {"gained": len(after[wid] - before[wid]), "lost": len(before[wid] - after[wid])}
            for wid in before}


def changed_spans(before, after):
    return {
        "gained": {wid: sorted(after[wid] - before[wid]) for wid in before},
        "lost": {wid: sorted(before[wid] - after[wid]) for wid in before},
    }


def accepted(candidate, baseline):
    return (candidate[1]["matched"] > baseline[1]["matched"]
            and all(candidate[wid]["meetings"] >= baseline[wid]["meetings"] for wid in baseline)
            and candidate[458]["matched"] >= baseline[458]["matched"])


async def run(root, stage):
    async with lifespan(app):
        replay = Replay(root, app.state)
        if stage == "supplement-overlap":
            misses = []
            control = WORKER_SETTINGS.model_copy(update={"supplement_overlap_penalty": 0.0})
            stats, before = replay.run(control, "dates", selection_misses=misses)
            assert (stats[1]["matched"], len(misses)) == (376, 122), stats
            print("control", stats, flush=True)
            trials = []
            for penalty, reuse in ((1.0, False), (2.0, False), (4.0, False), (2.0, True)):
                candidate = control.model_copy(update={
                    "supplement_overlap_penalty": penalty, "reuse_spare_slots": reuse,
                })
                candidate_stats, after = replay.run(candidate, "dates")
                delta = compare(before, after)
                trials.append({
                    "penalty": penalty, "reuse": reuse, "stats": candidate_stats, "delta": delta,
                    **changed_spans(before, after),
                })
                print("penalty", penalty, "reuse", reuse, candidate_stats, delta, flush=True)
            result = {"baseline": stats, "misses": misses, "trials": trials}
            (root / "supplement-overlap.json").write_text(json.dumps(result, indent=2) + "\n")
            return
        reports = {}
        if stage == "selection":
            settings = BASELINE
            baseline, before = replay.run(settings, "baseline", compare_spring=True)
            assert (baseline[1]["matched"], baseline[1]["total"]) == (295, 526), baseline
            cohort = "baseline"
        elif stage == "remaining":
            settings = RetrievalSettings.model_validate(read(root / "selection.json")["chosen"])
            cohort = "dates"
            baseline, before = replay.run(settings, "baseline")
        else:
            settings = RetrievalSettings.model_validate(read(root / "remaining.json")["chosen"])
            cohort = "dates"
            baseline, before = replay.run(settings, cohort)
        print("control", baseline, flush=True)
        initial = before

        def sweep(name, variants, *, control=cohort):
            nonlocal settings, baseline, before
            winner = (settings, baseline, before)
            result = []
            for changes in variants:
                trial = settings.model_copy(update=changes)
                stats, spans = replay.run(trial, control)
                delta = compare(before, spans)
                result.append({
                    "settings": trial.model_dump(), "stats": stats, "delta": delta,
                    **(changed_spans(before, spans) if stage == "selection" else {}),
                })
                print(name, changes, stats, delta, flush=True)
                if accepted(stats, winner[1]):
                    winner = trial, stats, spans
            settings, baseline, before = winner
            reports[name] = result

        if stage == "selection":
            sweep("selection", [{"supplement": m} for m in ("document", "chunk", "both")])
        elif stage == "remaining":
            dates, date_spans = replay.run(settings, cohort)
            reports["dates"] = {"stats": dates, "delta": compare(before, date_spans)}
            print("dates", reports["dates"], flush=True)
            baseline, before = dates, date_spans
            sweep("fusion", [{"fusion_depth": n} for n in (30, None)])
            sweep("search", [{key: n} for key in ("keyword_depth", "vector_depth") for n in (30, 40)])
            sweep("focus", [{"sentence_focus": True}])
            sweep("weak_cap", [{"weak_passage_tokens": n} for n in (160, None)])
            sweep("budget", [{"context_budget": n} for n in (1536, 2048)])
        else:
            sweep("fusion_after_budget",
                  [{"fusion_depth": n} for n in (22, 25, 30, None)]
                  + [{"fusion_reserve": n} for n in (4, 8)])
            sweep("joint_search", [{"vector_depth": n} for n in (30, 40)])
        result = {
            "chosen": settings.model_dump(), "stats": baseline, "variants": reports,
            "delta": compare(initial, before),
            **changed_spans(initial, before),
        }
        (root / f"{stage}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("chosen", result["chosen"], result["stats"], flush=True)


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("selection", "remaining", "followup", "supplement-overlap"))
    args = parser.parse_args()
    asyncio.run(run(args.root, args.stage))
