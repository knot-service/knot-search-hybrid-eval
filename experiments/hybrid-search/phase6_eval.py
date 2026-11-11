"""Summarize the mixed Spring SSE replay without publishing questions or source text."""
from collections import Counter
from math import ceil
import json
from pathlib import Path
import re
import sys

from database import LOCAL
from phase5_eval import FIND, QUESTION, SYNTHESIS, emitted_claims, read, write

ROUTES = (FIND, QUESTION, SYNTHESIS)
LINE_BREAK = re.compile(r"\r\n|[\n\v\f\r\x85\u2028\u2029]")


def latency(rows: list[dict], key: str) -> dict:
    values = sorted(row[key] for row in rows if row[key] is not None)
    return {"samples": len(values), **{
        label: round(values[ceil(quantile * len(values)) - 1] / 1e6, 2) if values else None
        for label, quantile in (("p95", 0.95), ("p99", 0.99))
    }}


def route_summary(rows: list[dict], route: str) -> dict:
    group = [row for row in rows if row["expectedRoute"] == route]
    return {
        "route": route,
        "requests": len(group),
        "terminals": dict(Counter(row["terminal"] for row in group)),
        "firstMs": latency(group, "firstNanos"),
        "completedMs": latency(group, "completedNanos"),
        "generationCalls": sum(row["generationCalls"] for row in group),
    }


def expected_meetings(row: dict, meetings: dict[int, int]) -> set[int]:
    return {meetings[doc_id] for doc_id in row["expectedIds"]}


def hit_meetings(row: dict, meetings: dict[int, int], key: str) -> set[int]:
    return expected_meetings(row, meetings) & {meetings[doc_id] for doc_id in row[key]}


def cited_all_meetings(rows: list[dict], meetings: dict[int, int]) -> int:
    return sum(hit_meetings(row, meetings, "citedIds") == expected_meetings(row, meetings) for row in rows)


def worker_timings(path: Path, summary: dict) -> None:
    log = path.read_text()
    wait = {int(answer): int(ms) for answer, ms in re.findall(
        r"phase=queue_wait workspaceId=\d+ answerId=(\d+) elapsedMs=(\d+)", log)}
    retrieval = {int(answer): (route, int(ms)) for answer, route, ms in re.findall(
        r"Hybrid search workspaceId=\d+ answerId=(\d+) route=(\w+) questionMs=\d+ retrievalMs=(\d+)", log)}
    ordered = sorted(retrieval)
    expected = summary["warmupRequests"] + summary["requests"]
    if len(ordered) != expected or wait.keys() != retrieval.keys():
        raise SystemExit("Worker log does not contain exactly this load run; omit the log to summarize SSE only.")
    offset = summary["warmupRequests"]
    for level in summary["levels"]:
        count = sum(route["requests"] for route in level["routes"])
        answers = ordered[offset:offset + count]
        offset += count
        for route in level["routes"]:
            matching = [answer for answer in answers if retrieval[answer][0] == route["route"]]
            assert len(matching) == route["requests"]
            rank = ceil(0.95 * len(matching)) - 1
            route["workerQueueP95Ms"] = sorted(wait[answer] for answer in matching)[rank]
            route["workerRetrievalP95Ms"] = sorted(retrieval[answer][1] for answer in matching)[rank]


def summarize(log_path: Path | None = None) -> None:
    fixture = read("phase5-fixtures.json")
    baseline = read("phase5-spring-report.json")
    baseline_replay = {row["caseId"]: row for row in baseline["replay"]}
    baseline_sse = {row["caseId"]: row for row in baseline["sse"]}
    load = read("phase6-spring-load-report.json")
    rows = load["requests"]
    levels = sorted({row["concurrency"] for row in rows})
    top = max(levels)
    quality = []
    audit = []
    for workspace in fixture["workspaces"]:
        workspace_id = workspace["id"]
        docs = {doc["id"]: doc for doc in workspace["documents"]}
        meetings = {doc_id: doc["meetingId"] for doc_id, doc in docs.items()}
        cases = {case["id"]: case for case in workspace["cases"]}
        source_fields = {
            doc_id: tuple(LINE_BREAK.sub(" ", doc[field]) for field in ("body", "summary"))
            for doc_id, doc in docs.items()
        }
        observed = [row for row in rows if row["workspaceId"] == workspace_id]
        answered = [row for row in observed if row["expectedRoute"] != FIND and row["terminal"] == "completed"]
        comparable = [row for row in observed if row["expectedRoute"] != FIND and row["actualRoute"] is not None]
        peak: dict[str, dict] = {}
        context_regressions = 0
        context_improvements = 0
        for row in comparable:
            previous = baseline_replay[row["caseId"]]
            assert previous["workspaceId"] == workspace_id and previous["expectedIds"] == row["expectedIds"]
            before = hit_meetings(previous, meetings, "candidateIds")
            after = hit_meetings(row, meetings, "candidateIds")
            context_regressions += bool(before - after)
            context_improvements += bool(after - before)

        claims_count = 0
        uncited_claims = 0
        citation_count = 0
        exact_citations = 0
        no_claim_answers = 0
        for row in answered:
            claims = emitted_claims(row["answer"])
            claims_count += len(claims)
            no_claim_answers += not claims
            for claim in claims:
                uncited_claims += not claim["sources"]
                for source in claim["sources"]:
                    doc_id = source["documentId"]
                    citation_count += 1
                    exact_citations += (
                        doc_id in row["citedIds"] and doc_id in row["candidateIds"]
                        and any(source["quote"] in field for field in source_fields.get(doc_id, ()))
                    )
            if row["concurrency"] == top and row["caseId"] not in peak:
                peak[row["caseId"]] = row
                audit.append({"workspaceId": workspace_id, "caseId": row["caseId"],
                              "question": cases[row["caseId"]]["question"], "claims": claims,
                              "expectedIds": row["expectedIds"], "citedIds": row["citedIds"]})
        pairs = [row for row in peak.values() if row["expectedRoute"] == SYNTHESIS]
        previous_pairs = [baseline_sse[row["caseId"]] for row in pairs]
        quality.append({
            "workspaceId": workspace_id,
            "contextComparisons": len(comparable),
            "contextRegressions": context_regressions,
            "contextImprovements": context_improvements,
            "completedAnswers": len(answered),
            "noClaimAnswers": no_claim_answers,
            "claims": claims_count,
            "uncitedClaims": uncited_claims,
            "exactCitations": exact_citations,
            "citationCount": citation_count,
            "pairedSynthesis": len(pairs),
            "allCitedMeetingsBefore": cited_all_meetings(previous_pairs, meetings),
            "allCitedMeetingsAfter": cited_all_meetings(pairs, meetings),
        })

    lookup = [row for row in rows if row["expectedRoute"] == FIND]
    summary = {key: value for key, value in load.items() if key != "requests"}
    summary.update(
        requests=len(rows),
        completed=sum(row["terminal"] == "completed" for row in rows),
        routeMatches=sum(row["actualRoute"] == row["expectedRoute"] for row in rows),
        lookupRequests=len(lookup),
        lookupGenerationCalls=sum(row["generationCalls"] for row in lookup),
        lookupClassificationCalls=sum(row["classificationCalls"] for row in lookup),
        quality=quality,
        levels=[{"concurrency": level, "routes": [
            route_summary([row for row in rows if row["concurrency"] == level], route)
            for route in ROUTES
        ]} for level in levels],
    )
    review_file = LOCAL / "phase6-semantic-review.json"
    if review_file.is_file():
        review = read(review_file.name)
        verdicts = {(row["workspaceId"], row["caseId"], row["claimIndex"]): row["verdict"] for row in review["claims"]}
        expected = {(row["workspaceId"], row["caseId"], index)
                    for row in audit for index, _ in enumerate(row["claims"], 1)}
        assert set(verdicts) == expected
        for workspace in quality:
            workspace_id = workspace["workspaceId"]
            claims = [verdict for (owner, _, _), verdict in verdicts.items() if owner == workspace_id]
            workspace["semanticClaims"] = len(claims)
            workspace["supportedClaims"] = claims.count("supported")

    if log_path is not None:
        worker_timings(log_path, summary)
    write("phase6-citation-audit.json", {"cases": audit})
    write("phase6-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
