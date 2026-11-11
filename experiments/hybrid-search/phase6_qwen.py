"""Compare both Worker settings using the same mixed Spring SSE cohort."""

from argparse import ArgumentParser
from collections import Counter
from math import ceil
from pathlib import Path
import re

from database import LOCAL
from phase5_eval import FIND, QUESTION, read, write
from phase6_eval import ROUTES, hit_meetings, sample_key, summarize


QUEUE = re.compile(r"phase=queue_wait workspaceId=(\d+) answerId=(\d+) elapsedMs=(\d+)")
SEARCH = re.compile(
    r"Hybrid search workspaceId=(\d+) answerId=(\d+) route=(\w+) questionMs=\d+ "
    r"retrievalMs=(\d+) rerankMs=(\S+)"
)
QWEN = re.compile(
    r"phase=llm_call operation=stream workspaceId=(\d+) answerId=(\d+) elapsedMs=(\d+) "
    r"messageContentChars=\d+ firstContentMs=(\d+) [^\n]*? outcome=success"
)
STAGES = ("workerQueue", "retrieval", "rerank", "qwenFirstContent", "qwenStreamTail", "firstSse", "completedSse")


def by_answer(pattern: re.Pattern, log: str) -> dict:
    result = {}
    for workspace, answer, *fields in pattern.findall(log):
        key = (int(workspace), int(answer))
        if key in result:
            raise ValueError(f"Duplicate timing for answer {answer} in Workspace {workspace}.")
        result[key] = fields
    return result


def timings(rows: list[dict], log: str) -> list[dict]:
    queue = by_answer(QUEUE, log)
    search = by_answer(SEARCH, log)
    qwen = by_answer(QWEN, log)
    measurements = []
    for row in rows:
        key = (row["workspaceId"], row["answerId"])
        first = row["firstNanos"]
        completed = row["completedNanos"]
        values = dict.fromkeys(STAGES)
        values["firstSse"] = first / 1e6 if first is not None else None
        values["completedSse"] = completed / 1e6 if completed is not None else None
        if key in search:
            route, retrieval, rerank = search[key]
            if route != row["actualRoute"] or key not in queue:
                raise ValueError(f"Worker log disagrees with replay for answer {key[1]}.")
            values.update(workerQueue=int(queue[key][0]), retrieval=int(retrieval), rerank=float(rerank))
        elif row["actualRoute"] is not None:
            raise ValueError(f"Missing Worker search timing for answer {key[1]}.")
        if key in qwen:
            total, before_first = map(int, qwen[key])
            if total < before_first or row["generationCalls"] != 1 or row["expectedRoute"] == FIND:
                raise ValueError(f"Unexpected Qwen stream timing for answer {key[1]}.")
            values.update(qwenFirstContent=before_first, qwenStreamTail=total - before_first)
        elif completed is not None and row["generationCalls"] == 1:
            raise ValueError(f"Missing successful Qwen stream timing for answer {key[1]}.")
        measurements.append(values)
    return measurements


def percentile(values: list[float], fraction: float) -> float | None:
    return round(sorted(values)[ceil(fraction * len(values)) - 1], 2) if values else None


def stage_summary(group: list[dict]) -> dict:
    stages = {}
    for stage in STAGES:
        values = [row[stage] for row in group if row[stage] is not None]
        stages[stage] = {"samples": len(values), "p95": percentile(values, .95), "p99": percentile(values, .99)}
    return stages


def first_sse_tail(group: list[dict]) -> list[dict]:
    # Select the actual request at each SSE percentile, then show its own timing components.
    ordered = sorted((row for row in group if row["firstSse"] is not None), key=lambda row: row["firstSse"])
    if not ordered:
        return []
    tail = []
    for label, fraction in (("p95", .95), ("p99", .99)):
        row = ordered[ceil(fraction * len(ordered)) - 1]
        tail.append({"percentile": label, **{stage: round(ms, 2) if ms is not None else None for stage, ms in row.items()}})
    return tail


def run_summary(directory: Path, load: dict, context_regressions: Counter) -> dict:
    summary = summarize(run_dir=directory, compare_shared=False)
    rows = load["requests"]
    measured = timings(rows, (LOCAL / directory / "worker.xml").read_text())
    groups: dict[tuple, list[dict]] = {}
    for row, stages in zip(rows, measured, strict=True):
        groups.setdefault((row["concurrency"], row["expectedRoute"]), []).append(stages)

    quality = [{
        **{field: row[field] for field in (
            "workspaceId", "completedAnswers", "exactCitations", "citationCount", "claims", "uncitedClaims", "noClaimAnswers",
            "pairedSynthesis", "allCitedMeetingsAfter", "semanticClaims", "supportedClaims",
        ) if field in row},
        "meetingContextRegressions": context_regressions[row["workspaceId"]],
    } for row in summary["quality"]]
    return {
        **{field: summary[field] for field in (
            "warmupRequests", "wavesPerLevel", "workerConcurrency", "lookupConcurrency", "streamConcurrency",
            "pollIntervalMs", "requests", "completed", "routeMatches", "lookupRequests", "lookupGenerationCalls",
            "lookupClassificationCalls",
        )},
        "quality": quality,
        "levels": [{"concurrency": concurrency, "routes": [{
            "route": route,
            "stages": stage_summary(groups[concurrency, route]),
            "firstSseTail": first_sse_tail(groups[concurrency, route]) if route == QUESTION else [],
        } for route in ROUTES]} for concurrency in sorted({row["concurrency"] for row in rows})],
    }


def compare(shared_dir: Path, split_dir: Path) -> None:
    before = read(str(shared_dir / "phase6-spring-load-report.json"))
    after = read(str(split_dir / "phase6-spring-load-report.json"))
    before_rows = {sample_key(row): row for row in before["requests"]}
    after_rows = {sample_key(row): row for row in after["requests"]}
    for field in ("workerConcurrency", "streamConcurrency", "pollIntervalMs", "warmupRequests", "wavesPerLevel"):
        if before[field] != after[field]:
            raise ValueError(f"Replay settings differ for {field}.")
    if before["lookupConcurrency"] != 0 or after["lookupConcurrency"] < 1:
        raise ValueError("The first run must share Workers and the second must isolate document lookup.")
    if len(before_rows) != len(before["requests"]) or len(after_rows) != len(after["requests"]) or before_rows.keys() != after_rows.keys():
        raise ValueError("Replays must contain exactly one of each request in the same cohort.")
    meetings = {workspace["id"]: {doc["id"]: doc["meetingId"] for doc in workspace["documents"]}
                for workspace in read("phase5-fixtures.json")["workspaces"]}
    regressions = Counter()
    comparisons = Counter()
    for key, row in after_rows.items():
        previous = before_rows[key]
        if row["expectedIds"] != previous["expectedIds"] or row["expectedRoute"] != previous["expectedRoute"]:
            raise ValueError(f"Evaluation targets differ for {key}.")
        if row["expectedRoute"] == FIND:
            continue
        workspace = row["workspaceId"]
        if row["actualRoute"] != row["expectedRoute"] or previous["actualRoute"] != previous["expectedRoute"]:
            continue
        comparisons[workspace] += 1
        before_hits = hit_meetings(previous, meetings[workspace], "candidateIds")
        after_hits = hit_meetings(row, meetings[workspace], "candidateIds")
        regressions[workspace] += bool(before_hits - after_hits)

    result = {
        "shared": run_summary(shared_dir, before, Counter()),
        "split": run_summary(split_dir, after, regressions),
        "contextComparisons": [{"workspaceId": workspace, "requests": count}
                               for workspace, count in sorted(comparisons.items())],
    }
    print(write(str(split_dir / "phase6-qwen-comparison.json"), result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("shared", type=Path, help="Shared-Worker run directory under .local")
    parser.add_argument("split", type=Path, help="Split-Worker run directory under .local")
    args = parser.parse_args()
    compare(args.shared, args.split)
