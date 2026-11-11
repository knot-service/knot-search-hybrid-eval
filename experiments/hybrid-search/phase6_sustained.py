"""Compare question priority and synthesis promotion when arrivals continue past the threshold."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from database import LOCAL
from phase5_eval import QUESTION, SYNTHESIS, read, write
from phase6_aging import compare
from phase6_qwen import QUEUE, ROUTES, by_answer


def queue_evidence(directory: Path, load: dict, threshold_ms: int) -> dict:
    log = (LOCAL / directory / "worker.xml").read_text()
    waits = by_answer(QUEUE, log)
    waves = defaultdict(list)
    for row in load["requests"]:
        waves[row["wave"]].append(row)

    def start_ms(row: dict) -> int:
        wait = int(waits[row["workspaceId"], row["answerId"]][0])
        return row["acceptedAtMs"] + wait

    evidence = {"waves": len(waves), "lateQuestionWaves": 0, "contestedAtThresholdWaves": 0,
                "overtookQueuedQuestionWaves": 0, "overtookWaitingQuestionWaves": 0}
    for rows in waves.values():
        synthesis, = [row for row in rows if row["expectedRoute"] == SYNTHESIS]
        questions = [row for row in rows if row["expectedRoute"] == QUESTION]
        deadline = synthesis["acceptedAtMs"] + threshold_ms
        synthesis_start = start_ms(synthesis)
        evidence["lateQuestionWaves"] += any(row["acceptedAtMs"] > deadline for row in questions)
        evidence["overtookWaitingQuestionWaves"] += any(
            row["acceptedAtMs"] < synthesis_start < start_ms(row) for row in questions
        )
        queued = [row for row in questions if row["acceptedAtMs"] < deadline < start_ms(row)]
        if synthesis_start >= deadline and queued:
            evidence["contestedAtThresholdWaves"] += 1
            evidence["overtookQueuedQuestionWaves"] += any(start_ms(row) > synthesis_start for row in queued)
    return evidence


def first_delta_target(load: dict, target_ms: int) -> list[dict]:
    result = []
    for route in ROUTES:
        rows = [row for row in load["requests"] if row["expectedRoute"] == route]
        within = sum(row["firstNanos"] is not None and row["firstNanos"] <= target_ms * 1_000_000 for row in rows)
        result.append({"route": route, "within": within, "requests": len(rows)})
    return result


def paired_report(before_dir: Path, after_dir: Path, before: dict, meetings: dict, target_ms: int) -> dict:
    after = read(str(after_dir / "phase6-spring-load-report.json"))
    plan = before["plannedDispatchDelaysMs"]
    threshold_ms = after["synthesisPromotionWaitMs"]
    if not plan or plan != after["plannedDispatchDelaysMs"]:
        raise ValueError("Both runs must use the same arrival plan.")
    report = compare(before_dir, after_dir, "question_sustained", meetings)
    report["queueEvidence"] = {
        "before": queue_evidence(before_dir, before, threshold_ms),
        "after": queue_evidence(after_dir, after, threshold_ms),
    }
    report["firstDeltaTarget"] = first_delta_target(after, target_ms)
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[], help="Additional threshold run directory")
    parser.add_argument("--first-delta-target-ms", type=int, default=4000)
    args = parser.parse_args()
    if args.first_delta_target_ms <= 0:
        parser.error("--first-delta-target-ms must be positive")
    before = read(str(args.before / "phase6-spring-load-report.json"))
    meetings = {workspace["id"]: {doc["id"]: doc["meetingId"] for doc in workspace["documents"]}
                for workspace in read("phase5-fixtures.json")["workspaces"]}
    runs = [paired_report(args.before, directory, before, meetings, args.first_delta_target_ms)
            for directory in [args.after, *args.candidate]]
    runs.sort(key=lambda run: run["promotionWaitMs"])
    if len({run["promotionWaitMs"] for run in runs}) != len(runs):
        raise ValueError("Each threshold may be measured only once in the comparison.")
    target = {
        "beforeFirstDeltaTarget": first_delta_target(before, args.first_delta_target_ms),
        "firstDeltaTargetMs": args.first_delta_target_ms,
        "plannedDispatchDelaysMs": before["plannedDispatchDelaysMs"],
    }
    report = (target | runs[0] if not args.candidate else target | {
        "before": runs[0]["before"],
        "variants": [{key: value for key, value in run.items() if key != "before"} for run in runs],
    })
    output = (args.candidate[-1] if args.candidate else args.after) / "phase6-sustained-comparison.json"
    print(write(str(output), report))
