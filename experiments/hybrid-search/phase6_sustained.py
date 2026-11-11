"""Compare question priority and synthesis promotion when arrivals continue past the threshold."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from database import LOCAL
from phase5_eval import QUESTION, SYNTHESIS, read, write
from phase6_aging import compare
from phase6_qwen import QUEUE, by_answer


def queue_evidence(directory: Path, load: dict, threshold_ms: int) -> dict:
    log = (LOCAL / directory / "worker.xml").read_text()
    waits = by_answer(QUEUE, log)
    waves = defaultdict(list)
    for row in load["requests"]:
        waves[row["wave"]].append(row)

    def start_ms(row: dict) -> int:
        wait = int(waits[row["workspaceId"], row["answerId"]][0])
        return row["acceptedAtMs"] + wait

    evidence = {"waves": len(waves), "lateQuestionWaves": 0, "contestedAtThresholdWaves": 0, "overtookQueuedQuestionWaves": 0}
    for rows in waves.values():
        synthesis, = [row for row in rows if row["expectedRoute"] == SYNTHESIS]
        questions = [row for row in rows if row["expectedRoute"] == QUESTION]
        deadline = synthesis["acceptedAtMs"] + threshold_ms
        synthesis_start = start_ms(synthesis)
        evidence["lateQuestionWaves"] += any(row["acceptedAtMs"] > deadline for row in questions)
        queued = [row for row in questions if row["acceptedAtMs"] < deadline < start_ms(row)]
        if synthesis_start >= deadline and queued:
            evidence["contestedAtThresholdWaves"] += 1
            evidence["overtookQueuedQuestionWaves"] += any(start_ms(row) > synthesis_start for row in queued)
    return evidence


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before = read(str(args.before / "phase6-spring-load-report.json"))
    after = read(str(args.after / "phase6-spring-load-report.json"))
    plan = before["plannedDispatchDelaysMs"]
    threshold_ms = after["synthesisPromotionWaitMs"]
    if not plan or plan != after["plannedDispatchDelaysMs"] or max(plan) <= threshold_ms:
        raise ValueError("Both runs must use the same arrival plan and send requests past the promotion threshold.")
    meetings = {workspace["id"]: {doc["id"]: doc["meetingId"] for doc in workspace["documents"]}
                for workspace in read("phase5-fixtures.json")["workspaces"]}
    report = compare(args.before, args.after, "question_sustained", meetings)
    report["plannedDispatchDelaysMs"] = plan
    report["queueEvidence"] = {
        "before": queue_evidence(args.before, before, threshold_ms),
        "after": queue_evidence(args.after, after, threshold_ms),
    }
    print(write(str(args.after / "phase6-sustained-comparison.json"), report))
