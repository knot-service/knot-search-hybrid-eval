"""Compare synthesis wait promotion against question priority under two load patterns."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import re

from database import LOCAL
from phase5_eval import QUESTION, SYNTHESIS, read, write
from phase6_qwen import QUEUE, by_answer, compare_cohort, generation_summary


def promotions(directory: Path, load: dict) -> dict[int, int]:
    log = (LOCAL / directory / "worker.xml").read_text()
    promoted = set(map(int, re.findall(r"phase=synthesis_promoted answerId=(\d+)", log)))
    queue = by_answer(QUEUE, log)
    counts = Counter()
    for row in load["requests"]:
        if row["answerId"] not in promoted:
            continue
        wait = int(queue[row["workspaceId"], row["answerId"]][0])
        if row["expectedRoute"] != SYNTHESIS or wait < load["synthesisPromotionWaitMs"]:
            raise ValueError("An unexpected request was promoted.")
        counts[row["concurrency"]] += 1
    return dict(counts)


def waves_with_earlier_synthesis(directory: Path, load: dict) -> int:
    log = (LOCAL / directory / "worker.xml").read_text()
    starts = {int(answer): index for index, (_, answer, _) in enumerate(QUEUE.findall(log))}
    waves: dict[int, list[dict]] = {}
    for row in load["requests"]:
        waves.setdefault(row["wave"], []).append(row)
    return sum(
        min(starts[row["answerId"]] for row in rows if row["expectedRoute"] == SYNTHESIS)
        < max(starts[row["answerId"]] for row in rows if row["expectedRoute"] == QUESTION)
        for rows in waves.values()
    )


def compare(before_dir: Path, after_dir: Path, pattern: str, meetings: dict) -> dict:
    before = read(str(before_dir / "phase6-spring-load-report.json"))
    after = read(str(after_dir / "phase6-spring-load-report.json"))
    if ((LOCAL / before_dir / "phase5-fixtures.json").read_bytes()
            != (LOCAL / after_dir / "phase5-fixtures.json").read_bytes()):
        raise ValueError("The replay snapshots differ.")
    for run in (before, after):
        if (run.get("loadPattern", "balanced") != pattern or run["workerConcurrency"] != 2
                or run["lookupConcurrency"] != 1 or not run["shareGenerationSlots"]):
            raise ValueError("The replay pattern or Worker settings do not match.")
    if before.get("synthesisPromotionWaitMs", 0) != 0 or after["synthesisPromotionWaitMs"] <= 0:
        raise ValueError("The first run must use question priority and the second must enable promotion.")
    regressions, comparisons = compare_cohort(before, after, meetings)
    result = {
        "before": generation_summary(before_dir, before, "priority", meetings, Counter()),
        "after": generation_summary(after_dir, after, "priority", meetings, regressions),
        "contextComparisons": comparisons,
        "promotedByConcurrency": promotions(after_dir, after),
        "promotionWaitMs": after["synthesisPromotionWaitMs"],
    }
    if pattern == "question_surge":
        result["wavesWithEarlierSynthesis"] = {
            "before": waves_with_earlier_synthesis(before_dir, before),
            "after": waves_with_earlier_synthesis(after_dir, after),
        }
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("balanced_before", type=Path)
    parser.add_argument("balanced_after", type=Path)
    parser.add_argument("surge_before", type=Path)
    parser.add_argument("surge_after", type=Path)
    args = parser.parse_args()
    meetings = {workspace["id"]: {doc["id"]: doc["meetingId"] for doc in workspace["documents"]}
                for workspace in read("phase5-fixtures.json")["workspaces"]}
    balanced = compare(args.balanced_before, args.balanced_after, "balanced", meetings)
    surge = compare(args.surge_before, args.surge_after, "question_surge", meetings)
    if balanced["promotionWaitMs"] != surge["promotionWaitMs"]:
        raise ValueError("The promotion thresholds differ.")
    print(write(str(args.balanced_after / "phase6-aging-comparison.json"), {"balanced": balanced, "surge": surge}))
