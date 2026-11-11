"""Prepare the paired corpus replay and summarize meeting coverage and emitted citations."""
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
from statistics import mean
import sys
from tempfile import NamedTemporaryFile

from database import LOCAL

LIVE = {
    1: {"W1-0001-01", "W1-0024-01", "W1-0033-01", "W1-0062-01",
        "W1-0322-01", "W1-0326-01", "W1-0334-01"},
    458: {"S04-01", "S16-01", "S18-01", "S21-01", "S53-01", "S53-02", "S71-01"},
}
FIND = "DOCUMENT_FIND"
QUESTION = "MEETING_QUESTION"
SYNTHESIS = "MULTI_MEETING_SYNTHESIS"
SOURCE = re.compile(r"^  - 문서 #(\d+) 근거: (.*)$")


def read(name: str) -> dict:
    return json.loads((LOCAL / name).read_bytes())


def write(name: str, data: dict) -> Path:
    target = LOCAL / name
    with NamedTemporaryFile(mode="w", dir=LOCAL, encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
        json.dump(data, output, ensure_ascii=False, indent=2)
    temporary.replace(target)
    return target


def prepare() -> None:
    dataset = read("phase4-fixtures.json")
    for workspace in dataset["workspaces"]:
        workspace_id = workspace["id"]
        cases = workspace["cases"]
        present = {case["id"] for case in cases}
        if not LIVE[workspace_id] <= present:
            raise SystemExit(f"Missing fixed live cases in Workspace {workspace_id}.")
        if workspace_id == 1:
            eligible = [case for case in cases if case["route"] == QUESTION and case["id"].endswith("-01")]
            sampled = sorted(eligible, key=lambda case: sha256(f"phase5:{case['id']}".encode()).digest())[:28]
            selected = {case["id"] for case in sampled} | LIVE[workspace_id]
            cases = [case for case in cases if case["route"] != QUESTION or case["id"] in selected]
            workspace["cases"] = cases
        for case in cases:
            case["live"] = case["id"] in LIVE[workspace_id]
        print(f"Workspace {workspace_id}: {len(cases)} retrieval; {len(LIVE[workspace_id])} live")
    print(write("phase5-fixtures.json", dataset))


def coverage(rows: list[dict], meetings: dict[int, int]) -> dict:
    recalls = []
    complete = 0
    for row in rows:
        expected = {meetings[doc_id] for doc_id in row["expectedIds"]}
        actual = {meetings[doc_id] for doc_id in row["candidateIds"]}
        recalls.append(len(expected & actual) / len(expected))
        complete += expected <= actual
    return {"cases": len(rows), "allMeetings": complete, "meetingRecall": mean(recalls)}


def latency(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"p50": ordered[ceil(0.5 * len(ordered)) - 1], "p95": ordered[ceil(0.95 * len(ordered)) - 1]}


def emitted_claims(answer: str) -> list[dict]:
    claims: list[dict] = []
    for line in answer.splitlines():
        source = SOURCE.fullmatch(line)
        if source and claims:
            claims[-1]["sources"].append({"documentId": int(source[1]), "quote": source[2]})
        elif line.startswith("- "):
            claims.append({"text": line[2:], "sources": []})
    return claims


def summarize() -> None:
    dataset = read("phase5-fixtures.json")
    before = {row["caseId"]: row for row in read("phase4-spring-report.json")["replay"]}
    after = read("phase5-spring-report.json")
    draft = {case["id"]: case for case in read("workspace1-evaluation-draft.json")["cases"]}
    summary = {}
    audit = []
    for workspace in dataset["workspaces"]:
        workspace_id = workspace["id"]
        docs = {doc["id"]: doc for doc in workspace["documents"]}
        meetings = {doc_id: doc["meetingId"] for doc_id, doc in docs.items()}
        cases = {case["id"]: case for case in workspace["cases"]}
        rows = {row["caseId"]: row for row in after["replay"] if row["workspaceId"] == workspace_id}
        assert rows.keys() == cases.keys()
        assert all(before[case_id]["workspaceId"] == workspace_id
                   and before[case_id]["expectedIds"] == row["expectedIds"]
                   and before[case_id]["version"] == row["version"] for case_id, row in rows.items())
        syntheses = [row for row in rows.values() if row["expectedRoute"] == SYNTHESIS]
        generated = [row for row in rows.values() if row["actualRoute"] != FIND]
        lookup = [row for row in rows.values() if row["actualRoute"] == FIND]
        assert all(not row["passages"] and row["rerankMs"] == 0 and row["contextMs"] == 0 for row in lookup)
        assert all(sum(passage["tokenCount"] for passage in row["passages"]) <= 1024 for row in generated)
        data = {
            "replay": len(rows),
            "synthesisBefore": coverage([before[row["caseId"]] for row in syntheses], meetings),
            "synthesisAfter": coverage(syntheses, meetings),
            "rerankMs": latency([row["rerankMs"] for row in generated if row["passages"]]),
            "contextMs": latency([row["contextMs"] for row in generated if row["passages"]]),
            "documentRecallBefore": mean(len(set(row["expectedIds"]) & set(before[row["caseId"]]["candidateIds"]))
                                         / len(row["expectedIds"]) for row in generated),
            "documentRecallAfter": mean(len(set(row["expectedIds"]) & set(row["candidateIds"]))
                                        / len(row["expectedIds"]) for row in generated),
        }
        if workspace_id == 1:
            checked = [row for row in generated if row["expectedRoute"] != FIND]
            spans = [(span, row["passages"]) for row in checked for span in draft[row["caseId"]]["evidence"]]
            present = sum(any(p["documentId"] == span["documentId"] and p["field"] == span["field"]
                              and p["start"] <= span["startOffset"] and p["end"] >= span["endOffset"]
                              for p in passages) for span, passages in spans)
            data["goldSpans"] = {"matched": present, "total": len(spans)}
        live = [row for row in after["sse"] if row["workspaceId"] == workspace_id]
        data["live"] = {route: [row["generationCalls"] for row in live if row["actualRoute"] == route]
                        for route in (FIND, QUESTION, SYNTHESIS)}
        data["liveCompleted"] = sum(row["completed"] for row in live)
        live_syntheses = [row for row in live if row["expectedRoute"] == SYNTHESIS]
        data["liveCitedMeetings"] = coverage(
            [dict(row, candidateIds=row["citedIds"]) for row in live_syntheses], meetings)
        sources_ok = []
        for row in live:
            if row["actualRoute"] == FIND:
                continue
            claims = emitted_claims(row["answer"])
            for claim in claims:
                for source in claim["sources"]:
                    document = docs[source["documentId"]]
                    fields = (re.sub(r"\r\n|[\n\v\f\r\x85\u2028\u2029]", " ", document[field])
                              for field in ("body", "summary"))
                    sources_ok.append(source["documentId"] in row["citedIds"]
                                      and any(source["quote"] in field for field in fields))
            audit.append({"workspaceId": workspace_id, "caseId": row["caseId"],
                          "question": cases[row["caseId"]]["question"], "claims": claims,
                          "expectedIds": row["expectedIds"], "citedIds": row["citedIds"]})
        data["citationIntegrity"] = {"matched": sum(sources_ok), "total": len(sources_ok)}
        summary[str(workspace_id)] = data
    write("phase5-citation-audit.json", {"cases": audit})
    write("phase5-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--summarize"]:
        summarize()
    else:
        prepare()
