"""Prepare local-only corpora and expected routes for the Spring Boot replay."""
import json
from hashlib import sha256
from pathlib import Path

from build_dataset import documents
from database import LOCAL, snapshot


def main() -> None:
    workspace1 = documents(1)
    draft = json.loads((LOCAL / "workspace1-evaluation-draft.json").read_bytes())
    source_hash = sha256(json.dumps([doc.model_dump() for doc in workspace1],
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if draft["sourceHash"] != source_hash:
        raise SystemExit("Workspace 1 snapshot does not match the evaluation dataset.")
    synthetic = json.loads(snapshot())
    meetings = {doc.id: doc.meetingId for doc in workspace1}
    cases = []
    for case in draft["cases"]:
        expected = case["expectedEvidenceIds"]
        if case["history"] or case["expectedType"] != "answer" or not expected:
            continue
        route = "MEETING_QUESTION"
        if case["category"] == "문서 찾기":
            route = "DOCUMENT_FIND"
        elif len({meetings[doc_id] for doc_id in expected}) > 1:
            route = "MULTI_MEETING_SYNTHESIS"
        cases.append({"id": case["id"], "question": case["question"], "route": route, "expectedIds": expected})
    for case in synthetic["cases"]:
        scenario = case["id"].split("-", 1)[0]
        route = "MEETING_QUESTION"
        if scenario == "S53":
            route = "DOCUMENT_FIND"
        elif scenario in {"S18", "S21", "S39", "S71"}:
            route = "MULTI_MEETING_SYNTHESIS"
        case["route"] = route
    payload = {
        "workspaces": [
            {"id": 1, "version": draft["version"], "documents": [doc.model_dump() for doc in workspace1], "cases": cases},
            {"id": 458, "version": synthetic["datasetVersion"], "documents": synthetic["documents"],
             "cases": synthetic["cases"]},
        ],
    }
    path = LOCAL / "phase4-fixtures.json"
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8", opener=private_file) as output:
        json.dump(payload, output, ensure_ascii=False)
    temporary.replace(path)
    print(f"Workspace 1: {len(cases)}, Workspace 458: {len(synthetic['cases'])}; {path}")


def private_file(path: str | Path, flags: int) -> int:
    from os import open as open_descriptor
    return open_descriptor(path, flags, 0o600)


if __name__ == "__main__":
    main()
