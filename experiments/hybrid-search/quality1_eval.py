"""Compare the paired Spring Worker context and SSE replays. Publish counts only."""
import argparse
import json
from pathlib import Path

from database import LOCAL
from phase5_eval import FIND, SYNTHESIS, emitted_claims

# v1.30 identifies documents. These manually checked alternatives label a small
# fixed set of facts at the span level, using the synthetic snapshot.
SYNTHETIC_SPANS = {
    "S04-01": [(2145, ("로그인 오류 수정 작업의 담당자로 민지", "로그인 오류 수정 작업의 최종 담당자로 민지"))],
    "S16-01": [
        (2141, ("카드 결제를 먼저 지원하기로 했으며", "첫 번째로 지원할 결제 수단은 카드 결제다.")),
        (2141, ("11월 출시를 목표로 준비한다.", "11월을 출시 목표로 삼는다.")),
    ],
    "S18-01": [(2158, ("구체적인 최종 수치는 미정으로 남겨두었다.", "구체적인 로그인 유지 시간 수치는 이번 회의에서 확정되지 않았다.", "최종 적용할 구체적인 시간 값은 추가 분석 후 결정하기로 했다."))],
    "S21-01": [(2154, ("수동 배포를 자동 배포로 변경", "수동 배포에서 자동 배포로 변경", "수동으로 진행하던 배포 방식을 자동 배포로 변경"))],
    "S71-01": [
        (2207, ("JWT 토큰의 만료 시간도 동일하게 1시간", "JWT 토큰의 만료 시간(Expire Time)을 **1시간**", "JWT 토큰의 유효 기간을 1시간")),
        (2209, ("JWT 토큰의 만료 시간이 30분으로 명시", "'JWT 30분' 만료 시간 기준")),
    ],
}


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def rate(matched: int, total: int) -> dict:
    return {"matched": matched, "total": total}


def span_match(passages: list[dict], span: dict) -> bool:
    return any(
        p["documentId"] == span["documentId"] and p["field"] == span["field"]
        and p["start"] <= span["startOffset"] and p["end"] >= span["endOffset"]
        for p in passages
    )


def synthetic_match(passages: list[dict], document_id: int, alternatives: tuple[str, ...]) -> bool:
    return any(
        p["documentId"] == document_id and any(quote in p["quote"] for quote in alternatives)
        for p in passages
    )


def tracked(rows: list[dict]) -> dict:
    cases = [r for r in rows if r["caseId"].startswith("S18-01")]
    if not cases or any(not r["model"] for r in cases):
        raise ValueError("The tracked case requires KNOT_QUALITY_CAPTURE_MODEL=true.")
    inputs = [json.loads(row["model"]["messages"][1]["content"])["evidence"] for row in cases]
    return {
        "requests": len(cases),
        "inContext": sum(any(doc["documentId"] == 2158 for doc in evidence) for evidence in inputs),
        "modelCited": sum(any(s["evidenceId"] == "E2158" for c in json.loads(r["model"]["raw"])["claims"]
                              for s in c["sources"]) for r in cases),
        "verified": sum(2158 in r["citedIds"] for r in cases),
    }


def tracked_passages(report: dict) -> dict:
    row = next(row for row in report["replay"] if row["caseId"] == "S18-01")
    return {
        "total": sum(p["tokenCount"] for p in row["passages"]),
        "offTopic": sum(p["tokenCount"] for p in row["passages"] if p["documentId"] in (2153, 2178)),
    }


def metrics(workspace: dict, report: dict, draft: dict, review: dict) -> dict:
    workspace_id = workspace["id"]
    docs = {doc["id"]: doc for doc in workspace["documents"]}
    replay = {row["caseId"]: row for row in report["replay"] if row["workspaceId"] == workspace_id
              and "-repeat-" not in row["caseId"]}
    live = [row for row in report["sse"] if row["workspaceId"] == workspace_id]
    lookup = [row for row in live if row["expectedRoute"] == FIND]
    generated = [row for row in live if row["expectedRoute"] != FIND and "-repeat-" not in row["caseId"]]
    assert all(row["completed"] and row["generationCalls"] == 0 for row in lookup)
    assert all(row["completed"] and row["generationCalls"] == 1 for row in generated)

    def meetings(ids: list[int]) -> set[int]:
        return {docs[doc_id]["meetingId"] for doc_id in ids}

    synthesis = [row for row in replay.values() if row["expectedRoute"] == SYNTHESIS]
    span_cases = {case_id: draft[case_id]["evidence"] for case_id, row in replay.items()
                  if row["expectedRoute"] != FIND and row["actualRoute"] != FIND} if workspace_id == 1 else SYNTHETIC_SPANS
    if workspace_id == 1:
        spans = [span_match(replay[case_id]["passages"], span)
                 for case_id, spans in span_cases.items() for span in spans]
    else:
        spans = [synthetic_match(replay[case_id]["passages"], doc_id, alternatives)
                 for case_id, spans in span_cases.items() for doc_id, alternatives in spans]

    exact = []
    supported = []
    for row in generated:
        claims = emitted_claims(row["answer"])
        decisions = review[row["caseId"]]
        assert len(decisions) == len(claims) and all(type(decision) is bool for decision in decisions)
        supported.extend(decisions)
        for claim in claims:
            for citation in claim["sources"]:
                doc_id = citation["documentId"]
                exact.append(doc_id in row["citedIds"] and any(
                    citation["quote"] in " ".join(docs[doc_id][field].splitlines())
                    for field in ("body", "summary")
                ))
    return {
        "replayCases": len(replay), "spanCases": len(span_cases), "spans": rate(sum(spans), len(spans)),
        "allMeetingsInContext": rate(sum(meetings(r["expectedIds"]) <= meetings(r["candidateIds"]) for r in synthesis), len(synthesis)),
        "allExpectedCited": rate(sum(set(r["expectedIds"]) <= set(r["citedIds"]) for r in generated), len(generated)),
        "exactCitations": rate(sum(exact), len(exact)), "supportedClaims": rate(sum(supported), len(supported)),
        "lookup": {"requests": len(lookup), "generationCalls": sum(r["generationCalls"] for r in lookup)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("semantic_review", type=Path)
    parser.add_argument("--load-before", type=Path, required=True)
    parser.add_argument("--load-after", type=Path, required=True)
    args = parser.parse_args()
    fixture = args.before / "phase5-fixtures.json"
    if fixture.read_bytes() != (args.after / fixture.name).read_bytes():
        raise SystemExit("Before and after must use the same fixture.")
    dataset = read(fixture)
    draft = {case["id"]: case for case in read(LOCAL / "workspace1-evaluation-draft.json")["cases"]}
    review = read(args.semantic_review)
    reports = {name: read(path / "phase5-spring-report.json")
               for name, path in (("before", args.before), ("after", args.after))}
    load_reports = {name: read(path)["requests"] for name, path in
                    (("before", args.load_before), ("after", args.load_after))}
    for before, after in ((reports["before"]["replay"], reports["after"]["replay"]),
                          (reports["before"]["sse"], reports["after"]["sse"]),
                          (load_reports["before"], load_reports["after"])):
        keys = ("workspaceId", "caseId", "expectedRoute", "expectedIds")
        if [[row[key] for key in keys] for row in before] != [[row[key] for key in keys] for row in after]:
            raise SystemExit("Before and after report different cases.")
    print(json.dumps({
        "workspaces": [{"workspaceId": ws["id"], **{
            name: metrics(ws, report, draft, review[name]) for name, report in reports.items()
        }} for ws in dataset["workspaces"]],
        "s18": {
            "sequential": {name: tracked(report["sse"]) for name, report in reports.items()},
            "sustained": {name: tracked(rows) for name, rows in load_reports.items()},
            "passageTokens": {name: tracked_passages(report) for name, report in reports.items()},
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
