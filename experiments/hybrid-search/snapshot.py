"""Export Workspace 458's synthetic corpus and retrieval-only gold cases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SQL = """
begin read only;
select json_build_object(
  'documents', coalesce(json_agg(json_build_object(
    'id', d.id, 'meetingId', r.id, 'recordedAt', r.started_at,
    'title', d.title, 'topic', coalesce(d.topic, ''),
    'summary', coalesce(d.summary, ''), 'body', coalesce(d.content, '')
  ) order by d.id), '[]'::json)
)
from documents d
join recording_sessions r on r.id = d.recording_session_id and r.workspace_id = d.workspace_id
join workspaces w on w.id = d.workspace_id and w.deleted_at is null
where d.workspace_id = 458 and d.status in ('DRAFT','ARCHIVED')
  and exists (
    select 1 from workspace_members wm join members m on m.id = wm.member_id
    where wm.workspace_id = d.workspace_id and wm.member_id = 1
      and wm.left_at is null and m.deleted_at is null
  );
commit;
"""


GOLD_SQL = """
begin read only;
select json_build_object(
  'version', d->>'version',
  'cases', (select json_agg(json_build_object(
      'id', c->>'id', 'question', c->>'question', 'expectedEvidenceIds', c->'expectedEvidenceIds'
    ) order by c->>'id') from jsonb_array_elements(d->'cases') as c
    where coalesce(c->'history', '[]'::jsonb) = '[]'::jsonb
      and nullif(c->>'executionFixture', '') is null)
)
from eval_state, jsonb_array_elements(data->'datasets') as d
where id=1 and d->>'id'='knot-workspace-458' and d->>'version'='1.30'
  and d->>'source'='synthetic' and d->>'locked'='true';
commit;
"""


def read_local_db(container: str, database: str, sql: str) -> bytes:
    return subprocess.check_output(
        ["docker", "exec", container, "psql", "-XqAt", "-v", "ON_ERROR_STOP=1",
         "-U", database, "-d", database, "-c", sql],
    )


def export(separate_path: Path) -> None:
    dataset = json.loads(read_local_db("knot-eval-isolated-postgres-1", "knot_eval", GOLD_SQL))
    documents = json.loads(read_local_db(
        "knot-search-playground-db", "knot_playground", SQL
    ))["documents"]
    if not documents:
        raise SystemExit("The active synthetic member cannot read any Workspace 458 documents.")
    separate = {case["caseId"] for case in json.loads(separate_path.read_bytes())["cases"]}
    available = {f'document-{doc["id"]}': doc["id"] for doc in documents}
    cases = [
        {"id": case["id"], "question": case["question"],
         "expectedIds": [available[doc] for doc in case["expectedEvidenceIds"]]}
        for case in dataset["cases"]
        if case["id"] not in separate and case["expectedEvidenceIds"]
        and all(doc in available for doc in case["expectedEvidenceIds"])
    ]
    snapshot = {"workspaceId": 458, "memberId": 1, "datasetVersion": dataset["version"],
                "documents": documents, "cases": cases}
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()
    path = ROOT / ".local" / "workspace458.json"
    path.parent.mkdir(mode=0o700, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded)
    temporary.chmod(0o600)
    temporary.replace(path)
    print(f"documents={len(documents)} cases={len(cases)} sha256={hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
