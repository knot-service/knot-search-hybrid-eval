"""Manage the independent local PostgreSQL corpus and pgvector index."""
import json
from pathlib import Path
import secrets
import sys

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict

LOCAL = Path(__file__).resolve().parent / ".local"
SCHEMA = """
create extension if not exists vector;
create table if not exists workspaces (
    id integer primary key,
    source text not null check (source in ('synthetic', 'production_snapshot')),
    snapshot jsonb not null
);
create table if not exists documents (
    workspace_id integer not null references workspaces(id) on delete cascade,
    id integer not null,
    meeting_id integer not null,
    recorded_at timestamptz not null,
    title text not null,
    topic text not null,
    summary text not null,
    body text not null,
    embedding vector(384),
    primary key (workspace_id, id)
);
"""


class ImportDocument(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    meetingId: int
    recordedAt: str
    title: str
    topic: str
    summary: str
    body: str


class ImportEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    workspaceId: int
    documents: tuple[ImportDocument, ...]


def prepare() -> None:
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    path = LOCAL / "postgres.env"
    try:
        with path.open("x") as file:
            path.chmod(0o600)
            file.write("POSTGRES_USER=knot_hybrid\nPOSTGRES_DB=knot_hybrid\n")
            file.write(f"POSTGRES_PASSWORD={secrets.token_urlsafe(32)}\n")
    except FileExistsError:
        pass


def connect() -> psycopg.Connection:
    settings = dict(line.split("=", 1) for line in (LOCAL / "postgres.env").read_text().splitlines())
    return psycopg.connect(
        host="127.0.0.1", port=55448, dbname="knot_hybrid", user="knot_hybrid",
        password=settings["POSTGRES_PASSWORD"], connect_timeout=3,
    )


def import_workspace(path: Path) -> None:
    corpus = ImportEnvelope.model_validate_json(path.read_bytes())
    if corpus.workspaceId not in (1, 458) or not corpus.documents:
        raise SystemExit("Expected a nonempty Workspace 1 or Workspace 458 snapshot.")
    scope = "synthetic" if corpus.workspaceId == 458 else "production_snapshot"
    metadata = corpus.model_dump(exclude={"documents"}, mode="json")
    with connect() as db, db.cursor() as cursor:
        cursor.execute(SCHEMA)
        cursor.execute(
            "insert into workspaces values (%s,%s,%s) on conflict (id) do update "
            "set source=excluded.source, snapshot=excluded.snapshot",
            (corpus.workspaceId, scope, Jsonb(metadata)),
        )
        cursor.execute("delete from documents where workspace_id=%s", (corpus.workspaceId,))
        cursor.executemany(
            "insert into documents (workspace_id,id,meeting_id,recorded_at,title,topic,summary,body) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s)",
            [(corpus.workspaceId, doc.id, doc.meetingId, doc.recordedAt, doc.title,
              doc.topic, doc.summary, doc.body) for doc in corpus.documents],
        )
    print(f"workspace={corpus.workspaceId} source={scope} documents={len(corpus.documents)}")


def snapshot() -> bytes:
    with connect() as db:
        row = db.execute(
            """select (w.snapshot || jsonb_build_object(
                'documents', jsonb_agg(jsonb_build_object(
                    'id', d.id, 'meetingId', d.meeting_id, 'recordedAt', d.recorded_at,
                    'title', d.title, 'topic', d.topic, 'summary', d.summary, 'body', d.body
                ) order by d.id)
            ))::text from workspaces w join documents d on d.workspace_id=w.id
            where w.id=458 and w.source='synthetic' group by w.id"""
        ).fetchone()
    if row is None:
        raise SystemExit("Import the Workspace 458 snapshot before starting the runner.")
    return str(row[0]).encode()


def vector_literal(values: list[float]) -> str:
    return json.dumps(values, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "prepare":
        prepare()
    else:
        import_workspace(Path(sys.argv[1]))
