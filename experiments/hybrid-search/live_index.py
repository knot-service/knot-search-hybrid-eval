"""Maintain the live index from the document transaction's versioned work rows."""

from dataclasses import dataclass
from datetime import datetime
import logging
from threading import Event, Lock
from typing import TypedDict

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast

from build_dataset import SourceDoc
from database import vector_literal
from phase1 import Child, chunks

LOG = logging.getLogger(__name__)
BATCH_SIZE = 16


@dataclass(frozen=True, slots=True)
class IndexWork:
    workspace_id: int
    document_id: int
    version: int
    document: SourceDoc | None


@dataclass(frozen=True, slots=True)
class EncodedChild:
    child: Child
    embedding: str


class SourceRow(TypedDict):
    document_id: int
    meeting_id: int
    recorded_at: datetime
    title: str
    topic: str
    summary: str
    body: str


def source_document(row: SourceRow) -> SourceDoc:
    """Use the same source fields and offsets as the snapshot retriever."""
    return SourceDoc(id=row["document_id"], meetingId=row["meeting_id"], recordedAt=row["recorded_at"].isoformat(),
                     title=row["title"], topic=row["topic"], summary=row["summary"], body=row["body"])


class LiveIndex:
    def __init__(self, dsn: str, workspaces: tuple[int, ...]) -> None:
        self.dsn = dsn
        self.workspaces = workspaces

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, connect_timeout=3)

    def pending(self) -> tuple[IndexWork, ...]:
        """Read source data and its version in one MVCC snapshot, without locking writers during encoding."""
        with self.connect() as db, db.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute("""
                SELECT j.workspace_id, j.document_id, j.source_version, r.id AS meeting_id,
                       r.started_at AS recorded_at, d.title, d.topic, coalesce(d.summary, '') AS summary, d.content AS body
                FROM hybrid_document_index_jobs j
                LEFT JOIN documents d ON d.id = j.document_id AND d.workspace_id = j.workspace_id
                     AND d.status IN ('DRAFT', 'ARCHIVED')
                LEFT JOIN recording_sessions r ON r.id = d.recording_session_id AND r.workspace_id = j.workspace_id
                WHERE j.workspace_id = any(%s) AND j.source_version > j.indexed_version AND j.retry_at <= now()
                ORDER BY j.retry_at, j.workspace_id, j.document_id LIMIT %s
                """, (list(self.workspaces), BATCH_SIZE)).fetchall()
        return tuple(IndexWork(row["workspace_id"], row["document_id"], row["source_version"],
                               source_document(row) if row["meeting_id"] is not None else None) for row in rows)

    def apply(self, work: IndexWork, encoded: tuple[EncodedChild, ...]) -> bool:
        """Publish chunks only if this work is still current. The version and chunks commit together."""
        with self.connect() as db, db.cursor() as cursor:
            claimed = cursor.execute("""
                UPDATE hybrid_document_index_jobs SET indexed_version = %s, failures = 0
                WHERE workspace_id = %s AND document_id = %s AND source_version = %s AND indexed_version < %s
                RETURNING document_id
                """, (work.version, work.workspace_id, work.document_id, work.version, work.version)).fetchone()
            if claimed is None:
                return False
            cursor.execute("DELETE FROM hybrid_document_index_chunks WHERE workspace_id=%s AND document_id=%s",
                           (work.workspace_id, work.document_id))
            cursor.executemany("""
                INSERT INTO hybrid_document_index_chunks VALUES (%s,%s,%s,%s,%s,%s::vector)
                """, [(work.workspace_id, work.document_id, item.child.field, item.child.start,
                       item.child.end, item.embedding) for item in encoded])
        return True

    def retry(self, work: IndexWork) -> None:
        """Back off failed jobs without delaying a replacement version."""
        with self.connect() as db:
            db.execute("""
                UPDATE hybrid_document_index_jobs
                SET failures = least(failures + 1, 10),
                    retry_at = now() + make_interval(secs => least(60, power(2, failures)))
                WHERE workspace_id = %s AND document_id = %s AND source_version = %s AND indexed_version < %s
                """, (work.workspace_id, work.document_id, work.version, work.version))

    def run(self, stop: Event, model: SentenceTransformer, model_lock: Lock,
            tokenizer: PreTrainedTokenizerFast) -> None:
        """Poll durable work. Multiple workers can retry the same version safely."""
        while not stop.is_set():
            try:
                work = self.pending()
                for job in work:
                    if stop.is_set():
                        return
                    try:
                        self.apply(job, encode(job, model, model_lock, tokenizer))
                    except (psycopg.Error, RuntimeError, ValueError) as error:
                        LOG.warning("Hybrid index job failed workspace=%s document=%s version=%s type=%s",
                                    job.workspace_id, job.document_id, job.version, type(error).__name__)
                        self.retry(job)
                if not work:
                    stop.wait(0.5)
            except psycopg.Error as error:
                LOG.warning("Hybrid index database unavailable type=%s", type(error).__name__)
                stop.wait(2)


def encode(work: IndexWork, model: SentenceTransformer, model_lock: Lock,
           tokenizer: PreTrainedTokenizerFast) -> tuple[EncodedChild, ...]:
    """Encode one document or return an empty replacement for its deletion."""
    if work.document is None:
        return ()
    with model_lock:
        children = chunks(work.document, "fixed", tokenizer)
        if not children:
            return ()
        vectors = model.encode([f"passage: {child.text}" for child in children], batch_size=32,
                               normalize_embeddings=True, show_progress_bar=False)
    return tuple(EncodedChild(child, vector_literal(vector.tolist()))
                 for child, vector in zip(children, vectors, strict=True))


READY_DOCUMENTS = """
    FROM documents d
    JOIN hybrid_document_index_jobs j ON j.workspace_id=d.workspace_id AND j.document_id=d.id
    JOIN recording_sessions r ON r.id=d.recording_session_id AND r.workspace_id=d.workspace_id
    WHERE d.workspace_id=%s AND j.source_version=j.indexed_version AND d.status IN ('DRAFT','ARCHIVED')
"""


def source_versions(db: psycopg.Connection, workspace_id: int) -> tuple[tuple[int, int, datetime], ...]:
    return tuple(db.execute("SELECT d.id, j.indexed_version, r.started_at " + READY_DOCUMENTS +
                            " ORDER BY r.started_at,d.id", (workspace_id,)).fetchall())


def live_documents(db: psycopg.Connection, workspace_id: int) -> tuple[SourceDoc, ...]:
    with db.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute("""
            SELECT d.id AS document_id, r.id AS meeting_id, r.started_at AS recorded_at,
                   d.title, d.topic, coalesce(d.summary, '') AS summary, d.content AS body
            """ + READY_DOCUMENTS + " ORDER BY r.started_at,d.id", (workspace_id,)).fetchall()
    return tuple(source_document(row) for row in rows)
