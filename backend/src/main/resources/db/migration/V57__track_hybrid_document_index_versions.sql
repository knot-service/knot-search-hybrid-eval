CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE hybrid_document_index_jobs (
    workspace_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    source_version BIGINT NOT NULL DEFAULT 1,
    indexed_version BIGINT NOT NULL DEFAULT 0,
    retry_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    failures INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (workspace_id, document_id),
    CHECK (indexed_version >= 0 AND indexed_version <= source_version)
);

CREATE INDEX idx_hybrid_index_pending
    ON hybrid_document_index_jobs (workspace_id, retry_at, document_id)
    WHERE source_version > indexed_version;

CREATE TABLE hybrid_document_index_chunks (
    workspace_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    field TEXT NOT NULL CHECK (field IN ('body', 'summary')),
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
    embedding vector(384) NOT NULL,
    PRIMARY KEY (workspace_id, document_id, field, start_offset),
    FOREIGN KEY (workspace_id, document_id)
        REFERENCES hybrid_document_index_jobs (workspace_id, document_id) ON DELETE CASCADE
);

CREATE FUNCTION queue_hybrid_document_version() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    target_workspace BIGINT;
    target_document BIGINT;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF (OLD.workspace_id, OLD.id, OLD.recording_session_id, OLD.title, OLD.topic, OLD.summary, OLD.content, OLD.status)
           IS NOT DISTINCT FROM
           (NEW.workspace_id, NEW.id, NEW.recording_session_id, NEW.title, NEW.topic, NEW.summary, NEW.content, NEW.status) THEN
            RETURN NULL;
        END IF;
        IF (OLD.workspace_id, OLD.id) IS DISTINCT FROM (NEW.workspace_id, NEW.id) THEN
            INSERT INTO hybrid_document_index_jobs (workspace_id, document_id)
            VALUES (OLD.workspace_id, OLD.id)
            ON CONFLICT (workspace_id, document_id) DO UPDATE
            SET source_version = hybrid_document_index_jobs.source_version + 1, retry_at = now(), failures = 0;
        END IF;
    END IF;

    IF TG_OP = 'DELETE' THEN
        target_workspace := OLD.workspace_id;
        target_document := OLD.id;
    ELSE
        target_workspace := NEW.workspace_id;
        target_document := NEW.id;
    END IF;

    INSERT INTO hybrid_document_index_jobs (workspace_id, document_id)
    VALUES (target_workspace, target_document)
    ON CONFLICT (workspace_id, document_id) DO UPDATE
    SET source_version = hybrid_document_index_jobs.source_version + 1, retry_at = now(), failures = 0;
    RETURN NULL;
END;
$$;

CREATE TRIGGER queue_hybrid_document_version
AFTER INSERT OR UPDATE OF workspace_id, id, recording_session_id, title, topic, summary, content, status OR DELETE
ON documents FOR EACH ROW EXECUTE FUNCTION queue_hybrid_document_version();

INSERT INTO hybrid_document_index_jobs (workspace_id, document_id)
SELECT workspace_id, id FROM documents
ON CONFLICT (workspace_id, document_id) DO NOTHING;
