package com.knot.backend.search.infrastructure.persistence;

import com.knot.backend.search.application.SearchHybridQuestion;
import com.knot.backend.search.application.SearchToolScope;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

@Repository
@RequiredArgsConstructor
public class SearchHybridDocumentScope {

    private final JdbcClient jdbc;

    public List<EligibleDocument> find(
            SearchToolScope scope,
            SearchHybridQuestion question
    ) {
        return jdbc.sql("""
                SELECT d.id, r.id AS meeting_id, r.started_at
                FROM documents d JOIN recording_sessions r ON r.id = d.recording_session_id
                WHERE d.workspace_id = :workspace AND r.workspace_id = :workspace
                  AND d.status IN ('DRAFT', 'ARCHIVED') AND d.created_at <= :cutoff
                  AND r.started_at <= :cutoff
                  AND (CAST(:from AS timestamptz) IS NULL OR r.started_at >= :from)
                  AND (CAST(:before AS timestamptz) IS NULL OR r.started_at < :before)
                ORDER BY r.started_at DESC, d.id DESC
                """)
                .param(
                        "workspace",
                        scope.workspaceId()
                )
                .param(
                        "cutoff",
                        Timestamp.from(scope.startedAt())
                )
                .param(
                        "from",
                        timestamp(question.recordedFrom())
                )
                .param(
                        "before",
                        timestamp(question.recordedBefore())
                )
                .query(
                        (
                                row,
                                index
                        ) -> new EligibleDocument(
                                row.getLong("id"),
                                row.getLong("meeting_id"),
                                row.getTimestamp("started_at")
                                        .toInstant()
                        )
                )
                .list();
    }

    private Timestamp timestamp(Instant value) {
        if (value == null) {
            return null;
        }
        return Timestamp.from(value);
    }

    public record EligibleDocument(
            long id,
            long meetingId,
            Instant recordedAt
    ) {
    }
}
