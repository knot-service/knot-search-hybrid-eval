package com.knot.backend.search.infrastructure.persistence;

import com.knot.backend.search.application.SearchHybridQuestion;
import com.knot.backend.search.application.SearchToolScope;
import java.sql.Timestamp;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

@Repository
@RequiredArgsConstructor
public class SearchHybridDocumentScope {
    private static final Duration RECENTLY_ENDED_WINDOW = Duration.ofHours(1);

    private final JdbcClient jdbc;

    public List<EligibleDocument> find(
            SearchToolScope scope,
            SearchHybridQuestion question
    ) {
        return jdbc.sql("""
                SELECT d.id, r.id AS meeting_id, r.started_at, d.title, COALESCE(d.topic, '') AS topic
                FROM documents d JOIN recording_sessions r ON r.id = d.recording_session_id
                WHERE d.workspace_id = :workspace AND r.workspace_id = :workspace
                  AND d.status IN ('DRAFT', 'ARCHIVED') AND d.created_at <= :cutoff
                  AND r.started_at <= :cutoff
                  AND (NOT :recentlyEnded OR r.ended_at BETWEEN :recentCutoff AND :cutoff)
                  AND (CAST(:dates AS tstzmultirange) IS NULL OR r.started_at <@ CAST(:dates AS tstzmultirange))
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
                .param("recentlyEnded", question.recentlyEndedMeeting())
                .param("recentCutoff", Timestamp.from(scope.startedAt().minus(RECENTLY_ENDED_WINDOW)))
                .param(
                        "dates",
                        recordedDates(question)
                )
                .query(
                        (
                                row,
                                index
                        ) -> new EligibleDocument(
                                row.getLong("id"),
                                row.getLong("meeting_id"),
                                        row.getTimestamp("started_at").toInstant(),
                                        row.getString("title"),
                                        row.getString("topic")
                        )
                )
                .list();
    }

    private String recordedDates(SearchHybridQuestion question) {
        if (question.recordedDates().isEmpty()) {
            return null;
        }
        return question.recordedDates().stream()
                .map(date -> "[" + date.from() + "," + date.before() + ")")
                .collect(Collectors.joining(",", "{", "}"));
    }

    public record EligibleDocument(
            long id,
            long meetingId,
            Instant recordedAt,
            String title,
            String topic
    ) {
    }
}
