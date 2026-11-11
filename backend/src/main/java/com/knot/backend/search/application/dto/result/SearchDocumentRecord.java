package com.knot.backend.search.application.dto.result;

import com.knot.backend.document.domain.model.DocumentStatus;
import java.time.Instant;

public record SearchDocumentRecord(
        long documentId,
        String title,
        String topic,
        String content,
        String summary,
        DocumentStatus status,
        Instant recordingStartedAt,
        Instant createdAt,
        Instant archivedAt,
        int score
) {
}
