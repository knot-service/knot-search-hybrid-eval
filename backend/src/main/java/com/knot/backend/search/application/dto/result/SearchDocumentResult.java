package com.knot.backend.search.application.dto.result;

import java.util.List;
import java.util.Set;

public record SearchDocumentResult(
        long documentId,
        String title,
        String topic,
        String content,
        List<String> excerpts,
        Long meetingId,
        Set<Integer> summaryExcerptIndexes
) {
    public SearchDocumentResult {
        excerpts = List.copyOf(excerpts);
        summaryExcerptIndexes = Set.copyOf(summaryExcerptIndexes);
    }

    public SearchDocumentResult(
            long documentId,
            String title,
            String topic,
            String content,
            List<String> excerpts,
            Long meetingId
    ) {
        this(documentId, title, topic, content, excerpts, meetingId, Set.of());
    }

    public SearchDocumentResult(
            long documentId,
            String title,
            String topic,
            String content,
            List<String> excerpts
    ) {
        this(documentId, title, topic, content, excerpts, null);
    }

    public SearchDocumentResult(
            long documentId,
            String title,
            String topic,
            String content
    ) {
        this(
                documentId,
                title,
                topic,
                content,
                List.of()
        );
    }
}
