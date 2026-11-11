package com.knot.backend.search.application.dto.result;

import java.util.List;

public record SearchDocumentResult(
        long documentId,
        String title,
        String topic,
        String content,
        List<String> excerpts,
        Long meetingId
) {
    public SearchDocumentResult {
        excerpts = List.copyOf(excerpts);
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
