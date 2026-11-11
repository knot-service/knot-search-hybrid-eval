package com.knot.backend.search.application.dto.result;

import java.util.List;

public record SearchDocumentResult(
        long documentId,
        String title,
        String topic,
        String content,
        List<String> excerpts
) {
    public SearchDocumentResult {
        excerpts = List.copyOf(excerpts);
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
