package com.knot.backend.search.application;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.List;
import java.util.function.Consumer;

public interface SearchAnswerGenerator {

    List<Long> generate(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            Consumer<String> onDelta
    );

    List<Long> generateGrounded(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            Consumer<String> onDelta
    );
}
