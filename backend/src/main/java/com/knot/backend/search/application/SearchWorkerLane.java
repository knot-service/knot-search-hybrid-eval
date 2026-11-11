package com.knot.backend.search.application;

import com.knot.backend.global.config.SearchWorkerProperties;
import com.knot.backend.search.domain.model.SearchExecutionLane;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
class SearchWorkerLane {

    private final SearchWorkerProperties workerProperties;
    private final ObjectProvider<SearchHybridWorkflow> hybridWorkflows;

    SearchExecutionLane classify(long workspaceId, String question) {
        if (workerProperties.getLookupConcurrency() == 0) {
            return SearchExecutionLane.DEFAULT;
        }
        SearchHybridWorkflow hybrid = hybridWorkflows.getIfAvailable();
        if (hybrid == null || !hybrid.enabledFor(workspaceId)) {
            return SearchExecutionLane.DEFAULT;
        }
        return switch (SearchHybridQuestion.classifyRoute(question)) {
            case DOCUMENT_FIND -> SearchExecutionLane.LOOKUP;
            case MEETING_QUESTION -> workerProperties.isSplitGenerationRoutes()
                    ? SearchExecutionLane.QUESTION : SearchExecutionLane.DEFAULT;
            case MULTI_MEETING_SYNTHESIS -> workerProperties.isSplitGenerationRoutes()
                    ? SearchExecutionLane.SYNTHESIS : SearchExecutionLane.DEFAULT;
        };
    }
}
