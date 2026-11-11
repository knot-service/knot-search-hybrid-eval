package com.knot.backend.search.application;

import com.knot.backend.global.config.SearchWorkerProperties;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
class SearchLookupLane {

    private final SearchWorkerProperties workerProperties;
    private final ObjectProvider<SearchHybridWorkflow> hybridWorkflows;

    boolean usesLookupWorker(long workspaceId, String question) {
        if (workerProperties.getLookupConcurrency() == 0 || !SearchHybridQuestion.isDocumentFind(question)) {
            return false;
        }
        SearchHybridWorkflow hybrid = hybridWorkflows.getIfAvailable();
        return hybrid != null && hybrid.enabledFor(workspaceId);
    }
}
