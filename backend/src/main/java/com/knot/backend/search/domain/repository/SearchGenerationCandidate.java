package com.knot.backend.search.domain.repository;

import com.knot.backend.search.domain.model.SearchExecutionLane;

public record SearchGenerationCandidate(long answerId, boolean questionOnly, boolean overdueSynthesis) {

    public SearchGenerationCandidate(long answerId, boolean questionOnly) {
        this(answerId, questionOnly, false);
    }

    public SearchExecutionLane lane() {
        return questionOnly ? SearchExecutionLane.QUESTION : SearchExecutionLane.DEFAULT;
    }
}
