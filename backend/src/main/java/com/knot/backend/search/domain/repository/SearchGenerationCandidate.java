package com.knot.backend.search.domain.repository;

import com.knot.backend.search.domain.model.SearchExecutionLane;

public record SearchGenerationCandidate(long answerId, boolean questionOnly) {

    public SearchExecutionLane lane() {
        return questionOnly ? SearchExecutionLane.QUESTION : SearchExecutionLane.DEFAULT;
    }
}
