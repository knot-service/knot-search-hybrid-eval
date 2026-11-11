package com.knot.backend.search.domain.model;

import com.knot.backend.search.domain.SearchErrorCode;
import com.knot.backend.search.domain.SearchException;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Duration;
import java.time.Instant;
import java.util.UUID;
import lombok.Getter;

@Getter
@Entity
@Table(name = "search_answer_executions")
public class SearchAnswerExecution {
    @Id
    @Column(name = "answer_id", updatable = false)
    private Long answerId;

    @Column(name = "conversation_id", nullable = false, updatable = false)
    private long conversationId;

    @Column(name = "question_id", nullable = false, updatable = false)
    private long questionId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private SearchExecutionState state;

    @Column(name = "execution_token")
    private UUID executionToken;

    @Column(name = "deadline_at")
    private Instant deadlineAt;

    @Enumerated(EnumType.STRING)
    @Column(length = 20)
    private SearchProgressStage stage;

    @Column(name = "excluded_count")
    private Integer excludedCount;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "lookup_only", nullable = false)
    private boolean lookupOnly;

    @Column(name = "question_only", nullable = false)
    private boolean questionOnly;

    @Column(name = "synthesis_queued_at")
    private Instant synthesisQueuedAt;

    protected SearchAnswerExecution() {}

    public static SearchAnswerExecution queue(
            long conversationId,
            long questionId,
            long answerId,
            Instant createdAt
    ) {
        return queue(conversationId, questionId, answerId, createdAt, SearchExecutionLane.DEFAULT);
    }

    public static SearchAnswerExecution queue(
            long conversationId,
            long questionId,
            long answerId,
            Instant createdAt,
            SearchExecutionLane lane
    ) {
        if (conversationId <= 0 || questionId <= 0 || answerId <= 0 || questionId == answerId) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        if (createdAt == null) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        SearchAnswerExecution execution = new SearchAnswerExecution();
        execution.conversationId = conversationId;
        execution.questionId = questionId;
        execution.answerId = answerId;
        execution.state = SearchExecutionState.QUEUED;
        execution.createdAt = createdAt;
        execution.assignLane(lane, createdAt);
        return execution;
    }

    public boolean claim(
            UUID token,
            Instant now,
            Duration lease
    ) {
        if (state != SearchExecutionState.QUEUED) {
            return false;
        }
        if (token == null || now == null || lease == null || lease.isNegative() || lease.isZero()) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        executionToken = token;
        deadlineAt = now.plus(lease);
        stage = SearchProgressStage.SEARCHING;
        state = SearchExecutionState.RUNNING;
        return true;
    }

    public boolean isCurrent(
            UUID token,
            Instant now
    ) {
        return state == SearchExecutionState.RUNNING && executionToken.equals(token) && now.isBefore(deadlineAt);
    }

    public boolean isExpired(Instant now) {
        return state == SearchExecutionState.RUNNING && !now.isBefore(deadlineAt);
    }

    public void recordRetrieval(int count) {
        if (state != SearchExecutionState.RUNNING || count < 0) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        excludedCount = count;
        stage = SearchProgressStage.GENERATING;
    }

    public void stop() {
        if (state == SearchExecutionState.FINISHED) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        state = SearchExecutionState.FINISHED;
    }

    public void retry() {
        retry(synthesisQueuedAt == null ? lane() : SearchExecutionLane.SYNTHESIS, createdAt);
    }

    public SearchExecutionLane lane() {
        if (lookupOnly) {
            return SearchExecutionLane.LOOKUP;
        }
        return questionOnly ? SearchExecutionLane.QUESTION : SearchExecutionLane.DEFAULT;
    }

    public void retry(SearchExecutionLane lane, Instant queuedAt) {
        if (state != SearchExecutionState.FINISHED) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        executionToken = null;
        deadlineAt = null;
        stage = null;
        excludedCount = null;
        assignLane(lane, queuedAt);
        state = SearchExecutionState.QUEUED;
    }

    private void assignLane(SearchExecutionLane lane, Instant queuedAt) {
        lookupOnly = lane == SearchExecutionLane.LOOKUP;
        questionOnly = lane == SearchExecutionLane.QUESTION;
        synthesisQueuedAt = lane == SearchExecutionLane.SYNTHESIS ? queuedAt : null;
    }

    public void finish() {
        if (state != SearchExecutionState.RUNNING) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_STATE);
        }
        state = SearchExecutionState.FINISHED;
    }
}
