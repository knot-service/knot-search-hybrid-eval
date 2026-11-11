package com.knot.backend.search.domain.repository;

import com.knot.backend.search.domain.model.SearchAnswerExecution;
import com.knot.backend.search.domain.model.SearchConversation;
import com.knot.backend.search.domain.model.SearchEvidence;
import com.knot.backend.search.domain.model.SearchMessage;
import com.knot.backend.search.domain.model.SearchRequestReceipt;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface SearchRepository {

    SearchConversation save(SearchConversation conversation);

    SearchMessage save(SearchMessage message);

    SearchRequestReceipt save(SearchRequestReceipt receipt);

    SearchAnswerExecution save(SearchAnswerExecution execution);

    SearchEvidence save(SearchEvidence evidence);

    Optional<SearchRequestReceipt> findFirstReceipt(
            long workspaceId,
            long memberId,
            UUID requestId
    );

    Optional<SearchConversation> findConversation(long conversationId);

    Optional<SearchRequestReceipt> findRetryReceipt(
            long workspaceId,
            long memberId,
            UUID requestId
    );

    Optional<SearchRequestReceipt> findLatestReceipt(long answerId);

    Optional<SearchMessage> findMessageBySequence(
            long conversationId,
            int sequence
    );

    void deleteEvidences(long answerId);

    Optional<SearchRequestReceipt> findQuestionReceipt(
            long workspaceId,
            long memberId,
            long conversationId,
            UUID requestId
    );

    boolean existsStreamingAnswer(long conversationId);

    Optional<SearchMessage> findLastMessage(long conversationId);

    Optional<SearchConversation> findConversationForUpdate(long conversationId);

    Optional<SearchMessage> findMessage(long messageId);

    Optional<SearchAnswerExecution> findExecution(long answerId);

    Optional<SearchAnswerExecution> findExecutionForUpdate(long answerId);

    List<Long> findQueuedAnswerIds(int limit);

    List<Long> findQueuedAnswerIds(boolean lookupOnly, int limit);

    List<Long> findQueuedGenerationAnswerIds(boolean questionOnly, int limit);

    List<SearchGenerationCandidate> findQueuedGenerationByPriority(int limit);

    List<Long> findExpiredAnswerIds(
            Instant now,
            int limit
    );

    void flush();
}
