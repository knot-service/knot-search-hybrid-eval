package com.knot.backend.search.infrastructure.persistence;

import com.knot.backend.search.domain.model.SearchAnswerExecution;
import com.knot.backend.search.domain.model.SearchConversation;
import com.knot.backend.search.domain.model.SearchEvidence;
import com.knot.backend.search.domain.model.SearchMessage;
import com.knot.backend.search.domain.model.SearchMessageRole;
import com.knot.backend.search.domain.model.SearchMessageStatus;
import com.knot.backend.search.domain.repository.SearchRepository;
import com.knot.backend.search.domain.model.SearchRequestReceipt;
import jakarta.persistence.EntityManager;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Repository;

@Repository
@RequiredArgsConstructor
public class SearchRepositoryAdapter implements SearchRepository {

    private final SearchConversationWriteJpaRepository conversations;
    private final SearchMessageWriteJpaRepository messages;
    private final SearchRequestReceiptJpaRepository receipts;
    private final SearchAnswerExecutionJpaRepository executions;
    private final SearchEvidenceWriteJpaRepository evidences;
    private final EntityManager entityManager;

    public SearchConversation save(SearchConversation conversation) {
        return conversations.save(conversation);
    }

    public SearchMessage save(SearchMessage message) {
        return messages.save(message);
    }

    public SearchRequestReceipt save(SearchRequestReceipt receipt) {
        return receipts.save(receipt);
    }

    public SearchAnswerExecution save(SearchAnswerExecution execution) {
        return executions.save(execution);
    }

    public SearchEvidence save(SearchEvidence evidence) {
        return evidences.save(evidence);
    }

    public Optional<SearchRequestReceipt> findFirstReceipt(
            long workspaceId,
            long memberId,
            UUID requestId
    ) {
        return receipts.findFirstReceipt(
                workspaceId,
                memberId,
                requestId
        );
    }

    public Optional<SearchConversation> findConversation(long id) {
        return conversations.findById(id);
    }

    public Optional<SearchRequestReceipt> findRetryReceipt(
            long workspaceId,
            long memberId,
            UUID requestId
    ) {
        return receipts.findRetryReceipt(
                workspaceId,
                memberId,
                requestId
        );
    }

    public Optional<SearchRequestReceipt> findLatestReceipt(long answerId) {
        return receipts.findFirstByAnswerMessageIdOrderByIdDesc(answerId);
    }

    public Optional<SearchMessage> findMessageBySequence(
            long conversationId,
            int sequence
    ) {
        return messages.findByConversationIdAndSequence(
                conversationId,
                sequence
        );
    }

    public void deleteEvidences(long answerId) {
        evidences.deleteByIdMessageId(answerId);
    }

    public Optional<SearchRequestReceipt> findQuestionReceipt(
            long workspaceId,
            long memberId,
            long conversationId,
            UUID requestId
    ) {
        return receipts.findQuestionReceipt(
                workspaceId,
                memberId,
                conversationId,
                requestId
        );
    }

    public boolean existsStreamingAnswer(long conversationId) {
        return messages.existsByConversationIdAndRoleAndStatus(
                conversationId,
                SearchMessageRole.ASSISTANT,
                SearchMessageStatus.STREAMING
        );
    }

    public Optional<SearchMessage> findLastMessage(long conversationId) {
        return messages.findFirstByConversationIdOrderBySequenceDesc(conversationId);
    }

    public Optional<SearchConversation> findConversationForUpdate(long id) {
        Optional<SearchConversation> locked = conversations.findForUpdate(id);
        locked.ifPresent(entityManager::refresh);
        return locked;
    }

    public Optional<SearchMessage> findMessage(long id) {
        return messages.findById(id);
    }

    public Optional<SearchAnswerExecution> findExecution(long id) {
        return executions.findById(id);
    }

    public Optional<SearchAnswerExecution> findExecutionForUpdate(long id) {
        Optional<SearchAnswerExecution> locked = executions.findForUpdate(id);
        // 잠금 대기 전에 관리된 객체가 있으면 DB의 최신 선점 상태를 다시 읽는다.
        locked.ifPresent(entityManager::refresh);
        return locked;
    }

    public List<Long> findQueuedAnswerIds(int limit) {
        return executions.findQueued(
                PageRequest.of(
                        0,
                        limit
                )
        );
    }

    public List<Long> findQueuedAnswerIds(boolean lookupOnly, int limit) {
        return executions.findQueuedByLane(lookupOnly, PageRequest.of(0, limit));
    }

    public List<Long> findExpiredAnswerIds(
            Instant now,
            int limit
    ) {
        return executions.findExpired(
                now,
                PageRequest.of(
                        0,
                        limit
                )
        );
    }

    public void flush() {
        conversations.flush();
    }
}
