package com.knot.backend.search.application;

import com.knot.backend.global.logging.ServerErrorLogger;
import com.knot.backend.search.application.SearchHybridQuestion.Route;
import com.knot.backend.search.application.dto.result.SearchAnswerWorkResult;
import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import com.knot.backend.search.application.dto.result.SearchRetrievalResult;
import com.knot.backend.search.domain.model.SearchIntent;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import com.knot.backend.global.config.SearchAgentProperties;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.CancellationException;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

@Component
@RequiredArgsConstructor
@Slf4j
public class SearchAnswerWorker {
    private static final int DOCUMENT_LINK_LIMIT = 3;
    private static final String NO_DOCUMENT_GUIDANCE = "현재 검색 가능한 회의 문서에서는 해당 내용을 확인하지 못했습니다. 내용을 녹음하고 문서 준비가 끝난 뒤 다시 질문해 주세요.";

    private final SearchAnswerExecutionService executions;
    private final ObjectProvider<SearchDocumentRetriever> retrievers;
    private final ObjectProvider<SearchAnswerGenerator> generators;
    private final Clock clock;
    private final SearchAgentProperties agentProperties;
    private final ObjectProvider<SearchIntentClassifier> classifiers;
    private final ObjectProvider<SearchHybridWorkflow> hybridWorkflows;

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    public void execute(long answerId) {
        Instant workerStartedAt = clock.instant();
        Optional<SearchAnswerWorkResult> claimed = executions.claim(answerId);
        if (claimed.isEmpty()) {
            return;
        }
        SearchAnswerWorkResult work = claimed.orElseThrow();
        long queueWaitMs = Duration.between(
                work.createdAt(),
                workerStartedAt
        )
                .toMillis();
        log.info(
                "Search latency phase=queue_wait workspaceId={} answerId={} elapsedMs={} outcome=success",
                work.workspaceId(),
                answerId,
                Math.max(
                        0,
                        queueWaitMs
                )
        );
        AtomicLong persistenceCount = new AtomicLong();
        AtomicLong persistenceNanos = new AtomicLong();
        try {
            generate(
                    work,
                    persistenceCount,
                    persistenceNanos
            );
        } catch (RuntimeException exception) {
            ServerErrorLogger.recordTaskFailure(
                    log,
                    "Search generation failed workspaceId=" + work.workspaceId() + " answerId=" + answerId
                            + " conversationId=" + work.conversationId() + " phase=generation",
                    exception
            );
            executions.fail(
                    answerId,
                    work.token(),
                    "ANSWER_GENERATION_FAILED"
            );
        } finally {
            log.info(
                    "Search latency phase=content_persistence workspaceId={} answerId={} count={} elapsedMs={}",
                    work.workspaceId(),
                    answerId,
                    persistenceCount.get(),
                    persistenceNanos.get() / 1_000_000
            );
        }
    }

    private void generate(
            SearchAnswerWorkResult work,
            AtomicLong persistenceCount,
            AtomicLong persistenceNanos
    ) {
        Instant startedAt = clock.instant();
        SearchToolScope scope = SearchToolScope.from(
                work,
                startedAt,
                startedAt.plus(agentProperties.getBudget())
        );
        SearchHybridWorkflow hybrid = hybridWorkflows.getIfAvailable();
        boolean hybridEnabled = hybrid != null && hybrid.enabledFor(scope.workspaceId());
        if (work.lookupOnly() && (!hybridEnabled || !SearchHybridQuestion.isDocumentFind(work.question()))) {
            throw new IllegalStateException("Lookup-only execution cannot use the generation path");
        }
        if (hybridEnabled) {
            SearchHybridWorkflow.Result result = hybrid.retrieve(scope, work.question());
            if (work.lookupOnly() && result.route() != Route.DOCUMENT_FIND) {
                throw new IllegalStateException("Lookup-only execution returned a generation route");
            }
            log.info(
                    "Hybrid search workspaceId={} answerId={} route={} questionMs={} retrievalMs={} rerankMs={} contextMs={} candidates={}",
                    work.workspaceId(), work.answerId(), result.route(), result.questionNanos() / 1_000_000,
                    result.retrievalNanos() / 1_000_000, result.rerankMs(), result.contextMs(), result.documents().size()
            );
            deliver(work, scope, new SearchRetrievalResult(result.documents(), 0),
                    result.route(), true, persistenceCount, persistenceNanos);
            return;
        }
        long classificationStarted = System.nanoTime();
        SearchIntent intent;
        try {
            intent = classifiers.getObject()
                    .classify(
                            scope,
                            work.question()
                    );
            log.info(
                    "Search latency phase=intent_classification workspaceId={} answerId={} elapsedMs={} inputChars={} outcome=success intent={}",
                    work.workspaceId(),
                    work.answerId(),
                    (System.nanoTime() - classificationStarted) / 1_000_000,
                    work.question()
                            .length(),
                    intent
            );
        } catch (RuntimeException exception) {
            log.info(
                    "Search latency phase=intent_classification workspaceId={} answerId={} elapsedMs={} inputChars={} outcome=failure",
                    work.workspaceId(),
                    work.answerId(),
                    (System.nanoTime() - classificationStarted) / 1_000_000,
                    work.question()
                            .length()
            );
            throw exception;
        }
        if (!intent.requiresSearch()) {
            completeGuidance(
                    work,
                    intent,
                    persistenceCount,
                    persistenceNanos
            );
            return;
        }
        SearchDocumentRetriever retriever = retrievers.getObject();
        long retrievalStarted = System.nanoTime();
        SearchRetrievalResult retrieval;
        try {
            retrieval = retriever.retrieve(
                    scope,
                    work.question()
            );
            log.info(
                    "Search latency phase=document_retrieval workspaceId={} answerId={} elapsedMs={} inputChars={} candidates={} excludedDocuments={} outcome=success",
                    work.workspaceId(),
                    work.answerId(),
                    (System.nanoTime() - retrievalStarted) / 1_000_000,
                    work.question()
                            .length(),
                    retrieval.documents()
                            .size(),
                    retrieval.excludedCount()
            );
        } catch (RuntimeException exception) {
            log.info(
                    "Search latency phase=document_retrieval workspaceId={} answerId={} elapsedMs={} inputChars={} outcome=failure",
                    work.workspaceId(),
                    work.answerId(),
                    (System.nanoTime() - retrievalStarted) / 1_000_000,
                    work.question()
                            .length()
            );
            throw exception;
        }
        deliver(work, scope, retrieval, Route.MEETING_QUESTION, false, persistenceCount, persistenceNanos);
    }

    private void deliver(
            SearchAnswerWorkResult work,
            SearchToolScope scope,
            SearchRetrievalResult retrieval,
            Route route,
            boolean grounded,
            AtomicLong persistenceCount,
            AtomicLong persistenceNanos
    ) {
        if (!executions.recordRetrieval(
                work.answerId(),
                work.token(),
                retrieval.excludedCount(),
                retrieval.documents()
        )) {
            return;
        }
        List<SearchDocumentResult> documents = retrieval.documents();
        if (documents.isEmpty()) {
            append(
                    work,
                    NO_DOCUMENT_GUIDANCE,
                    persistenceCount,
                    persistenceNanos
            );
            executions.complete(
                    work.answerId(),
                    work.token(),
                    List.of(),
                    Set.of()
            );
            return;
        }
        if (route == Route.DOCUMENT_FIND) {
            List<SearchDocumentResult> linked = documents.stream().limit(DOCUMENT_LINK_LIMIT).toList();
            List<Long> ids = linked.stream().map(SearchDocumentResult::documentId).toList();
            String titles = linked.stream()
                    .map(document -> "- " + document.title().replaceAll("\\R", " "))
                    .collect(Collectors.joining("\n"));
            append(work, titles, persistenceCount, persistenceNanos);
            executions.complete(work.answerId(), work.token(), ids, Set.copyOf(ids));
            return;
        }
        SearchAnswerGenerator generator = generators.getObject();
        List<Long> usedIds;
        try (SearchAnswerContentBuffer content = new SearchAnswerContentBuffer(
                text -> append(
                        work,
                        text,
                        persistenceCount,
                        persistenceNanos
                )
        )) {
            long generationStarted = System.nanoTime();
            try {
                usedIds = generateAnswer(generator, grounded, scope, work.question(), documents, content);
                log.info(
                        "Search latency phase=answer_generation workspaceId={} answerId={} elapsedMs={} inputChars={} documentCount={} outcome=success",
                        work.workspaceId(),
                        work.answerId(),
                        (System.nanoTime() - generationStarted) / 1_000_000,
                        work.question()
                                .length(),
                        documents.size()
                );
            } catch (RuntimeException exception) {
                log.info(
                        "Search latency phase=answer_generation workspaceId={} answerId={} elapsedMs={} inputChars={} documentCount={} outcome=failure",
                        work.workspaceId(),
                        work.answerId(),
                        (System.nanoTime() - generationStarted) / 1_000_000,
                        work.question()
                                .length(),
                        documents.size()
                );
                throw exception;
            }
        }
        Set<Long> candidates = documents.stream()
                .map(SearchDocumentResult::documentId)
                .collect(Collectors.toSet());
        executions.complete(
                work.answerId(),
                work.token(),
                usedIds,
                candidates
        );
    }

    private List<Long> generateAnswer(
            SearchAnswerGenerator generator,
            boolean grounded,
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            SearchAnswerContentBuffer content
    ) {
        if (grounded) {
            return generator.generateGrounded(scope, question, documents, content);
        }
        return generator.generate(scope, question, documents, content);
    }

    private void completeGuidance(
            SearchAnswerWorkResult work,
            SearchIntent intent,
            AtomicLong persistenceCount,
            AtomicLong persistenceNanos
    ) {
        if (!executions.recordRetrieval(
                work.answerId(),
                work.token(),
                0,
                List.of()
        )) {
            return;
        }
        append(
                work,
                SearchIntentGuidance.forIntent(intent),
                persistenceCount,
                persistenceNanos
        );
        executions.complete(
                work.answerId(),
                work.token(),
                List.of(),
                Set.of()
        );
    }

    private void append(
            SearchAnswerWorkResult work,
            String text,
            AtomicLong persistenceCount,
            AtomicLong persistenceNanos
    ) {
        long persistenceStarted = System.nanoTime();
        try {
            if (!executions.append(
                    work.answerId(),
                    work.token(),
                    text
            )) {
                throw new CancellationException("answer execution ended");
            }
        } finally {
            persistenceCount.incrementAndGet();
            persistenceNanos.addAndGet(System.nanoTime() - persistenceStarted);
        }
    }
}
