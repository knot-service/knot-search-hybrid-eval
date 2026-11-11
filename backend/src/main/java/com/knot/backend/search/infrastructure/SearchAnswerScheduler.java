package com.knot.backend.search.infrastructure;

import com.knot.backend.global.logging.ServerErrorLogger;
import com.knot.backend.global.config.SearchWorkerProperties;
import com.knot.backend.search.application.SearchAnswerExecutionService;
import com.knot.backend.search.application.SearchAnswerWorker;
import com.knot.backend.search.domain.model.SearchExecutionLane;
import com.knot.backend.search.domain.repository.SearchRepository;
import java.time.Clock;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.task.TaskRejectedException;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;

@Component
@EnableScheduling
@Slf4j
@ConditionalOnProperty(prefix = "knot.search.worker", name = "enabled", havingValue = "true")
public class SearchAnswerScheduler {

    private final SearchRepository repository;
    private final SearchAnswerWorker worker;
    private final SearchAnswerExecutionService executions;
    private final SearchWorkerProperties properties;
    private final ThreadPoolTaskExecutor executor;
    private final ThreadPoolTaskExecutor lookupExecutor;
    private final ThreadPoolTaskExecutor synthesisExecutor;
    private final Clock clock;

    public SearchAnswerScheduler(
            SearchRepository repository,
            SearchAnswerWorker worker,
            SearchAnswerExecutionService executions,
            SearchWorkerProperties properties,
            @Qualifier("searchGenerationExecutor") ThreadPoolTaskExecutor executor,
            @Qualifier("searchLookupExecutor") ThreadPoolTaskExecutor lookupExecutor,
            @Qualifier("searchSynthesisExecutor") ThreadPoolTaskExecutor synthesisExecutor,
            Clock clock
    ) {
        this.repository = repository;
        this.worker = worker;
        this.executions = executions;
        this.properties = properties;
        this.executor = executor;
        this.lookupExecutor = lookupExecutor;
        this.synthesisExecutor = synthesisExecutor;
        this.clock = clock;
    }

    @Scheduled(fixedDelayString = "${knot.search.worker.poll-interval:1s}")
    public void dispatchQueued() {
        try {
            dispatchQueuedCandidates();
        } catch (RuntimeException exception) {
            ServerErrorLogger.record(
                    log,
                    "SearchAnswerScheduler phase=dispatchQueued-candidates",
                    exception
            );
        }
    }

    private void dispatchQueuedCandidates() {
        int limit = properties.getCandidateLimit();
        if (properties.getLookupConcurrency() == 0) {
            dispatch(executor, repository.findQueuedAnswerIds(limit));
            return;
        }
        if (properties.isSplitGenerationRoutes()) {
            dispatch(lookupExecutor, repository.findQueuedAnswerIds(true, limit), SearchExecutionLane.LOOKUP);
            dispatch(executor, repository.findQueuedGenerationAnswerIds(true, limit), SearchExecutionLane.QUESTION);
            dispatch(synthesisExecutor, repository.findQueuedGenerationAnswerIds(false, limit), SearchExecutionLane.DEFAULT);
        } else {
            dispatch(lookupExecutor, repository.findQueuedAnswerIds(true, limit));
            dispatch(executor, repository.findQueuedAnswerIds(false, limit));
        }
    }

    private void dispatch(ThreadPoolTaskExecutor target, List<Long> candidates) {
        dispatch(target, candidates, null);
    }

    private void dispatch(ThreadPoolTaskExecutor target, List<Long> candidates, SearchExecutionLane expectedLane) {
        for (long answerId : candidates) {
            try {
                target.execute(() -> execute(answerId, expectedLane));
            } catch (TaskRejectedException exception) {
                return;
            }
        }
    }

    @Scheduled(fixedDelayString = "${knot.search.worker.recovery-interval:5s}")
    public void recoverExpired() {
        try {
            recoverExpiredCandidates();
        } catch (RuntimeException exception) {
            ServerErrorLogger.record(
                    log,
                    "SearchAnswerScheduler phase=recoverExpired-candidates",
                    exception
            );
        }
    }

    private void recoverExpiredCandidates() {
        for (long answerId : repository.findExpiredAnswerIds(
                clock.instant(),
                properties.getCandidateLimit()
        )) {
            try {
                executions.expire(answerId);
            } catch (RuntimeException exception) {
                ServerErrorLogger.record(
                        log,
                        "Search recovery deferred answerId=" + answerId + " phase=recovery",
                        exception
                );
            }
        }
    }

    private void execute(long answerId, SearchExecutionLane expectedLane) {
        try {
            if (expectedLane == null) {
                worker.execute(answerId);
            } else {
                worker.execute(answerId, expectedLane);
            }
        } catch (RuntimeException exception) {
            ServerErrorLogger.record(
                    log,
                    "Search execution deferred answerId=" + answerId + " phase=execute",
                    exception
            );
        }
    }
}
