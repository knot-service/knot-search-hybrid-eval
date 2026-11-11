package com.knot.backend.search.infrastructure;

import static org.assertj.core.api.Assertions.assertThat;
import ch.qos.logback.classic.Level;
import com.knot.backend.testsupport.LogCapture;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

import com.knot.backend.global.config.SearchWorkerProperties;
import com.knot.backend.search.application.SearchAnswerExecutionService;
import com.knot.backend.search.application.SearchAnswerWorker;
import com.knot.backend.search.domain.model.SearchExecutionLane;
import com.knot.backend.search.domain.repository.SearchGenerationCandidate;
import com.knot.backend.search.domain.repository.SearchRepository;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.task.TaskRejectedException;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

class SearchAnswerSchedulerTest {
    private static final Instant NOW = Instant.parse("2026-10-11T00:00:00Z");

    private SearchRepository repository;
    private SearchAnswerWorker worker;
    private SearchAnswerExecutionService executions;
    private ThreadPoolTaskExecutor executor;
    private SearchAnswerScheduler scheduler;
    private ThreadPoolTaskExecutor lookupExecutor;
    private ThreadPoolTaskExecutor synthesisExecutor;
    private SearchWorkerProperties properties;

    @BeforeEach
    void setUp() {
        repository = mock(SearchRepository.class);
        worker = mock(SearchAnswerWorker.class);
        executions = mock(SearchAnswerExecutionService.class);
        executor = mock(ThreadPoolTaskExecutor.class);
        lookupExecutor = mock(ThreadPoolTaskExecutor.class);
        synthesisExecutor = mock(ThreadPoolTaskExecutor.class);
        properties = new SearchWorkerProperties();
        scheduler = new SearchAnswerScheduler(
                repository,
                worker,
                executions,
                properties,
                executor,
                lookupExecutor,
                synthesisExecutor,
                Clock.fixed(
                        NOW,
                        ZoneOffset.UTC
                )
        );
    }

    @Test
    @DisplayName("영속 대기 작업을 worker executor에 전달한다")
    void dispatchQueued_success() {
        when(repository.findQueuedAnswerIds(20)).thenReturn(
                List.of(
                        1L,
                        2L
                )
        );
        doAnswer(call -> {
            Runnable task = call.getArgument(0);
            task.run();
            return null;
        }).when(executor)
                .execute(any());
        scheduler.dispatchQueued();
        verify(worker).execute(1L);
        verify(worker).execute(2L);
    }

    @Test
    @DisplayName("executor 용량 부족은 접수 작업을 실패로 바꾸지 않고 다음 poll로 남긴다")
    void dispatchQueued_success_capacityDeferred() {
        when(repository.findQueuedAnswerIds(20)).thenReturn(
                List.of(
                        1L,
                        2L
                )
        );
        doThrow(new TaskRejectedException("full")).when(executor)
                .execute(any());
        scheduler.dispatchQueued();
        verify(
                executor,
                times(1)
        ).execute(any());
        verifyNoInteractions(
                worker,
                executions
        );
    }

    @Test
    void sharedPoolAcceptsBothRoutesWithoutUsingFixedExecutor() {
        properties.setLookupConcurrency(1);
        properties.setSplitGenerationRoutes(true);
        properties.setShareGenerationSlots(true);
        when(repository.findQueuedGenerationByPriority(20)).thenReturn(
                List.of(new SearchGenerationCandidate(2L, true), new SearchGenerationCandidate(3L, true)),
                List.of(new SearchGenerationCandidate(4L, false), new SearchGenerationCandidate(5L, false)));
        doAnswer(call -> {
            Runnable task = call.getArgument(0);
            task.run();
            return null;
        }).when(executor).execute(any());

        scheduler.dispatchQueued();
        scheduler.dispatchQueued();

        verify(worker).execute(2L, SearchExecutionLane.QUESTION);
        verify(worker).execute(3L, SearchExecutionLane.QUESTION);
        verify(worker).execute(4L, SearchExecutionLane.DEFAULT);
        verify(worker).execute(5L, SearchExecutionLane.DEFAULT);
        verifyNoInteractions(synthesisExecutor);
    }

    @Test
    @DisplayName("생성 슬롯이 가득 차도 문서 찾기 큐를 처리한다")
    void dispatchLookupWhenGenerationIsSaturated() {
        properties.setLookupConcurrency(1);
        when(repository.findQueuedAnswerIds(true, 20)).thenReturn(List.of(2L));
        when(repository.findQueuedAnswerIds(false, 20)).thenReturn(List.of(1L));
        doThrow(new TaskRejectedException("full")).when(executor).execute(any());
        doAnswer(call -> {
            Runnable task = call.getArgument(0);
            task.run();
            return null;
        }).when(lookupExecutor).execute(any());

        scheduler.dispatchQueued();

        verify(worker).execute(2L);
        verify(executor).execute(any());
        verifyNoInteractions(executions);
    }

    @Test
    void dispatchQuestionWhenSynthesisIsSaturated() {
        properties.setLookupConcurrency(1);
        properties.setSplitGenerationRoutes(true);
        when(repository.findQueuedGenerationAnswerIds(true, 20)).thenReturn(List.of(2L));
        when(repository.findQueuedGenerationAnswerIds(false, 20)).thenReturn(List.of(1L));
        doThrow(new TaskRejectedException("full")).when(synthesisExecutor).execute(any());
        doAnswer(call -> {
            Runnable task = call.getArgument(0);
            task.run();
            return null;
        }).when(executor).execute(any());

        scheduler.dispatchQueued();

        verify(worker).execute(2L, SearchExecutionLane.QUESTION);
        verify(synthesisExecutor).execute(any());
        verifyNoInteractions(executions);
    }

    @Test
    @DisplayName("worker 내부 오류는 다른 후보의 전달을 막지 않는다")
    void dispatchQueued_success_isolatesWorkerFailure() {
        when(repository.findQueuedAnswerIds(20)).thenReturn(
                List.of(
                        1L,
                        2L
                )
        );
        doAnswer(call -> {
            Runnable task = call.getArgument(0);
            task.run();
            return null;
        }).when(executor)
                .execute(any());
        doThrow(new IllegalStateException("DB unavailable")).when(worker)
                .execute(1);
        scheduler.dispatchQueued();
        verify(worker).execute(2);
    }

    @Test
    @DisplayName("기한 지난 실행을 복구하며 한 건의 DB 오류는 다음 실행의 복구를 막지 않는다")
    void recoverExpired_success() {
        when(
                repository.findExpiredAnswerIds(
                        eq(NOW),
                        eq(20)
                )
        ).thenReturn(
                List.of(
                        1L,
                        2L
                )
        );
        doThrow(new IllegalStateException("DB unavailable")).when(executions)
                .expire(1);
        scheduler.recoverExpired();
        verify(executions).expire(2);
        verifyNoInteractions(worker);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @DisplayName("탐색 후보 조회 오류는 한 번 기록하고 다음 poll을 계속한다")
    void poll_failure_queryThenNextRun(boolean expired) {
        IllegalStateException failure = new IllegalStateException("private-sql");
        if (expired) {
            when(
                    repository.findExpiredAnswerIds(
                            NOW,
                            20
                    )
            ).thenThrow(failure)
                    .thenReturn(List.of(1L));
        }
        if (!expired) {
            when(repository.findQueuedAnswerIds(20)).thenThrow(failure)
                    .thenReturn(List.of(1L));
        }
        try (LogCapture logs = new LogCapture(SearchAnswerScheduler.class)) {
            poll(expired);
            poll(expired);
            assertThat(logs.events()).singleElement()
                    .satisfies(event -> {
                        assertThat(event.getLevel()).isEqualTo(Level.ERROR);
                        assertThat(event.getFormattedMessage()).contains("phase=")
                                .doesNotContain("private-sql");
                        assertThat(event.getThrowableProxy()).isNull();
                    });
        }
        if (expired) {
            verify(executions).expire(1L);
            return;
        }
        verify(executor).execute(any());
    }

    private void poll(boolean expired) {
        if (expired) {
            scheduler.recoverExpired();
            return;
        }
        scheduler.dispatchQueued();
    }

}
