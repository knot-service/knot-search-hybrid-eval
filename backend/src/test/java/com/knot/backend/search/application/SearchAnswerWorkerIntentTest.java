package com.knot.backend.search.application;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import com.knot.backend.global.config.SearchAgentProperties;
import com.knot.backend.search.application.dto.result.SearchAnswerWorkResult;
import com.knot.backend.search.application.dto.result.SearchRetrievalResult;
import com.knot.backend.search.domain.model.SearchIntent;
import com.knot.backend.search.domain.SearchToolException;
import java.time.Clock;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.beans.factory.support.StaticListableBeanFactory;

class SearchAnswerWorkerIntentTest {

    private final SearchAnswerExecutionService executions = mock(SearchAnswerExecutionService.class);
    private final SearchDocumentRetriever retriever = mock(SearchDocumentRetriever.class);
    private final SearchAnswerGenerator generator = mock(SearchAnswerGenerator.class);
    private final SearchIntentClassifier classifier = mock(SearchIntentClassifier.class);
    private final UUID token = UUID.randomUUID();
    private SearchAnswerWorker worker;

    @BeforeEach
    void setUp() {
        StaticListableBeanFactory beans = new StaticListableBeanFactory();
        beans.addBean(
                "retriever",
                retriever
        );
        beans.addBean(
                "generator",
                generator
        );
        beans.addBean(
                "classifier",
                classifier
        );
        worker = new SearchAnswerWorker(
                executions,
                beans.getBeanProvider(SearchDocumentRetriever.class),
                beans.getBeanProvider(SearchAnswerGenerator.class),
                Clock.systemUTC(),
                new SearchAgentProperties(),
                beans.getBeanProvider(SearchIntentClassifier.class),
                beans.getBeanProvider(SearchHybridWorkflow.class)
        );
        when(executions.claim(5)).thenReturn(
                Optional.of(
                        new SearchAnswerWorkResult(
                                1,
                                2,
                                3,
                                4,
                                5,
                                token,
                                "질문",
                                Clock.systemUTC()
                                        .instant()
                        )
                )
        );
    }

    @Test
    @DisplayName("분류가 끝나기 전 검색·생성·본문 저장이 없다")
    void execute_success_noDeltaBeforeClassification() throws Exception {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(
                classifier.classify(
                        any(),
                        eq("질문")
                )
        ).thenAnswer(call -> {
            entered.countDown();
            assertThat(
                    release.await(
                            5,
                            TimeUnit.SECONDS
                    )
            ).isTrue();
            return SearchIntent.OUT_OF_SCOPE;
        });
        when(
                executions.recordRetrieval(
                        5,
                        token,
                        0,
                        List.of()
                )
        ).thenReturn(true);
        when(
                executions.append(
                        eq(5L),
                        eq(token),
                        any()
                )
        ).thenReturn(true);
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<?> task = executor.submit(() -> worker.execute(5));
            try {
                assertThat(
                        entered.await(
                                5,
                                TimeUnit.SECONDS
                        )
                ).isTrue();
                verify(
                        executions,
                        never()
                ).append(
                        anyLong(),
                        any(),
                        any()
                );
                verifyNoInteractions(
                        retriever,
                        generator
                );
            } finally {
                release.countDown();
            }
            task.get(
                    5,
                    TimeUnit.SECONDS
            );
        }
        verify(executions).complete(
                5,
                token,
                List.of(),
                Set.of()
        );
    }

    @ParameterizedTest
    @EnumSource(value = SearchIntent.class, names = "MEETING_SEARCH", mode = EnumSource.Mode.EXCLUDE)
    @DisplayName("비검색 의도는 서버 고정 안내와 빈 근거로 완료한다")
    void execute_success_fixedGuidance(SearchIntent intent) {
        when(
                classifier.classify(
                        any(),
                        any()
                )
        ).thenReturn(intent);
        when(
                executions.recordRetrieval(
                        5,
                        token,
                        0,
                        List.of()
                )
        ).thenReturn(true);
        when(
                executions.append(
                        eq(5L),
                        eq(token),
                        any()
                )
        ).thenReturn(true);
        worker.execute(5);
        verify(executions).append(
                5,
                token,
                SearchIntentGuidance.forIntent(intent)
        );
        verify(executions).complete(
                5,
                token,
                List.of(),
                Set.of()
        );
        verifyNoInteractions(
                retriever,
                generator
        );
    }

    @Test
    @DisplayName("검색 의도만 기존 문서 검색으로 연결한다")
    void execute_success_meetingSearch() {
        when(
                classifier.classify(
                        any(),
                        any()
                )
        ).thenReturn(SearchIntent.MEETING_SEARCH);
        when(
                retriever.retrieve(
                        any(),
                        any()
                )
        ).thenReturn(
                new SearchRetrievalResult(
                        List.of(),
                        0
                )
        );
        when(
                executions.recordRetrieval(
                        5,
                        token,
                        0,
                        List.of()
                )
        ).thenReturn(true);
        when(
                executions.append(
                        eq(5L),
                        eq(token),
                        any()
                )
        ).thenReturn(true);
        worker.execute(5);
        verify(retriever).retrieve(
                any(),
                eq("질문")
        );
        verify(executions).complete(
                5,
                token,
                List.of(),
                Set.of()
        );
    }

    @Test
    @DisplayName("분류 실패는 자유 답변과 검색 없이 FAILED 처리한다")
    void execute_failure_classifierError() {
        when(
                classifier.classify(
                        any(),
                        any()
                )
        ).thenThrow(new SearchToolException("INVALID_INTENT_RESPONSE"));
        worker.execute(5);
        verify(executions).fail(
                5,
                token,
                "ANSWER_GENERATION_FAILED"
        );
        verify(
                executions,
                never()
        ).append(
                anyLong(),
                any(),
                any()
        );
        verifyNoInteractions(
                retriever,
                generator
        );
    }

    @Test
    @DisplayName("분류 이후 중지된 작업에는 고정 안내도 저장하지 않는다")
    void execute_success_staleGuidanceIgnored() {
        when(
                classifier.classify(
                        any(),
                        any()
                )
        ).thenReturn(SearchIntent.OUT_OF_SCOPE);
        when(
                executions.recordRetrieval(
                        5,
                        token,
                        0,
                        List.of()
                )
        ).thenReturn(false);
        worker.execute(5);
        verify(
                executions,
                never()
        ).append(
                anyLong(),
                any(),
                any()
        );
        verify(
                executions,
                never()
        ).complete(
                anyLong(),
                any(),
                any(),
                any()
        );
        verifyNoInteractions(
                retriever,
                generator
        );
    }
}
