package com.knot.backend.search.application;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mockingDetails;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.knot.backend.auth.domain.AuthTokenProvider;
import com.knot.backend.auth.domain.model.AuthenticatedMember;
import com.knot.backend.document.DocumentFixtures;
import com.knot.backend.global.config.SearchWorkerProperties;
import com.knot.backend.search.application.SearchHybridQuestion.Route;
import com.knot.backend.search.application.dto.command.SearchQuestionCommand;
import com.knot.backend.search.application.dto.result.SearchAnswerWorkResult;
import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import com.knot.backend.search.infrastructure.llm.SearchLlmClient;
import com.knot.backend.search.infrastructure.llm.SearchHybridIndexClient.Passage;
import com.knot.backend.testsupport.TestApplicationProperties;
import com.knot.backend.testsupport.TestcontainersConfiguration;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.sql.Timestamp;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIfEnvironmentVariable;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Import;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean;
import org.springframework.transaction.support.TransactionTemplate;
import tools.jackson.databind.ObjectMapper;

@Tag("integration")
@ActiveProfiles("dev")
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {
        "knot.search.worker.enabled=true", "knot.search.worker.poll-interval=100ms", "knot.search.agent.enabled=true",
        "knot.search.agent.signing-key=phase4-test-signing-key-32-characters", "knot.llm.enabled=true",
        "knot.search.hybrid.workspaces=1,458"})
@Import(TestcontainersConfiguration.class)
@TestApplicationProperties
@EnabledIfEnvironmentVariable(named = "KNOT_PHASE5_FIXTURES", matches = ".+")
class SearchHybridDatasetReplayTest {
    @Autowired
    private SearchWorkerProperties workerProperties;
    @Autowired
    private JdbcClient jdbc;
    @Autowired
    private ObjectMapper mapper;
    @Autowired
    private AuthTokenProvider tokens;
    @Autowired
    private SearchQuestionIntakeService intake;
    @Autowired
    private SearchAnswerExecutionService executions;
    @Autowired
    private Clock clock;
    @Autowired
    private TransactionTemplate transaction;
    @MockitoSpyBean
    private SearchHybridWorkflow hybrid;
    @MockitoSpyBean
    private SearchLlmClient llm;
    @LocalServerPort
    private int port;

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry properties) {
        properties.add(
                "knot.search.hybrid.index-url",
                () -> System.getenv("KNOT_PHASE5_INDEX_URL")
        );
    }

    @Test
    @DisabledIfEnvironmentVariable(named = "KNOT_PHASE6_LOAD", matches = "true")
    void replaysEvidenceAndVerifiesRealWorkerSse() throws Exception {
        Dataset dataset = mapper.readValue(
                Path.of(System.getenv("KNOT_PHASE5_FIXTURES"))
                        .toFile(),
                Dataset.class
        );
        DocumentFixtures fixtures = new DocumentFixtures(jdbc);
        Map<Long, Long> members = new LinkedHashMap<>();
        for (Workspace workspace : dataset.workspaces()) {
            members.put(
                    workspace.id(),
                    seed(
                            fixtures,
                            workspace
                    )
            );
        }

        Map<Long, SearchHybridWorkflow.Result> observed = new ConcurrentHashMap<>();
        Map<Long, ModelObservation> modelObservations = observeModel();
        doAnswer(call -> {
            SearchToolScope scope = call.getArgument(0);
            SearchHybridWorkflow.Result result = (SearchHybridWorkflow.Result) call.callRealMethod();
            observed.put(
                    scope.answerId(),
                    result
            );
            return result;
        }).when(hybrid)
                .retrieve(
                        any(),
                        any()
                );

        List<Observation> replay = new ArrayList<>();
        List<SseObservation> sse = new ArrayList<>();
        try (HttpClient http = HttpClient.newHttpClient()) {
            for (Workspace workspace : dataset.workspaces()) {
                long member = members.get(workspace.id());
                replay.addAll(
                        replay(
                                workspace,
                                member
                        )
                );
                for (EvalCase item : workspace.cases()) {
                    if (item.live()) {
                        sse.add(
                                verifySse(
                                        http,
                                        workspace.id(),
                                        member,
                                        item,
                                        observed,
                                        modelObservations
                                )
                        );
                    }
                }
            }

            var unauthorized = http.send(
                    request(
                            1,
                            members.get(458L),
                            "회의 문서 찾아줘"
                    ),
                    HttpResponse.BodyHandlers.discarding()
            );
            assertThat(unauthorized.statusCode()).isEqualTo(403);
        }

        Path output = Path.of(System.getenv("KNOT_PHASE5_FIXTURES"))
                .resolveSibling("phase5-spring-report.json");
        Path staging = Files.createTempFile(
                output.getParent(),
                "phase5-",
                ".json",
                PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------"))
        );
        mapper.writerWithDefaultPrettyPrinter()
                .writeValue(
                        staging.toFile(),
                        new Report(
                                replay,
                                sse
                        )
                );
        Files.move(
                staging,
                output,
                java.nio.file.StandardCopyOption.REPLACE_EXISTING
        );
        long expectedLive = dataset.workspaces()
                .stream()
                .flatMap(
                        workspace -> workspace.cases()
                                .stream()
                )
                .filter(EvalCase::live)
                .count();
        assertThat(sse).hasSize((int) expectedLive);
        assertThat(expectedLive).isGreaterThanOrEqualTo(6);
        for (SseObservation row : sse) {
            assertThat(row.completed()).as(
                    "SSE completed %s",
                    row.caseId()
            )
                    .isTrue();
            assertThat(row.actualRoute()).as(
                    "route %s",
                    row.caseId()
            )
                    .isEqualTo(row.expectedRoute());
            int expectedCalls = 1;
            if (row.expectedRoute() == Route.DOCUMENT_FIND) {
                expectedCalls = 0;
            }
            assertThat(row.generationCalls()).as(
                    "generation calls %s",
                    row.caseId()
            )
                    .isEqualTo(expectedCalls);
            assertThat(row.candidateIds()).containsAll(row.citedIds());
            if (row.expectedRoute() == Route.DOCUMENT_FIND) {
                assertThat(row.citedIds()).as(
                        "document lookup %s",
                        row.caseId()
                )
                        .containsAll(row.expectedIds());
            }
        }
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "KNOT_PHASE6_LOAD", matches = "true")
    void measuresMixedConcurrentSse() throws Exception {
        var fixture = Path.of(System.getenv("KNOT_PHASE5_FIXTURES"));
        Dataset dataset = mapper.readValue(
                fixture.toFile(),
                Dataset.class
        );
        DocumentFixtures fixtures = new DocumentFixtures(jdbc);
        Map<Long, Long> members = new LinkedHashMap<>();
        for (Workspace workspace : dataset.workspaces()) {
            members.put(
                    workspace.id(),
                    seed(
                            fixtures,
                            workspace
                    )
            );
        }
        Map<Long, SearchHybridWorkflow.Result> observed = new ConcurrentHashMap<>();
        Map<Long, ModelObservation> modelObservations = observeModel();
        doAnswer(call -> {
            SearchToolScope scope = call.getArgument(0);
            SearchHybridWorkflow.Result result = (SearchHybridWorkflow.Result) call.callRealMethod();
            observed.put(
                    scope.answerId(),
                    result
            );
            return result;
        }).when(hybrid)
                .retrieve(
                        any(),
                        any()
                );

        int waves = Integer.parseInt(
                System.getenv()
                        .getOrDefault(
                                "KNOT_PHASE6_WAVES",
                                "12"
                        )
        );
        assertThat(waves).isBetween(
                1,
                100
        );
        String loadPattern = System.getenv().getOrDefault("KNOT_PHASE6_LOAD_PATTERN", "balanced");
        assertThat(loadPattern).isIn("balanced", "question_surge", "question_sustained");
        boolean surge = loadPattern.equals("question_surge");
        boolean sustained = loadPattern.equals("question_sustained");
        List<Long> sustainedDelays = List.of(0L, 0L, 0L, 0L, 250L, 250L, 1000L, 2200L, 4500L, 5800L);
        List<Integer> levels = sustained ? List.of(sustainedDelays.size()) : surge ? List.of(6) : List.of(3, 6);
        List<LoadSample> samples = new ArrayList<>();
        try (HttpClient http = HttpClient.newHttpClient()) {
            List<LoadQuestion> warmup = questions(
                    dataset,
                    6,
                    0
            );
            var warmed = SearchHybridLoadReplay.wave(
                    http,
                    mapper,
                    requests(
                            warmup,
                            members
                    )
            );
            assertThat(warmed).allMatch(sample -> sample.completedNanos() != null);
            clearInvocations(llm);
            observed.clear();
            for (int concurrency : levels) {
                for (int wave = 0; wave < waves; wave++) {
                    List<LoadQuestion> questions = sustained ? sustainedQuestions(dataset, wave)
                            : surge ? surgeQuestions(dataset, wave) : questions(dataset, concurrency, wave);
                    List<HttpRequest> batch = requests(questions, members);
                    var events = sustained
                            ? SearchHybridLoadReplay.wave(http, mapper, batch, sustainedDelays, workerProperties.getStreamConcurrency())
                            : surge
                            ? SearchHybridLoadReplay.wave(http, mapper, batch, List.of(0L, 0L, 300L, 300L, 800L, 1200L))
                            : SearchHybridLoadReplay.wave(http, mapper, batch);
                    for (int index = 0; index < events.size(); index++) {
                        samples.add(
                                new LoadSample(
                                        concurrency,
                                        wave,
                                        sustained ? index : null,
                                        questions.get(index),
                                        events.get(index)
                                )
                        );
                    }
                }
                System.out.printf(
                        "phase6 concurrency=%d measured=%d%n",
                        concurrency,
                        concurrency * waves
                );
            }
        }

        Map<Long, Integer> generations = new HashMap<>();
        Map<Long, Integer> classifications = new HashMap<>();
        for (var invocation : mockingDetails(llm).getInvocations()) {
            if (invocation.getArguments().length == 0
                    || !(invocation.getArgument(0) instanceof SearchToolScope scope)) {
                continue;
            }
            switch (invocation.getMethod()
                    .getName()) {
                case "stream", "streamStructuredAnswer" -> generations.merge(
                        scope.answerId(),
                        1,
                        Integer::sum
                );
                case "intent" -> classifications.merge(
                        scope.answerId(),
                        1,
                        Integer::sum
                );
                default -> {
                }
            }
        }
        List<LoadObservation> results = new ArrayList<>();
        for (LoadSample sample : samples) {
            var event = sample.event();
            var question = sample.question();
            var item = question.item();
            var result = observed.get(event.answerId());
            Route route = null;
            List<Long> candidates = List.of();
            if (result != null) {
                route = result.route();
                candidates = ids(result);
            }
            List<Long> cited = jdbc.sql("SELECT document_id FROM search_evidences WHERE message_id = :id ORDER BY rank")
                    .param(
                            "id",
                            event.answerId()
                    )
                    .query(Long.class)
                    .list();
            boolean lookupOnly = jdbc.sql("SELECT lookup_only FROM search_answer_executions WHERE answer_id = :id")
                    .param("id", event.answerId()).query(Boolean.class).optional().orElse(false);
            boolean questionOnly = jdbc.sql("SELECT question_only FROM search_answer_executions WHERE answer_id = :id")
                    .param("id", event.answerId()).query(Boolean.class).optional().orElse(false);
            Long acceptedAtMs = sustained ? jdbc.sql("SELECT created_at FROM search_answer_executions WHERE answer_id = :id")
                    .param("id", event.answerId()).query(Timestamp.class).single().toInstant().toEpochMilli() : null;
            results.add(
                    new LoadObservation(
                            sample.concurrency(),
                            sample.wave(),
                            sample.wavePosition(),
                            acceptedAtMs,
                            question.workspaceId(),
                            item.id(),
                            item.route(),
                            route,
                            event.answerId(),
                            event.httpStatus(),
                            event.terminal(),
                            lookupOnly,
                            questionOnly,
                            generations.getOrDefault(
                                    event.answerId(),
                                    0
                            ),
                            classifications.getOrDefault(
                                    event.answerId(),
                                    0
                            ),
                            item.expectedIds(),
                            candidates,
                            cited,
                            event.answer(),
                            event.firstNanos(),
                            event.completedNanos(),
                            modelObservations.remove(event.answerId())
                    )
            );
        }

        Path output = fixture.resolveSibling("phase6-spring-load-report.json");
        Path staging = Files.createTempFile(
                output.getParent(),
                "phase6-",
                ".json",
                PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------"))
        );
        mapper.writerWithDefaultPrettyPrinter()
                .writeValue(
                        staging.toFile(),
                        new LoadReport(
                                6,
                                waves,
                                workerProperties.getConcurrency(),
                                workerProperties.getLookupConcurrency(),
                                workerProperties.isSplitGenerationRoutes(),
                                workerProperties.isShareGenerationSlots(),
                                workerProperties.getSynthesisPromotionWait().toMillis(),
                                loadPattern,
                                sustained ? sustainedDelays : List.of(),
                                workerProperties.getStreamConcurrency(),
                                workerProperties.getPollInterval()
                                        .toMillis(),
                                results
                        )
                );
        Files.move(
                staging,
                output,
                java.nio.file.StandardCopyOption.REPLACE_EXISTING
        );
        assertThat(results).hasSize(levels.stream().mapToInt(Integer::intValue).sum() * waves);
        assertThat(results).allSatisfy(row -> {
            assertThat(row.questionOnly()).isEqualTo(workerProperties.isSplitGenerationRoutes()
                    && row.expectedRoute() == Route.MEETING_QUESTION);
            assertThat(row.actualRoute()).isEqualTo(row.expectedRoute());
            assertThat(row.terminal()).isEqualTo("completed");
        });
        assertThat(results).allSatisfy(row -> assertThat(row.lookupOnly())
                .isEqualTo(workerProperties.getLookupConcurrency() > 0 && row.expectedRoute() == Route.DOCUMENT_FIND));
        assertThat(
                results.stream()
                        .filter(row -> row.expectedRoute() == Route.DOCUMENT_FIND)
        ).allSatisfy(row -> {
            assertThat(row.actualRoute()).isEqualTo(Route.DOCUMENT_FIND);
            assertThat(row.generationCalls()).isZero();
            assertThat(row.classificationCalls()).isZero();
        });
    }

    private List<LoadQuestion> questions(
            Dataset dataset,
            int concurrency,
            int wave
    ) {
        List<LoadQuestion> questions = new ArrayList<>();
        List<Route> routes = List.of(
                Route.DOCUMENT_FIND,
                Route.MEETING_QUESTION,
                Route.MULTI_MEETING_SYNTHESIS
        );
        for (int index = 0; index < concurrency; index++) {
            int occurrence = wave * (concurrency / routes.size()) + index / routes.size();
            int routeIndex = index % routes.size();
            questions.add(loadQuestion(dataset, routes.get(routeIndex), occurrence, routeIndex));
        }
        return questions;
    }

    private List<LoadQuestion> surgeQuestions(Dataset dataset, int wave) {
        int offset = wave * 4;
        return List.of(
                loadQuestion(dataset, Route.MEETING_QUESTION, offset, 1),
                loadQuestion(dataset, Route.MEETING_QUESTION, offset + 1, 1),
                loadQuestion(dataset, Route.MULTI_MEETING_SYNTHESIS, wave, 2),
                loadQuestion(dataset, Route.DOCUMENT_FIND, wave, 0),
                loadQuestion(dataset, Route.MEETING_QUESTION, offset + 2, 1),
                loadQuestion(dataset, Route.MEETING_QUESTION, offset + 3, 1));
    }

    private List<LoadQuestion> sustainedQuestions(Dataset dataset, int wave) {
        List<LoadQuestion> questions = new ArrayList<>();
        int offset = wave * 2;
        for (int index = 0; index < 4; index++) {
            questions.add(loadQuestion(dataset, Route.MEETING_QUESTION, offset + index, 1));
        }
        questions.add(loadQuestion(dataset, Route.MULTI_MEETING_SYNTHESIS, wave, 2));
        questions.add(loadQuestion(dataset, Route.DOCUMENT_FIND, wave, 0));
        for (int index = 4; index < 8; index++) {
            questions.add(loadQuestion(dataset, Route.MEETING_QUESTION, offset + index, 1));
        }
        return questions;
    }

    private LoadQuestion loadQuestion(Dataset dataset, Route route, int occurrence, int routeOffset) {
        int workspaceCount = dataset.workspaces().size();
        Workspace workspace = dataset.workspaces().get((occurrence + routeOffset) % workspaceCount);
        List<EvalCase> cohort = workspace.cases().stream()
                .filter(item -> item.live() && item.route() == route).toList();
        return new LoadQuestion(workspace.id(), cohort.get((occurrence / workspaceCount) % cohort.size()));
    }

    private List<HttpRequest> requests(
            List<LoadQuestion> questions,
            Map<Long, Long> members
    ) {
        return questions.stream()
                .map(
                        question -> request(
                                question.workspaceId(),
                                members.get(question.workspaceId()),
                                question.item()
                                        .question()
                        )
                )
                .toList();
    }

    private record LoadQuestion(
            long workspaceId,
            EvalCase item
    ) {
    }

    private record LoadSample(
            int concurrency,
            int wave,
            Integer wavePosition,
            LoadQuestion question,
            SearchHybridLoadReplay.Sample event
    ) {
    }

    private record LoadObservation(
            int concurrency,
            int wave,
            Integer wavePosition,
            Long acceptedAtMs,
            long workspaceId,
            String caseId,
            Route expectedRoute,
            Route actualRoute,
            long answerId,
            int httpStatus,
            String terminal,
            boolean lookupOnly,
            boolean questionOnly,
            int generationCalls,
            int classificationCalls,
            List<Long> expectedIds,
            List<Long> candidateIds,
            List<Long> citedIds,
            String answer,
            Long firstNanos,
            Long completedNanos,
            ModelObservation model
    ) {
    }

    private record LoadReport(
            int warmupRequests,
            int wavesPerLevel,
            int workerConcurrency,
            int lookupConcurrency,
            boolean splitGenerationRoutes,
            boolean shareGenerationSlots,
            long synthesisPromotionWaitMs,
            String loadPattern,
            List<Long> plannedDispatchDelaysMs,
            int streamConcurrency,
            long pollIntervalMs,
            List<LoadObservation> requests
    ) {
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "KNOT_ACCURACY_REPLAY", matches = "true")
    void replaysRetrievalWithoutGeneration() throws Exception {
        Path fixture = Path.of(System.getenv("KNOT_PHASE5_FIXTURES"));
        Dataset dataset = mapper.readValue(fixture.toFile(), Dataset.class);
        List<Observation> rows = new ArrayList<>();
        DocumentFixtures documents = new DocumentFixtures(jdbc);
        for (Workspace workspace : dataset.workspaces()) {
            rows.addAll(replay(workspace, seed(documents, workspace)));
        }
        assertThat(mockingDetails(llm).getInvocations()).isEmpty();
        Path output = Files.createTempFile(fixture.getParent(), "retrieval-", ".json",
                PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
        mapper.writeValue(output.toFile(), rows);
        Files.move(output, fixture.resolveSibling("retrieval-report.json"),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    private List<Observation> replay(
            Workspace workspace,
            long member
    ) {
        List<Observation> rows = new ArrayList<>();
        // A claimed execution supplies the same authorization as a real Worker. Batch
        // by lease.
        for (int offset = 0; offset < workspace.cases()
                .size(); offset += 100) {
            SearchAnswerWorkResult work = transaction.execute(
                    status -> executions.claim(
                            intake.acceptFirst(
                                    workspace.id(),
                                    member,
                                    new SearchQuestionCommand(
                                            "평가 실행",
                                            UUID.randomUUID()
                                    )
                            )
                                    .answerMessageId()
                    )
                            .orElseThrow()
            );
            Instant started = clock.instant();
            SearchToolScope scope = SearchToolScope.from(
                    work,
                    started,
                    started.plusSeconds(120)
            );
            int until = Math.min(
                    offset + 100,
                    workspace.cases()
                            .size()
            );
            for (EvalCase item : workspace.cases()
                    .subList(
                            offset,
                            until
                    )) {
                SearchHybridWorkflow.Result result = hybrid.retrieve(
                        scope,
                        item.question()
                );
                rows.add(
                        new Observation(
                                workspace.id(),
                                workspace.version(),
                                item.id(),
                                item.route(),
                                result.route(),
                                item.expectedIds(),
                                ids(result),
                                result.questionNanos(),
                                result.retrievalNanos(),
                                result.passages(),
                                result.rerankMs(),
                                result.contextMs()
                        )
                );
            }
            executions.append(
                    work.answerId(),
                    work.token(),
                    "평가 완료"
            );
            executions.complete(
                    work.answerId(),
                    work.token(),
                    List.of(),
                    Set.of()
            );
        }
        return rows;
    }

    private Map<Long, ModelObservation> observeModel() {
        Map<Long, ModelObservation> observations = new ConcurrentHashMap<>();
        if (!"true".equals(System.getenv("KNOT_QUALITY_CAPTURE_MODEL"))) {
            return observations;
        }
        doAnswer(call -> {
            SearchToolScope scope = call.getArgument(0);
            List<Map<String, Object>> messages = call.getArgument(1);
            Consumer<String> downstream = call.getArgument(2);
            StringBuilder raw = new StringBuilder();
            call.getRawArguments()[2] = (Consumer<String>) delta -> {
                raw.append(delta);
                downstream.accept(delta);
            };
            call.callRealMethod();
            observations.put(scope.answerId(), new ModelObservation(messages, raw.toString()));
            return null;
        }).when(llm).streamStructuredAnswer(any(), any(), any());
        return observations;
    }

    private SseObservation verifySse(
            HttpClient http,
            long workspace,
            long member,
            EvalCase item,
            Map<Long, SearchHybridWorkflow.Result> observed,
            Map<Long, ModelObservation> modelObservations
    ) throws Exception {
        clearInvocations(llm);
        long started = System.nanoTime();
        var response = http.send(
                request(
                        workspace,
                        member,
                        item.question()
                ),
                HttpResponse.BodyHandlers.ofLines()
        );
        assertThat(response.statusCode()).isEqualTo(200);
        long answerId = 0;
        long firstDeltaNanos = 0;
        boolean completed = false;
        StringBuilder answer = new StringBuilder();
        String event = "";
        try (var lines = response.body()) {
            var stream = lines.iterator();
            while (stream.hasNext()) {
                String line = stream.next();
                if (line.startsWith("event:")) {
                    event = line.substring(6)
                            .strip();
                }
                if (!line.startsWith("data:")) {
                    continue;
                }
                var data = mapper.readTree(line.substring(5));
                if (data.has("answerMessageId")) {
                    answerId = data.path("answerMessageId")
                            .asLong();
                }
                if (event.equals("delta")) {
                    String delta = data.path("text")
                            .asText();
                    answer.append(delta);
                    if (firstDeltaNanos == 0 && !delta.isBlank()) {
                        firstDeltaNanos = System.nanoTime() - started;
                    }
                }
                completed |= event.equals("completed");
            }
        }
        verify(
                llm,
                times(0)
        ).intent(
                any(),
                any()
        );
        long calls = mockingDetails(llm).getInvocations()
                .stream()
                .filter(
                        invocation -> Set.of(
                                "stream",
                                "streamStructuredAnswer"
                        )
                                .contains(
                                        invocation.getMethod()
                                                .getName()
                                )
                )
                .count();
        SearchHybridWorkflow.Result result = observed.get(answerId);
        assertThat(result).as(
                "observed Worker result %s",
                item.id()
        )
                .isNotNull();
        List<Long> cited = jdbc.sql("SELECT document_id FROM search_evidences WHERE message_id = :id ORDER BY rank")
                .param(
                        "id",
                        answerId
                )
                .query(Long.class)
                .list();
        return new SseObservation(
                workspace,
                item.id(),
                item.route(),
                result.route(),
                completed,
                calls,
                item.expectedIds(),
                ids(result),
                cited,
                answer.toString(),
                firstDeltaNanos,
                System.nanoTime() - started,
                modelObservations.remove(answerId)
        );
    }

    private HttpRequest request(
            long workspace,
            long member,
            String question
    ) {
        String csrf = UUID.randomUUID()
                .toString();
        return HttpRequest.newBuilder(
                URI.create("http://127.0.0.1:" + port + "/api/v1/workspaces/" + workspace + "/search/conversations")
        )
                .timeout(Duration.ofSeconds(145))
                .header(
                        "Cookie",
                        "KNOT_ACCESS_TOKEN=" + tokens.issue(
                                AuthenticatedMember.of(
                                        member,
                                        "평가",
                                        null
                                )
                        ) + "; XSRF-TOKEN=" + csrf
                )
                .header(
                        "X-XSRF-TOKEN",
                        csrf
                )
                .header(
                        "Content-Type",
                        "application/json"
                )
                .header(
                        "Accept",
                        "text/event-stream"
                )
                .POST(
                        HttpRequest.BodyPublishers.ofString(
                                mapper.writeValueAsString(
                                        Map.of(
                                                "content",
                                                question,
                                                "requestId",
                                                UUID.randomUUID()
                                                        .toString()
                                        )
                                )
                        )
                )
                .build();
    }

    private long seed(
            DocumentFixtures fixtures,
            Workspace workspace
    ) {
        long member = fixtures.saveMember("phase4-" + workspace.id());
        jdbc.sql("INSERT INTO workspaces (id, name, created_at) VALUES (:id, :name, now())")
                .param(
                        "id",
                        workspace.id()
                )
                .param(
                        "name",
                        "phase4-" + workspace.id()
                )
                .update();
        fixtures.join(
                workspace.id(),
                member
        );
        Map<Long, Meeting> meetings = new HashMap<>();
        for (Document doc : workspace.documents()) {
            Meeting meeting = meetings.computeIfAbsent(
                    doc.meetingId(),
                    ignored -> {
                        long recording = fixtures.saveRecording(
                                workspace.id(),
                                member,
                                1000
                        );
                        long transcript = fixtures.saveTranscript(recording);
                        jdbc.sql(
                                "UPDATE recording_sessions SET started_at = :time, ended_at = :time, last_seen_at = :time WHERE id = :id"
                        )
                                .param(
                                        "time",
                                        Timestamp.from(doc.recordedAt())
                                )
                                .param(
                                        "id",
                                        recording
                                )
                                .update();
                        return new Meeting(
                                recording,
                                transcript
                        );
                    }
            );
            long job = fixtures.saveJob(
                    meeting.transcript(),
                    "SUCCEEDED"
            );
            jdbc.sql(
                    """
                            INSERT INTO documents (id, workspace_id, recording_session_id, source_transcript_id,
                                document_generation_job_id, topic, title, summary, content, status, created_at)
                            OVERRIDING SYSTEM VALUE
                            VALUES (:id, :workspace, :recording, :transcript, :job, :topic, :title, :summary, :body, 'DRAFT', :time)
                            """
            )
                    .param(
                            "id",
                            doc.id()
                    )
                    .param(
                            "workspace",
                            workspace.id()
                    )
                    .param(
                            "recording",
                            meeting.recording()
                    )
                    .param(
                            "transcript",
                            meeting.transcript()
                    )
                    .param(
                            "job",
                            job
                    )
                    .param(
                            "topic",
                            doc.topic()
                    )
                    .param(
                            "title",
                            doc.title()
                    )
                    .param(
                            "summary",
                            doc.summary()
                    )
                    .param(
                            "body",
                            doc.body()
                    )
                    .param(
                            "time",
                            Timestamp.from(doc.recordedAt())
                    )
                    .update();
        }
        return member;
    }

    private List<Long> ids(SearchHybridWorkflow.Result result) {
        return result.documents()
                .stream()
                .map(SearchDocumentResult::documentId)
                .toList();
    }

    private record Dataset(List<Workspace> workspaces) {
    }
    private record Workspace(
            long id,
            String version,
            List<Document> documents,
            List<EvalCase> cases
    ) {
    }
    private record Document(
            long id,
            long meetingId,
            Instant recordedAt,
            String title,
            String topic,
            String summary,
            String body
    ) {
    }
    private record Meeting(
            long recording,
            long transcript
    ) {
    }
    private record EvalCase(
            String id,
            String question,
            Route route,
            List<Long> expectedIds,
            boolean live
    ) {
    }
    private record Observation(
            long workspaceId,
            String version,
            String caseId,
            Route expectedRoute,
            Route actualRoute,
            List<Long> expectedIds,
            List<Long> candidateIds,
            long questionNanos,
            long retrievalNanos,
            List<Passage> passages,
            double rerankMs,
            double contextMs
    ) {
    }
    private record SseObservation(
            long workspaceId,
            String caseId,
            Route expectedRoute,
            Route actualRoute,
            boolean completed,
            long generationCalls,
            List<Long> expectedIds,
            List<Long> candidateIds,
            List<Long> citedIds,
            String answer,
            long firstDeltaNanos,
            long totalNanos,
            ModelObservation model
    ) {
    }
    private record ModelObservation(
            List<Map<String, Object>> messages,
            String raw
    ) {
    }
    private record Report(
            List<Observation> replay,
            List<SseObservation> sse
    ) {
    }
}
