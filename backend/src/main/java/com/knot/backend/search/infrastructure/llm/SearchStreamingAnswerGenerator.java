package com.knot.backend.search.infrastructure.llm;

import com.knot.backend.search.infrastructure.persistence.SearchHistoryQuery;

import com.knot.backend.global.config.SearchAgentProperties;
import com.knot.backend.search.application.SearchAnswerGenerator;
import com.knot.backend.search.application.SearchToolAccess;
import com.knot.backend.search.application.SearchToolScope;
import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import tools.jackson.databind.ObjectMapper;

@RequiredArgsConstructor
public class SearchStreamingAnswerGenerator implements SearchAnswerGenerator {

    private static final String MEETING_GROUPS = """
            evidence는 회의 ID로 묶여 있다. 같은 날이나 비슷한 시각이라도 회의 ID가 다르면 독립된 기록이다.
            각 주장에는 그 사실을 기록한 그룹의 문서와 인용구를 붙인다.
            비교의 핵심 차이가 두 회의에 걸쳐 있으면 첫 문장에서 그 차이를 설명하고 두 회의의 출처를 붙인다.
            회의별 설명에서는 결정·검토·미정 상태를 구분한다.
            질문이 지정한 회의가 확인되지 않으면 다른 그룹으로 대체하지 말고 insufficient로 표시한다.
            sources에는 그룹 ID 대신 해당 문서의 evidenceId를 쓴다.
            """;

    private final SearchLlmClient llm;
    private final SearchHistoryQuery history;
    private final SearchToolAccess access;
    private final SearchAgentProperties properties;
    private final ObjectMapper mapper;

    @Override
    public List<Long> generate(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            Consumer<String> onDelta
    ) {
        access.validate(scope);
        List<Map<String, Object>> messages = messages(
                scope,
                question,
                SearchAnswerPrompt.system(),
                evidence(documents, Map.of())
        );
        SearchAgentContext context = new SearchAgentContext(
                properties,
                mapper
        );
        context.validate(messages);
        SearchAnswerJsonStream answer = new SearchAnswerJsonStream(
                mapper,
                text -> {
                    access.validate(scope);
                    onDelta.accept(text);
                }
        );
        llm.stream(scope, messages, text -> {
            access.validate(scope);
            answer.accept(text);
        });
        access.validate(scope);
        return answer.complete(documents);
    }

    @Override
    public List<Long> generateGrounded(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            Consumer<String> onDelta
    ) {
        access.validate(scope);
        var scopedDocuments = SearchGroundedDocumentScope.selectDatedDiscussion(question, documents);
        boolean grouped = !scopedDocuments.isEmpty() && scopedDocuments.stream().allMatch(doc -> doc.meetingId() != null);
        var sourceIndex = SearchGroundedEvidence.index(scopedDocuments);
        var messages = messages(
                scope,
                question,
                SearchAnswerPrompt.grounded() + (grouped ? "\n" + MEETING_GROUPS : ""),
                grouped ? meetingEvidence(scopedDocuments, sourceIndex) : evidence(scopedDocuments, sourceIndex)
        );
        new SearchAgentContext(properties, mapper).validate(messages);
        StringBuilder response = new StringBuilder();
        llm.streamStructuredAnswer(scope, messages, response::append);
        access.validate(scope);
        SearchGroundedAnswer.Verified verified = SearchGroundedAnswer.verify(response.toString(), sourceIndex, mapper);
        onDelta.accept(verified.render());
        return verified.documentIds();
    }

    private List<Map<String, Object>> messages(
            SearchToolScope scope,
            String question,
            String systemPrompt,
            List<Map<String, Object>> evidence
    ) {
        return List.of(
                Map.of(
                        "role",
                        "system",
                        "content",
                        systemPrompt
                ),
                Map.of(
                        "role",
                        "user",
                        "content",
                        mapper.writeValueAsString(
                                Map.of(
                                        "question",
                                        question,
                                        "conversationHistory",
                                        history.beforeQuestion(scope),
                                        "evidence",
                                        evidence
                                )
                        )
                )
        );
    }

    private List<Map<String, Object>> meetingEvidence(
            List<SearchDocumentResult> documents,
            Map<String, SearchGroundedEvidence> sourceIndex
    ) {
        return documents.stream()
                .collect(Collectors.groupingBy(SearchDocumentResult::meetingId, LinkedHashMap::new, Collectors.toList()))
                .entrySet()
                .stream()
                .map(group -> Map.<String, Object>of(
                        "meetingId", group.getKey(),
                        "documents", evidence(group.getValue(), sourceIndex)
                ))
                .toList();
    }

    private List<Map<String, Object>> evidence(
            List<SearchDocumentResult> documents,
            Map<String, SearchGroundedEvidence> sourceIndex
    ) {
        return documents.stream()
                .map(
                        document -> Map.<String, Object>of(
                                "evidenceId",
                                "E" + document.documentId(),
                                "documentId",
                                document.documentId(),
                                "title",
                                document.title(),
                                "topic",
                                document.topic(),
                                "content",
                                sourceIndex.isEmpty()
                                        ? document.content()
                                        : sourceIndex.get("E" + document.documentId()).content()
                        )
                )
                .toList();
    }
}
