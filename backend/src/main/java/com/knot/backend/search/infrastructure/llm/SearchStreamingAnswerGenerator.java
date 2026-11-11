package com.knot.backend.search.infrastructure.llm;

import com.knot.backend.search.infrastructure.persistence.SearchHistoryQuery;

import com.knot.backend.global.config.SearchAgentProperties;
import com.knot.backend.search.application.SearchAnswerGenerator;
import com.knot.backend.search.application.SearchToolAccess;
import com.knot.backend.search.application.SearchToolScope;
import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import lombok.RequiredArgsConstructor;
import tools.jackson.databind.ObjectMapper;

@RequiredArgsConstructor
public class SearchStreamingAnswerGenerator implements SearchAnswerGenerator {

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
                documents,
                SearchAnswerPrompt.system()
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
        var messages = messages(scope, question, documents, SearchAnswerPrompt.grounded());
        new SearchAgentContext(properties, mapper).validate(messages);
        StringBuilder response = new StringBuilder();
        llm.streamStructuredAnswer(scope, messages, response::append);
        access.validate(scope);
        SearchGroundedAnswer.Verified verified = SearchGroundedAnswer.verify(response.toString(), documents, mapper);
        onDelta.accept(verified.render());
        return verified.documentIds();
    }

    private List<Map<String, Object>> messages(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            String systemPrompt
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
                                        evidence(documents)
                                )
                        )
                )
        );
    }

    private List<Map<String, Object>> evidence(List<SearchDocumentResult> documents) {
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
                                document.content()
                        )
                )
                .toList();
    }
}
