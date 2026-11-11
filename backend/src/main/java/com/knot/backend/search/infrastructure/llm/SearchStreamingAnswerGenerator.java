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
        return generate(scope, question, documents, onDelta, false);
    }

    @Override
    public List<Long> generateSynthesis(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            Consumer<String> onDelta
    ) {
        return generate(scope, question, documents, onDelta, true);
    }

    private List<Long> generate(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents,
            Consumer<String> onDelta,
            boolean synthesis
    ) {
        access.validate(scope);
        List<Map<String, Object>> messages = messages(
                scope,
                question,
                documents
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
        Consumer<String> onModelText = text -> {
            access.validate(scope);
            answer.accept(text);
        };
        if (synthesis) {
            llm.streamStructuredAnswer(scope, messages, onModelText);
        } else {
            llm.stream(scope, messages, onModelText);
        }
        access.validate(scope);
        return answer.complete(documents);
    }

    private List<Map<String, Object>> messages(
            SearchToolScope scope,
            String question,
            List<SearchDocumentResult> documents
    ) {
        return List.of(
                Map.of(
                        "role",
                        "system",
                        "content",
                        SearchAnswerPrompt.system()
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
