package com.knot.backend.search.application;

import com.knot.backend.search.application.SearchHybridQuestion.Route;
import com.knot.backend.search.application.dto.result.SearchDocumentRecord;
import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import com.knot.backend.search.infrastructure.llm.SearchHybridIndexClient;
import com.knot.backend.search.infrastructure.persistence.SearchHybridDocumentScope;
import com.knot.backend.search.infrastructure.persistence.SearchHybridDocumentScope.EligibleDocument;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "knot.search.hybrid.index-url")
public class SearchHybridWorkflow {
    private static final int LIMIT = 10;

    private final SearchHybridIndexClient index;
    private final SearchHybridDocumentScope documentScope;
    private final SearchDocumentQuery documents;
    private final SearchToolAccess access;
    private final Set<Long> workspaces;

    public SearchHybridWorkflow(
            SearchHybridIndexClient index,
            SearchHybridDocumentScope documentScope,
            SearchDocumentQuery documents,
            SearchToolAccess access,
            @Value("${knot.search.hybrid.workspaces:}") Set<Long> workspaces
    ) {
        this.index = index;
        this.documentScope = documentScope;
        this.documents = documents;
        this.access = access;
        this.workspaces = Set.copyOf(workspaces);
    }

    public boolean enabledFor(long workspaceId) {
        return workspaces.contains(workspaceId);
    }

    public Result retrieve(
            SearchToolScope scope,
            String rawQuestion
    ) {
        long started = System.nanoTime();
        access.validate(scope);
        SearchHybridQuestion question = SearchHybridQuestion.parse(
                rawQuestion,
                scope.startedAt()
        );
        List<EligibleDocument> eligible = documentScope.find(
                scope,
                question
        );
        long questionNanos = System.nanoTime() - started;
        long retrievalStarted = System.nanoTime();
        if (eligible.isEmpty()) {
            return new Result(
                    question.route(),
                    List.of(),
                    questionNanos,
                    System.nanoTime() - retrievalStarted
            );
        }

        Map<Long, EligibleDocument> allowed = new LinkedHashMap<>();
        eligible.forEach(
                doc -> allowed.put(
                        doc.id(),
                        doc
                )
        );
        List<EligibleDocument> ranked = index.rank(
                scope.workspaceId(),
                rawQuestion,
                List.copyOf(allowed.keySet())
        )
                .stream()
                .filter(allowed::containsKey)
                .distinct()
                .map(allowed::get)
                .toList();
        access.validate(scope);
        List<SearchDocumentResult> selected = select(
                question,
                ranked
        ).stream()
                .map(
                        candidate -> documents.find(
                                scope,
                                candidate.id()
                        )
                                .map(
                                        record -> document(
                                                question,
                                                candidate,
                                                record
                                        )
                                )
                )
                .flatMap(Optional::stream)
                .toList();
        access.validate(scope);
        return new Result(
                question.route(),
                selected,
                questionNanos,
                System.nanoTime() - retrievalStarted
        );
    }

    private List<EligibleDocument> select(
            SearchHybridQuestion question,
            List<EligibleDocument> ranked
    ) {
        if (question.route() != Route.MULTI_MEETING_SYNTHESIS && question.recentMeetings() == 0) {
            return ranked.stream()
                    .limit(LIMIT)
                    .toList();
        }
        Map<Long, EligibleDocument> byMeeting = new LinkedHashMap<>();
        ranked.forEach(
                doc -> byMeeting.putIfAbsent(
                        doc.meetingId(),
                        doc
                )
        );
        List<EligibleDocument> meetings = new ArrayList<>(byMeeting.values());
        if (question.recentMeetings() > 0) {
            meetings.sort(
                    Comparator.comparing(EligibleDocument::recordedAt)
                            .reversed()
            );
            meetings = meetings.stream()
                    .limit(question.recentMeetings())
                    .toList();
        }
        List<EligibleDocument> chosen = meetings.stream()
                .limit(LIMIT)
                .toList();
        if (question.route() == Route.MULTI_MEETING_SYNTHESIS) {
            return chosen.stream()
                    .sorted(Comparator.comparing(EligibleDocument::recordedAt))
                    .toList();
        }
        return chosen;
    }

    private SearchDocumentResult document(
            SearchHybridQuestion question,
            EligibleDocument candidate,
            SearchDocumentRecord record
    ) {
        String content = record.content();
        if (question.route() == Route.MULTI_MEETING_SYNTHESIS) {
            content = "회의 시각: " + candidate.recordedAt() + "\n" + content;
        }
        return new SearchDocumentResult(
                record.documentId(),
                record.title(),
                record.topic(),
                content
        );
    }

    public record Result(
            Route route,
            List<SearchDocumentResult> documents,
            long questionNanos,
            long retrievalNanos
    ) {
    }
}
