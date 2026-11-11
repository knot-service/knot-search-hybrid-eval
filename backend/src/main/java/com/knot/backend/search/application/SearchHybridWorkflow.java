package com.knot.backend.search.application;

import com.knot.backend.search.application.SearchHybridQuestion.Route;
import com.knot.backend.search.application.dto.result.SearchDocumentRecord;
import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import com.knot.backend.search.infrastructure.llm.SearchHybridIndexClient;
import com.knot.backend.search.infrastructure.llm.SearchHybridIndexClient.Passage;
import com.knot.backend.search.infrastructure.persistence.SearchHybridDocumentScope;
import com.knot.backend.search.infrastructure.persistence.SearchHybridDocumentScope.EligibleDocument;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "knot.search.hybrid.index-url")
public class SearchHybridWorkflow {
    private static final int LIMIT = 10;
    private static final int CONTEXT_TOKENS = 3072;
    private static final int SUPPORTING_CONTEXT_TOKENS = 2048;
    private static final int MAX_PASSAGES_PER_KIND = 5;
    private static final int MAX_SUPPORTING_DOCUMENTS_PER_MEETING = 2;
    private static final DateTimeFormatter MEETING_TIME = DateTimeFormatter.ofPattern("uuuu-MM-dd HH:mm:ss XXX")
            .withZone(ZoneId.of("Asia/Seoul"));

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
                    List.of(),
                    questionNanos,
                    0,
                    0,
                    0
            );
        }

        Map<Long, EligibleDocument> allowed = new LinkedHashMap<>();
        eligible.forEach(
                doc -> allowed.put(
                        doc.id(),
                        doc
                )
        );
        var ranking = index.rank(
                scope.workspaceId(),
                rawQuestion,
                List.copyOf(allowed.keySet()),
                contextMode(question.route()),
                question.recentMeetings()
        );
        access.validate(scope);
        List<Passage> verified = new ArrayList<>();
        List<SearchDocumentResult> selected;
        if (question.route() == Route.DOCUMENT_FIND) {
            selected = lookup(
                    scope,
                    question,
                    ranking.documentIds(),
                    allowed
            );
        } else {
            selected = context(
                    scope,
                    question,
                    ranking.passages(),
                    ranking.supportingPassages(),
                    allowed,
                    verified
            );
        }
        access.validate(scope);
        return new Result(
                question.route(),
                selected,
                List.copyOf(verified),
                questionNanos,
                System.nanoTime() - retrievalStarted,
                ranking.rerankMs(),
                ranking.contextMs()
        );
    }

    private String contextMode(Route route) {
        return switch (route) {
            case DOCUMENT_FIND -> "none";
            case MEETING_QUESTION -> "question";
            case MULTI_MEETING_SYNTHESIS -> "synthesis";
        };
    }

    private List<SearchDocumentResult> lookup(
            SearchToolScope scope,
            SearchHybridQuestion question,
            List<Long> ids,
            Map<Long, EligibleDocument> allowed
    ) {
        var ranked = ids.stream()
                .filter(allowed::containsKey)
                .distinct()
                .map(allowed::get);
        if (question.recentMeetings() > 0) {
            Map<Long, EligibleDocument> byMeeting = new LinkedHashMap<>();
            ranked.forEach(
                    doc -> byMeeting.putIfAbsent(
                            doc.meetingId(),
                            doc
                    )
            );
            ranked = byMeeting.values()
                    .stream()
                    .sorted(
                            Comparator.comparing(EligibleDocument::recordedAt)
                                    .reversed()
                    )
                    .limit(question.recentMeetings());
        }
        return ranked.limit(LIMIT)
                .map(
                        candidate -> documents.find(
                                scope,
                                candidate.id()
                        )
                )
                .flatMap(Optional::stream)
                .map(
                        doc -> new SearchDocumentResult(
                                doc.documentId(),
                                doc.title(),
                                doc.topic(),
                                ""
                        )
                )
                .toList();
    }

    private List<SearchDocumentResult> context(
            SearchToolScope scope,
            SearchHybridQuestion question,
            List<Passage> passages,
            List<Passage> supportingPassages,
            Map<Long, EligibleDocument> allowed,
            List<Passage> verified
    ) {
        if (passages == null || passages.size() > MAX_PASSAGES_PER_KIND
                || supportingPassages.size() > MAX_PASSAGES_PER_KIND) {
            throw new IllegalStateException("Invalid hybrid context budget");
        }
        var allPassages = Stream.concat(passages.stream(), supportingPassages.stream()).toList();
        int budget = supportingPassages.isEmpty() ? CONTEXT_TOKENS : SUPPORTING_CONTEXT_TOKENS;
        if (allPassages.stream().anyMatch(p -> p.tokenCount() < 1)
                || allPassages.stream().mapToLong(Passage::tokenCount).sum() > budget) {
            throw new IllegalStateException("Invalid hybrid context budget");
        }
        var primaryDocuments = passages.stream().map(Passage::documentId).filter(allowed::containsKey)
                .collect(Collectors.toSet());
        var primaryMeetings = primaryDocuments.stream().map(id -> allowed.get(id).meetingId())
                .collect(Collectors.toSet());
        var supportCounts = new HashMap<Long, Integer>();
        var seenDocuments = new HashSet<>(primaryDocuments);
        var eligibleSupport = supportingPassages.stream()
                .filter(p -> allowed.containsKey(p.documentId()) && seenDocuments.add(p.documentId()))
                .filter(p -> {
                    long meeting = allowed.get(p.documentId()).meetingId();
                    return primaryMeetings.contains(meeting)
                            && supportCounts.merge(meeting, 1, Integer::sum) <= MAX_SUPPORTING_DOCUMENTS_PER_MEETING;
                });
        Map<Long, List<Passage>> byDocument = Stream.concat(passages.stream(), eligibleSupport)
                .filter(p -> allowed.containsKey(p.documentId()))
                .collect(
                        Collectors.groupingBy(
                                Passage::documentId,
                                LinkedHashMap::new,
                                Collectors.toList()
                        )
                );
        var entries = byDocument.entrySet()
                .stream();
        if (question.route() == Route.MULTI_MEETING_SYNTHESIS) {
            entries = entries.sorted(
                    Comparator.comparing(
                            entry -> allowed.get(entry.getKey())
                                    .recordedAt()
                    )
            );
        }
        return entries.map(
                entry -> documents.find(
                        scope,
                        entry.getKey()
                )
                        .flatMap(
                                doc -> excerpt(
                                        doc,
                                        entry.getValue(),
                                        verified
                                )
                        )
        )
                .flatMap(Optional::stream)
                .toList();
    }

    private Optional<SearchDocumentResult> excerpt(
            SearchDocumentRecord doc,
            List<Passage> passages,
            List<Passage> verified
    ) {
        List<Passage> valid = passages.stream()
                .filter(
                        passage -> exactPassage(
                                doc,
                                passage
                        )
                )
                .toList();
        if (valid.isEmpty()) {
            return Optional.empty();
        }
        verified.addAll(valid);
        List<String> quotes = valid.stream()
                .map(Passage::quote)
                .toList();
        String text = "회의 시각(한국): " + MEETING_TIME.format(doc.recordingStartedAt()) + "\n" + String.join(
                "\n\n",
                quotes
        );
        return Optional.of(
                new SearchDocumentResult(
                        doc.documentId(),
                        doc.title(),
                        doc.topic(),
                        text,
                        quotes
                )
        );
    }

    static boolean exactPassage(
            SearchDocumentRecord doc,
            Passage passage
    ) {
        String source = switch (passage.field()) {
            case "body" -> doc.content();
            case "summary" -> doc.summary();
            case null, default -> null;
        };
        if (doc.documentId() != passage.documentId() || source == null || passage.quote() == null || passage.quote()
                .isBlank() || passage.start() < 0 || passage.end() <= passage.start()
                || passage.end() > source.codePointCount(
                        0,
                        source.length()
                )) {
            return false;
        }
        int start = source.offsetByCodePoints(
                0,
                passage.start()
        );
        int end = source.offsetByCodePoints(
                0,
                passage.end()
        );
        return source.substring(
                start,
                end
        )
                .equals(passage.quote());
    }

    public record Result(
            Route route,
            List<SearchDocumentResult> documents,
            List<Passage> passages,
            long questionNanos,
            long retrievalNanos,
            double rerankMs,
            double contextMs
    ) {
    }
}
