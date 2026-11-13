package com.knot.backend.search.infrastructure.llm;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

final class SearchGroundedDocumentScope {
    private static final Pattern DISCUSSION = Pattern.compile(
            "(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}월\\s*[0-9]{1,2}일)\\s+"
                    + "((?:[\\p{L}\\p{N}]+\\s+){2,5})(?:논의|회의)에서"
    );
    private static final Pattern TECHNICAL_TERM = Pattern.compile("(?<![\\p{L}\\p{N}])[A-Z][A-Z0-9]{1,}(?![A-Za-z0-9])");
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+");

    private SearchGroundedDocumentScope() {
    }

    static List<SearchDocumentResult> selectDatedDiscussion(String question, List<SearchDocumentResult> documents) {
        var discussion = DISCUSSION.matcher(question);
        if (!discussion.find()) {
            return documents;
        }
        List<String> words = words(discussion.group(1));
        String phrase = String.join(" ", words.subList(words.size() - 2, words.size()));
        var titleMatch = unique(documents, document -> {
            var title = words(document.title());
            var topic = words(document.topic());
            return (containsAll(title, words) && String.join(" ", title).contains(phrase))
                    || (containsAll(topic, words) && String.join(" ", topic).contains(phrase));
        });
        if (titleMatch == documents || titleMatch.getFirst().equals(documents.getFirst())) {
            return titleMatch;
        }
        var ranked = documents.getFirst();
        var titled = titleMatch.getFirst();
        String rankedMeeting = meetingKey(ranked);
        boolean sameMeeting = !rankedMeeting.isEmpty() && rankedMeeting.equals(meetingKey(titled));
        if (sameMeeting || answersUnmatchedTechnicalTerm(question.substring(discussion.end()), ranked, titled)) {
            return List.of(ranked, titled);
        }
        return titleMatch;
    }

    private static boolean answersUnmatchedTechnicalTerm(
            String question, SearchDocumentResult ranked, SearchDocumentResult titled
    ) {
        return TECHNICAL_TERM.matcher(question).results()
                .map(match -> Pattern.compile("(?i)(?<![A-Za-z0-9])" + match.group() + "(?![A-Za-z0-9])"))
                .anyMatch(term -> ranked.excerpts().stream().anyMatch(excerpt -> term.matcher(excerpt).find())
                        && titled.excerpts().stream().noneMatch(excerpt -> term.matcher(excerpt).find()));
    }

    private static String meetingKey(SearchDocumentResult document) {
        if (document.meetingId() != null) {
            return document.meetingId().toString();
        }
        String content = document.content();
        return content.startsWith("회의 시각(한국): ") ? content.lines().findFirst().orElse("") : "";
    }

    private static boolean containsAll(List<String> candidate, List<String> query) {
        return query.stream().allMatch(word -> candidate.contains(word)
                || (word.length() > 1 && (word.endsWith("와") || word.endsWith("과"))
                        && candidate.contains(word.substring(0, word.length() - 1))));
    }

    private static List<SearchDocumentResult> unique(
            List<SearchDocumentResult> documents,
            java.util.function.Predicate<SearchDocumentResult> matches
    ) {
        var selected = documents.stream().filter(matches).limit(2).toList();
        return selected.size() == 1 ? selected : documents;
    }

    private static List<String> words(String value) {
        return WORD.matcher(value.toLowerCase(Locale.ROOT)).results().map(match -> match.group()).toList();
    }
}
