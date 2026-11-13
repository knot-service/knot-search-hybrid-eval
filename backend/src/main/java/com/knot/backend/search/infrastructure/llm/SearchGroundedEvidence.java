package com.knot.backend.search.infrastructure.llm;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

record SearchGroundedEvidence(long documentId, String title, String content, Map<String, String> quotes) {

    private static final int MAX_QUOTE_LENGTH = 300;

    static Map<String, SearchGroundedEvidence> index(List<SearchDocumentResult> documents) {
        return documents.stream().collect(Collectors.toUnmodifiableMap(
                document -> "E" + document.documentId(), SearchGroundedEvidence::from
        ));
    }

    private static SearchGroundedEvidence from(SearchDocumentResult document) {
        Map<String, String> quotes = new LinkedHashMap<>();
        List<String> content = new ArrayList<>();
        String body = document.content();
        if (body != null && body.startsWith("회의 시각(한국): ")) {
            content.add(body.lines().findFirst().orElseThrow());
        }
        boolean containsSummary = !document.summaryExcerptIndexes().isEmpty();
        for (int i = 0; i < document.excerpts().size(); i++) {
            if (containsSummary) {
                if (document.summaryExcerptIndexes().contains(i)) {
                    content.add("발췌 유형: 문서 요약");
                } else {
                    content.add("발췌 유형: 문서 본문");
                }
            }
            for (String line : document.excerpts().get(i).lines().toList()) {
                if (line.isBlank() || line.stripLeading().startsWith("#")) {
                    content.add(line);
                    continue;
                }
                for (String sentence : sentences(line)) {
                    if (sentence.length() < 8) {
                        content.add(sentence);
                        continue;
                    }
                    String id = "E" + document.documentId() + "Q" + (quotes.size() + 1);
                    quotes.put(id, sentence);
                    content.add("[" + id + "] " + sentence);
                }
            }
            content.add("");
        }
        return new SearchGroundedEvidence(document.documentId(), document.title(), String.join("\n", content),
                Map.copyOf(quotes));
    }

    private static List<String> sentences(String line) {
        if (line.length() <= MAX_QUOTE_LENGTH) {
            return List.of(line.strip());
        }
        BreakIterator boundaries = BreakIterator.getSentenceInstance(Locale.KOREAN);
        boundaries.setText(line);
        List<String> result = new ArrayList<>();
        int start = boundaries.first();
        for (int end = boundaries.next(); end != BreakIterator.DONE; end = boundaries.next()) {
            while (end - start > MAX_QUOTE_LENGTH) {
                int split = line.lastIndexOf(' ', start + MAX_QUOTE_LENGTH);
                if (split <= start) {
                    split = start + MAX_QUOTE_LENGTH;
                    if (Character.isHighSurrogate(line.charAt(split - 1))) {
                        split--;
                    }
                }
                result.add(line.substring(start, split).strip());
                start = split;
            }
            result.add(line.substring(start, end).strip());
            start = end;
        }
        return result;
    }
}
