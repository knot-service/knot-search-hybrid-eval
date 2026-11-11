package com.knot.backend.search.infrastructure.llm;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import tools.jackson.databind.ObjectMapper;

record SearchGroundedAnswer(
        List<Claim> claims,
        boolean insufficient
) {
    private static final int MAX_CLAIMS = 6;
    private static final int MAX_SOURCES = 3;
    private static final int MAX_SOURCES_PER_CLAIM = 2;

    static Map<String, Object> responseFormat() {
        var source = object(
                Map.of(
                        "evidenceId",
                        string(),
                        "quote",
                        string()
                ),
                List.of(
                        "evidenceId",
                        "quote"
                )
        );
        var claim = object(
                Map.of(
                        "text",
                        string(),
                        "sources",
                        array(
                                source,
                                MAX_SOURCES_PER_CLAIM
                        )
                ),
                List.of(
                        "text",
                        "sources"
                )
        );
        var schema = object(
                Map.of(
                        "claims",
                        array(
                                claim,
                                MAX_CLAIMS
                        ),
                        "insufficient",
                        Map.of(
                                "type",
                                "boolean"
                        )
                ),
                List.of(
                        "claims",
                        "insufficient"
                )
        );
        return Map.of(
                "type",
                "json_schema",
                "json_schema",
                Map.of(
                        "name",
                        "knot_grounded_answer",
                        "strict",
                        true,
                        "schema",
                        schema
                )
        );
    }

    private static Map<String, Object> object(
            Map<String, Object> properties,
            List<String> required
    ) {
        return Map.of(
                "type",
                "object",
                "properties",
                properties,
                "required",
                required,
                "additionalProperties",
                false
        );
    }

    private static Map<String, Object> array(
            Map<String, Object> items,
            int maximum
    ) {
        return Map.of(
                "type",
                "array",
                "items",
                items,
                "maxItems",
                maximum
        );
    }

    private static Map<String, Object> string() {
        return Map.of(
                "type",
                "string"
        );
    }

    static Verified verify(
            String json,
            List<SearchDocumentResult> documents,
            ObjectMapper mapper
    ) {
        SearchGroundedAnswer answer = mapper.reader()
                .with(tools.jackson.core.StreamReadFeature.STRICT_DUPLICATE_DETECTION)
                .with(tools.jackson.databind.DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
                .forType(SearchGroundedAnswer.class)
                .readValue(json);
        Map<String, SearchDocumentResult> sources = documents.stream()
                .collect(
                        Collectors.toMap(
                                doc -> "E" + doc.documentId(),
                                Function.identity()
                        )
                );
        List<Claim> valid = new ArrayList<>();
        Set<Long> used = new LinkedHashSet<>();
        boolean incomplete = answer.insufficient();
        if (answer.claims() == null || answer.claims()
                .size() > MAX_CLAIMS) {
            return new Verified(
                    List.of(),
                    List.of(),
                    true
            );
        }
        for (Claim claim : answer.claims()) {
            if (!validClaim(
                    claim,
                    sources
            )) {
                incomplete = true;
                continue;
            }
            Set<Long> combined = new LinkedHashSet<>(used);
            claim.sources()
                    .forEach(
                            source -> combined.add(
                                    sources.get(source.evidenceId())
                                            .documentId()
                            )
                    );
            if (combined.size() > MAX_SOURCES) {
                incomplete = true;
                continue;
            }
            used = combined;
            valid.add(claim);
        }
        return new Verified(
                List.copyOf(valid),
                List.copyOf(used),
                incomplete
        );
    }

    private static boolean validClaim(
            Claim claim,
            Map<String, SearchDocumentResult> sources
    ) {
        return claim != null && claim.text() != null && !claim.text()
                .isBlank() && claim.text()
                        .length() <= 600
                && claim.sources() != null && !claim.sources()
                        .isEmpty()
                && claim.sources()
                        .size() <= MAX_SOURCES_PER_CLAIM
                && claim.sources()
                        .stream()
                        .allMatch(
                                source -> matches(
                                        source,
                                        sources
                                )
                        );
    }

    private static boolean matches(
            Source source,
            Map<String, SearchDocumentResult> sources
    ) {
        if (source == null || source.quote() == null || source.quote()
                .strip()
                .length() < 8 || source.quote()
                        .strip()
                        .length() > 300) {
            return false;
        }
        SearchDocumentResult document = sources.get(source.evidenceId());
        return document != null && document.excerpts()
                .stream()
                .anyMatch(
                        excerpt -> excerpt.contains(
                                source.quote()
                                        .strip()
                        )
                );
    }

    record Claim(
            String text,
            List<Source> sources
    ) {
        String render() {
            String notes = sources.stream()
                    .map(
                            source -> "  - 문서 #" + source.evidenceId()
                                    .substring(1) + " 근거: "
                                    + source.quote()
                                            .strip()
                                            .replaceAll(
                                                    "\\R",
                                                    " "
                                            )
                    )
                    .collect(Collectors.joining("\n"));
            return "- " + text.strip()
                    .replaceAll(
                            "\\R",
                            " "
                    ) + "\n" + notes;
        }
    }

    record Source(
            String evidenceId,
            String quote
    ) {
    }

    record Verified(
            List<Claim> claims,
            List<Long> documentIds,
            boolean incomplete
    ) {
        String render() {
            if (claims.isEmpty()) {
                return "현재 찾은 회의 문서에서는 이 질문에 답할 근거를 확인하지 못했습니다.";
            }
            String answer = claims.stream()
                    .map(Claim::render)
                    .collect(Collectors.joining("\n"));
            if (incomplete) {
                return answer + "\n\n일부 내용은 찾은 회의 문서에서 확인하지 못했습니다.";
            }
            return answer;
        }
    }
}
