package com.knot.backend.search.infrastructure.llm;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import tools.jackson.databind.ObjectMapper;

record SearchGroundedAnswer(
        List<ModelClaim> claims,
        boolean insufficient
) {
    private static final int MAX_CLAIMS = 6;
    private static final int MAX_SOURCES_PER_CLAIM = 2;

    static Map<String, Object> responseFormat() {
        var source = object(
                Map.of(
                        "evidenceId",
                        string(),
                        "quoteId",
                        string()
                ),
                List.of(
                        "evidenceId",
                        "quoteId"
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
            Map<String, SearchGroundedEvidence> evidence,
            ObjectMapper mapper
    ) {
        SearchGroundedAnswer answer = mapper.reader()
                .with(tools.jackson.core.StreamReadFeature.STRICT_DUPLICATE_DETECTION)
                .with(tools.jackson.databind.DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
                .forType(SearchGroundedAnswer.class)
                .readValue(json);
        if (answer.claims() == null || answer.claims().size() > MAX_CLAIMS) {
            return new Verified(List.of(), List.of(), true);
        }
        List<Claim> valid = new ArrayList<>();
        Set<Long> used = new LinkedHashSet<>();
        boolean incomplete = answer.insufficient();
        for (ModelClaim claim : answer.claims()) {
            Claim checked = verifiedClaim(claim, evidence);
            if (checked == null) {
                incomplete = true;
                continue;
            }
            checked.sources().forEach(source -> used.add(source.documentId()));
            valid.add(checked);
        }
        return new Verified(List.copyOf(valid), List.copyOf(used), incomplete);
    }

    private static Claim verifiedClaim(
            ModelClaim claim,
            Map<String, SearchGroundedEvidence> evidence
    ) {
        if (claim == null || claim.text() == null || claim.text().isBlank()
                || claim.text().length() > 600 || claim.sources() == null
                || claim.sources().isEmpty() || claim.sources().size() > MAX_SOURCES_PER_CLAIM) {
            return null;
        }
        Set<Source> verified = new LinkedHashSet<>();
        for (ModelSource source : claim.sources()) {
            if (source == null || source.evidenceId() == null || source.quoteId() == null) {
                return null;
            }
            SearchGroundedEvidence document = evidence.get(source.evidenceId());
            String quote = document == null ? null : document.quotes().get(source.quoteId());
            if (quote == null) {
                return null;
            }
            verified.add(new Source(document.documentId(), quote));
        }
        return new Claim(claim.text(), List.copyOf(verified));
    }

    record ModelClaim(String text, List<ModelSource> sources) {
    }

    record ModelSource(String evidenceId, String quoteId) {
    }

    record Claim(
            String text,
            List<Source> sources
    ) {
        String render() {
            String references = sources.stream()
                    .map(Source::render)
                    .collect(Collectors.joining(" "));
            String paragraph = text.strip()
                    .replaceAll("\\R+", " ");
            return paragraph + " " + references;
        }
    }

    record Source(
            long documentId,
            String quote
    ) {
        String render() {
            String excerpt = quote.replaceAll("\\R", " ");
            return "문서 #" + documentId + "에는 “" + excerpt + "”라고 기록돼 있어요.";
        }
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
                    .collect(Collectors.joining("\n\n"));
            if (incomplete) {
                return answer + "\n\n일부 내용은 찾은 회의 문서에서 확인하지 못했습니다.";
            }
            return answer;
        }
    }
}
