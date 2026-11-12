package com.knot.backend.search.infrastructure.llm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;

class SearchGroundedAnswerTest {
    private final JsonMapper mapper = JsonMapper.builder()
            .build();

    @Test
    void rendersExactQuotesAndRejectsUnknownOrCrossDocumentReferences() {
        var docs = List.of(
                new SearchDocumentResult(
                        1,
                        "첫 회의",
                        "배포",
                        "",
                        List.of(
                                "자동 배포 전환을 제안했다.",
                                "금요일은 보류됐다."
                        )
                ),
                new SearchDocumentResult(
                        2,
                        "다음 회의",
                        "배포",
                        "",
                        List.of("다음 주에 전환하기로 합의했다.")
                )
        );
        var result = SearchGroundedAnswer.verify(
                """
                        {"insufficient":false,"claims":[
                          {"text":"처음 제안했고 다음 주 전환에 합의했다.","sources":[
                            {"evidenceId":"E1","quoteId":"E1Q1"},
                            {"evidenceId":"E2","quoteId":"E2Q1"}]},
                          {"text":"금요일에 한다.","sources":[{"evidenceId":"E2","quoteId":"E1Q2"}]},
                          {"text":"전환일은 금요일이다.","sources":[{"evidenceId":"E1","quoteId":"E1Q99"}]}
                        ]}
                        """,
                SearchGroundedEvidence.index(docs),
                mapper
        );
        assertThat(result.claims()).hasSize(1);
        assertThat(result.documentIds()).containsExactly(
                1L,
                2L
        );
        assertThat(result.render())
                .startsWith("처음 제안했고 다음 주 전환에 합의했다.")
                .contains(
                        "문서 #1에는 “자동 배포 전환을 제안했다.”라고 기록돼 있어요.",
                        "문서 #2에는 “다음 주에 전환하기로 합의했다.”라고 기록돼 있어요.",
                        "\n\n일부 내용"
                )
                .doesNotContain("- ", "근거:", "금요일");
    }

    @Test
    void keepsVerifiedClaimsThatNeedFourDistinctDocuments() {
        var docs = List.of(
                new SearchDocumentResult(1, "제안", "탐색", "", List.of("날짜별 탐색 구조가 제안되었다.")),
                new SearchDocumentResult(2, "시각화", "탐색", "", List.of("탐색의 시각화 방식은 아직 정해지지 않았다.")),
                new SearchDocumentResult(3, "할 일", "탐색", "", List.of("할 일 화면의 구체적 구현 방식은 미정이다.")),
                new SearchDocumentResult(408, "홈 & ADR $1", "탐색", "",
                        List.of("홈 화면의 독립 뷰 제공 여부는 정해지지 않았다."))
        );
        var result = SearchGroundedAnswer.verify(
                """
                        {"insufficient":false,"claims":[
                          {"text":"날짜별 탐색 구조는 제안이었다.","sources":[
                            {"evidenceId":"E1","quoteId":"E1Q1"}]},
                          {"text":"탐색의 시각화 방식은 미정이었다.","sources":[
                            {"evidenceId":"E2","quoteId":"E2Q1"}]},
                          {"text":"E3에서는 구현이 미정이었다. [E408Q1]에서도 독립 뷰는 미정이었다.","sources":[
                            {"evidenceId":"E3","quoteId":"E3Q1"},{"evidenceId":"E408","quoteId":"E408Q1"}]}
                        ]}
                        """,
                SearchGroundedEvidence.index(docs),
                mapper
        );
        assertThat(result.claims()).hasSize(3);
        assertThat(result.documentIds()).containsExactly(1L, 2L, 3L, 408L);
        assertThat(result.render()).contains(
                "「할 일」에서는 구현이 미정이었다. 「홈 & ADR $1」에서도 독립 뷰는 미정이었다.",
                "문서 #408에는 “홈 화면의 독립 뷰 제공 여부는 정해지지 않았다.”"
        ).doesNotContain("E3", "E408");
        assertThat(result.incomplete()).isFalse();
    }

    @Test
    void indexesOnlyContiguousExcerptsAndPreservesMeetingTimeForTheModel() {
        String longExcerpt = "고객 문의가 반복되어 개선 가능성을 검토했다. ".repeat(20).strip();
        String decision = "세션 유지 시간은 확정되지 않았다.";
        String time = "회의 시각(한국): 2026-10-15 23:59:00 +09:00";
        var document = new SearchDocumentResult(
                2158, "로그인", "세션", time + "\n" + decision,
                List.of("## 결정\n" + decision, longExcerpt)
        );
        var evidence = SearchGroundedEvidence.index(List.of(document)).get("E2158");
        assertThat(evidence.content()).startsWith(time).contains("[E2158Q1] " + decision);
        assertThat(evidence.quotes()).containsEntry("E2158Q1", decision);
        assertThat(evidence.quotes().values()).allMatch(quote ->
                quote.length() <= 300 && (decision.contains(quote) || longExcerpt.contains(quote)));
    }

    @Test
    void doesNotAcceptModelTextAfterJson() {
        assertThatThrownBy(
                () -> SearchGroundedAnswer.verify(
                        "{\"claims\":[],\"insufficient\":true} 검증되지 않은 문장",
                        Map.of(),
                        mapper
                )
        ).isInstanceOf(JacksonException.class);
    }
}
