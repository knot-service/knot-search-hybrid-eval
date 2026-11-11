package com.knot.backend.search.infrastructure.llm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;

class SearchGroundedAnswerTest {
    private final JsonMapper mapper = JsonMapper.builder()
            .build();

    @Test
    void keepsClaimsWithExactQuotesAndRejectsWrongDocumentAndStitchedPassages() {
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
                            {"evidenceId":"E1","quote":"자동 배포 전환을 제안했다."},
                            {"evidenceId":"E2","quote":"다음 주에 전환하기로 합의했다."}]},
                          {"text":"금요일에 한다.","sources":[{"evidenceId":"E2","quote":"금요일은 보류됐다."}]},
                          {"text":"전환일은 금요일이다.","sources":[{"evidenceId":"E1","quote":"제안했다. 금요일은 보류됐다."}]}
                        ]}
                        """,
                docs,
                mapper
        );
        assertThat(result.claims()).hasSize(1);
        assertThat(result.documentIds()).containsExactly(
                1L,
                2L
        );
        assertThat(result.render()).contains(
                "문서 #1 근거",
                "문서 #2 근거",
                "일부 내용"
        )
                .doesNotContain("금요일");
    }

    @Test
    void doesNotAcceptModelTextAfterJson() {
        assertThatThrownBy(
                () -> SearchGroundedAnswer.verify(
                        "{\"claims\":[],\"insufficient\":true} 검증되지 않은 문장",
                        List.of(),
                        mapper
                )
        ).isInstanceOf(JacksonException.class);
    }
}
