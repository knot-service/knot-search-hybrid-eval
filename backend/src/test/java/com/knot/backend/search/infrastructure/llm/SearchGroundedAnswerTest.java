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
    void acceptsExactExcerptAfterTheDocumentMeetingTime() {
        var document = new SearchDocumentResult(
                2158L,
                "로그인 유지 시간 논의",
                "세션",
                "회의 시각(한국): 2026-10-15 23:59:00 +09:00\n## 결정\n세션 유지 시간은 확정되지 않았다.",
                List.of("## 결정\n세션 유지 시간은 확정되지 않았다.")
        );
        var result = SearchGroundedAnswer.verify(
                """
                        {"insufficient":false,"claims":[
                          {"text":"세션 유지 시간은 미확정이다.","sources":[
                            {"evidenceId":"E2158","quote":"회의 시각(한국): 2026-10-15 23:59:00 +09:00\\n## 결정\\n세션 유지 시간은 확정되지 않았다."}]},
                          {"text":"시각이 다른 주장","sources":[
                            {"evidenceId":"E2158","quote":"회의 시각(한국): 2026-10-16 23:59:00 +09:00\\n세션 유지 시간은 확정되지 않았다."}]},
                          {"text":"원문이 바뀐 주장","sources":[
                            {"evidenceId":"E2158","quote":"회의 시각(한국): 2026-10-15 23:59:00 +09:00\\n세션 유지 시간은 1시간으로 확정되었다."}]}
                        ]}
                        """,
                List.of(document),
                mapper
        );
        assertThat(result.claims()).hasSize(1);
        assertThat(result.documentIds()).containsExactly(2158L);
        assertThat(result.render()).contains("세션 유지 시간은 확정되지 않았다.")
                .doesNotContain("회의 시각", "시각이 다른 주장", "원문이 바뀐 주장");
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
