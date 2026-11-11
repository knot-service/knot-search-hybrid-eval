package com.knot.backend.search.application;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.knot.backend.search.application.SearchHybridQuestion.Route;
import com.knot.backend.search.domain.SearchToolException;
import java.time.Instant;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class SearchHybridQuestionTest {
    private static final Instant REFERENCE = Instant.parse("2026-11-11T01:00:00Z");

    @ParameterizedTest
    @CsvSource({"지난번 배포 회의 찾아줘,DOCUMENT_FIND,1", "최근 세 번의 회의록을 찾아줘,DOCUMENT_FIND,3",
            "Blue-Green 배포를 선택한 이유를 찾아줘,MEETING_QUESTION,0", "10월 19일 회의 내용으로 확인해 줘,MEETING_QUESTION,0",
            "배포 관련 자료 찾아서 요점 알려줘,MEETING_QUESTION,0", "배포 회의 문서 찾아줘. 결정 여부도 알려줘,MEETING_QUESTION,0",
            "배포 결정을 담은 자료가 뭐야?,DOCUMENT_FIND,0", "최근 세 번의 회의에서 배포 전략이 어떻게 바뀌었어?,MULTI_MEETING_SYNTHESIS,3",
            "두 소셜 로그인 회의의 결정 차이를 알려줘,MULTI_MEETING_SYNTHESIS,0"})
    void selectsRoute(
            String question,
            Route route,
            int count
    ) {
        SearchHybridQuestion parsed = SearchHybridQuestion.parse(
                question,
                REFERENCE
        );
        assertThat(parsed.route()).isEqualTo(route);
        assertThat(parsed.recentMeetings()).isEqualTo(count);
    }

    @Test
    void appliesInclusiveKoreanDateRange() {
        SearchHybridQuestion parsed = SearchHybridQuestion.parse(
                "10월 1일부터 15일까지 로그인 논의를 모아줘",
                REFERENCE
        );
        assertThat(parsed.recordedFrom()).isEqualTo(Instant.parse("2026-09-30T15:00:00Z"));
        assertThat(parsed.recordedBefore()).isEqualTo(Instant.parse("2026-10-15T15:00:00Z"));
    }

    @Test
    void appliesPreviousCalendarWeek() {
        SearchHybridQuestion parsed = SearchHybridQuestion.parse(
                "지난주 배포 회의 찾아줘",
                REFERENCE
        );
        assertThat(parsed.recordedFrom()).isEqualTo(Instant.parse("2026-11-01T15:00:00Z"));
        assertThat(parsed.recordedBefore()).isEqualTo(Instant.parse("2026-11-08T15:00:00Z"));
    }

    @Test
    void rejectsInvalidDates() {
        assertThatThrownBy(
                () -> SearchHybridQuestion.parse(
                        "2월 30일 회의 찾아줘",
                        REFERENCE
                )
        ).isInstanceOf(SearchToolException.class);
    }
}
