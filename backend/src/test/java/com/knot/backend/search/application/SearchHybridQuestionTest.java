package com.knot.backend.search.application;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.knot.backend.search.application.SearchHybridQuestion.Route;
import com.knot.backend.search.application.SearchHybridQuestion.DateRange;
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
            "두 소셜 로그인 회의의 결정 차이를 알려줘,MULTI_MEETING_SYNTHESIS,0",
            "11월 1일이랑 11월 5일 얘기를 같이 보면 AI를 어떻게 다루기로 한 거야?,MULTI_MEETING_SYNTHESIS,0",
            "10월 20일이랑 22일에 각각 뭘 먼저 하기로 했는지 비교해 줘.,MULTI_MEETING_SYNTHESIS,0",
            "문서 277 찾아줘,DOCUMENT_FIND,0"})
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
    void recognizesOnlyAnExplicitSingleDocumentNumber() {
        assertThat(SearchHybridQuestion.explicitDocumentId("문서 #277에서는 어떻게 정했어?")).isEqualTo(277L);
        assertThat(SearchHybridQuestion.explicitDocumentId("10월 27일 회의의 자료는?")).isNull();
        assertThat(SearchHybridQuestion.explicitDocumentId("문서 277과 문서 278의 차이")).isNull();
        assertThat(SearchHybridQuestion.explicitDocumentId("문서 277건에서 관련 내용을 찾아줘")).isNull();
    }

    @Test
    void forwardsTheRequestedDaysWhenDateComparisonDeterminesTheRoute() {
        var parsed = SearchHybridQuestion.parse("10월 20일이랑 22일에 각각 뭘 먼저 하기로 했는지 비교해 줘.", REFERENCE);
        assertThat(parsed.inferredComparisonDays()).containsExactly("2026-10-20", "2026-10-22");
    }

    @Test
    void appliesInclusiveKoreanDateRange() {
        SearchHybridQuestion parsed = SearchHybridQuestion.parse(
                "10월 1일부터 15일까지 로그인 논의를 모아줘",
                REFERENCE
        );
        assertThat(parsed.recordedDates()).containsExactly(range("2026-09-30T15:00:00Z", "2026-10-15T15:00:00Z"));
    }

    @Test
    void appliesPreviousCalendarWeek() {
        SearchHybridQuestion parsed = SearchHybridQuestion.parse(
                "지난주 배포 회의 찾아줘",
                REFERENCE
        );
        assertThat(parsed.recordedDates()).containsExactly(range("2026-11-01T15:00:00Z", "2026-11-08T15:00:00Z"));
    }

    @Test
    void selectsOnlyTheTwoMentionedDays() {
        var parsed = SearchHybridQuestion.parse("10월 22일이랑 25일 회의 비교해 줘", REFERENCE);
        assertThat(parsed.recordedDates()).containsExactly(
                range("2026-10-21T15:00:00Z", "2026-10-22T15:00:00Z"),
                range("2026-10-24T15:00:00Z", "2026-10-25T15:00:00Z"));
    }

    @Test
    void distinguishesTheNameFromJune() {
        assertThat(SearchHybridQuestion.parse("6월이가 맡긴 업무는?", REFERENCE).recordedDates()).isEmpty();
        assertThat(SearchHybridQuestion.parse("6월 회의에서 맡긴 업무는?", REFERENCE).recordedDates())
                .containsExactly(range("2026-05-31T15:00:00Z", "2026-06-30T15:00:00Z"));
    }

    @Test
    void recognizesRecentlyEndedMeetingWithoutChangingLookupRoute() {
        var question = SearchHybridQuestion.parse("방금 끝난 정기 결제 리허설 회의에서 뭘 결정했어?", REFERENCE);
        assertThat(question.recentlyEndedMeeting()).isTrue();
        assertThat(question.meetingTopic()).containsExactly("정기", "결제", "리허설");

        var lookup = SearchHybridQuestion.parse("방금 끝난 회의 찾아줘", REFERENCE);
        assertThat(lookup.route()).isEqualTo(Route.DOCUMENT_FIND);
        assertThat(lookup.recentlyEndedMeeting()).isTrue();
        assertThat(lookup.meetingTopic()).isEmpty();

        var comparison = SearchHybridQuestion.parse("방금 끝난 배포 회의와 지난 회의의 차이를 비교해 줘", REFERENCE);
        assertThat(comparison.route()).isEqualTo(Route.MULTI_MEETING_SYNTHESIS);
        assertThat(comparison.recentlyEndedMeeting()).isFalse();
    }

    @Test
    void matchesTheDatedDiscussionTopicInDocumentMetadata() {
        var question = SearchHybridQuestion.parse(
                "잠깐, 10월 27일 유입 분석 논의에서 홍보 링크에 UTM을 붙이기로 했어?", REFERENCE
        );
        assertThat(question.meetingTopic()).containsExactly("유입", "분석");
        assertThat(question.matchesMeetingTopic("유입 소스 추적 및 분석 지표 설정", "")).isTrue();
        assertThat(question.matchesMeetingTopic("홍보 글 작성 및 이벤트 기획", "홍보")).isFalse();
        assertThat(SearchHybridQuestion.parse("10월 27일 회의에서 무엇을 결정했어?", REFERENCE).meetingTopic())
                .isEmpty();
    }

    private static DateRange range(String from, String before) {
        return new DateRange(Instant.parse(from), Instant.parse(before));
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
