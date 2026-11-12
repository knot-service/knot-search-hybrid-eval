package com.knot.backend.search.application;

import static org.assertj.core.api.Assertions.assertThat;

import com.knot.backend.search.infrastructure.persistence.SearchHybridDocumentScope.EligibleDocument;
import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.Test;

class SearchHybridMeetingScopeTest {
    private static final Instant REFERENCE = Instant.parse("2026-11-12T01:00:00Z");

    @Test
    void aNamedDiscussionKeepsOtherMeetingsAndDropsUnrelatedSiblingDocuments() {
        var analysis = document(1, 10, "유입 소스 추적 및 분석");
        var promotion = document(2, 10, "홍보 글 작성");
        var otherMeeting = document(3, 20, "사용자 피드백");
        var candidates = List.of(analysis, promotion, otherMeeting);

        var discussion = SearchHybridQuestion.parse("10월 27일 유입 분석 논의에서 뭐라고 했어?", REFERENCE);
        assertThat(SearchHybridWorkflow.scopeToMeetingTopic(discussion, candidates))
                .containsExactly(analysis, otherMeeting);

        var wholeMeeting = SearchHybridQuestion.parse("10월 27일 유입 분석 회의에서 뭐라고 했어?", REFERENCE);
        assertThat(SearchHybridWorkflow.scopeToMeetingTopic(wholeMeeting, candidates))
                .containsExactlyElementsOf(candidates);
    }

    @Test
    void recentlyEndedQuestionRequiresOneMatchingMeeting() {
        var matching = document(1, 10, "배포 리허설 준비");
        var another = document(2, 20, "배포 리허설 회고");
        var unrelated = document(3, 30, "결제 리허설");
        var question = SearchHybridQuestion.parse("방금 끝난 배포 리허설 회의에서 뭘 결정했어?", REFERENCE);

        assertThat(SearchHybridWorkflow.scopeToMeetingTopic(question, List.of(matching, unrelated)))
                .containsExactly(matching);
        assertThat(SearchHybridWorkflow.scopeToMeetingTopic(question, List.of(unrelated))).isEmpty();
        assertThat(SearchHybridWorkflow.scopeToMeetingTopic(question, List.of(matching, another))).isEmpty();

        var lookup = SearchHybridQuestion.parse("방금 끝난 배포 리허설 회의 찾아줘", REFERENCE);
        assertThat(SearchHybridWorkflow.scopeToMeetingTopic(lookup, List.of(matching, another, unrelated)))
                .containsExactly(matching, another);
    }

    private static EligibleDocument document(long id, long meetingId, String title) {
        return new EligibleDocument(id, meetingId, REFERENCE, title, "");
    }
}
