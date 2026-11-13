package com.knot.backend.search.infrastructure.llm;

import static org.assertj.core.api.Assertions.assertThat;

import com.knot.backend.search.application.dto.result.SearchDocumentResult;
import java.util.List;
import org.junit.jupiter.api.Test;

class SearchGroundedDocumentScopeTest {

    @Test
    void keepsTheHighestRankedEvidenceWhenTheDatedDiscussionTitleMatchesAnotherDocument() {
        var unrelated = document(1, "승인 전략과 운영 범위", 12L);
        var discussion = document(2, "배포 일정 및 승인 전략 논의", 12L);
        var reversedTitle = document(3, "승인 및 배포 전략 검토", 13L);
        var documents = List.of(unrelated, discussion, reversedTitle);

        assertThat(SearchGroundedDocumentScope.selectDatedDiscussion(
                "10월 24일 배포와 승인 전략 논의에서 승인 범위를 어디까지 정했어?", documents
        )).containsExactly(unrelated, discussion);
    }

    @Test
    void retainsRankedEvidenceFromAnotherMeetingWhenItAloneAnswersTheTechnicalTerm() {
        var ranked = new SearchDocumentResult(1L, "유입 지표 검토", "", "", List.of("UTM 채택은 미정이다."), 11L);
        var titled = document(2, "유료 광고 및 유입 분석", 12L);

        assertThat(SearchGroundedDocumentScope.selectDatedDiscussion(
                "10월 27일 유입 분석 논의에서 UTM 파라미터를 붙이기로 확정했어?",
                List.of(ranked, titled)
        )).containsExactly(ranked, titled);
    }

    @Test
    void keepsTheNamedMeetingWhenTheHigherRankedDocumentIsFromAnotherMeeting() {
        var ranked = document(1, "SNS 및 커뮤니티 홍보 채널 선정", 11L);
        var titled = document(2, "홍보 채널 선정 및 로그인 전략 논의", 12L);

        assertThat(SearchGroundedDocumentScope.selectDatedDiscussion(
                "10월 27일 홍보와 로그인 전략 논의에서 어떤 로그인을 추가하기로 했어?",
                List.of(ranked, titled)
        )).containsExactly(titled);
    }

    @Test
    void narrowsTheDatedDiscussionWhenRankingAndTitleMatchAgree() {
        var discussion = document(1, "배포 일정 및 승인 전략 논의", 11L);
        var other = document(2, "개발 환경 점검", 12L);

        assertThat(SearchGroundedDocumentScope.selectDatedDiscussion(
                "10월 24일 배포와 승인 전략 논의에서 승인 범위를 어디까지 정했어?",
                List.of(discussion, other)
        )).containsExactly(discussion);
    }

    @Test
    void prefersTheExactTitleOverASeparateMeetingWithTheSameTopic() {
        var later = new SearchDocumentResult(1, "계정 사용 여부 확인 결과", "계정 사용 여부 확인",
                "회의 시각(한국): 2026-10-19 15:12:44 +09:00", List.of(), null);
        var requested = new SearchDocumentResult(2, "계정 사용 여부 확인", "계정 사용 여부 확인",
                "회의 시각(한국): 2026-10-19 15:12:23 +09:00", List.of(), null);

        assertThat(SearchGroundedDocumentScope.selectDatedDiscussion(
                "10월 19일 계정 사용 여부 확인 회의에서 실제로 쓰는지 확인됐어?", List.of(later, requested)
        )).containsExactly(requested);
    }

    private static SearchDocumentResult document(long id, String title, long meetingId) {
        return new SearchDocumentResult(id, title, title, "", List.of(), meetingId);
    }
}
