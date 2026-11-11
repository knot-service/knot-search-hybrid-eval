package com.knot.backend.search.application;

import static org.assertj.core.api.Assertions.assertThat;

import com.knot.backend.document.domain.model.DocumentStatus;
import com.knot.backend.search.application.dto.result.SearchDocumentRecord;
import com.knot.backend.search.infrastructure.llm.SearchHybridIndexClient.Passage;
import java.time.Instant;
import org.junit.jupiter.api.Test;

class SearchHybridPassageTest {
    @Test
    void verifiesUnicodeOffsetsAndRejectsStaleText() {
        var doc = new SearchDocumentRecord(
                1,
                "제목",
                "주제",
                "😀원문 구절입니다.",
                "요약",
                DocumentStatus.DRAFT,
                Instant.EPOCH,
                Instant.EPOCH,
                null,
                0
        );
        assertThat(
                SearchHybridWorkflow.exactPassage(
                        doc,
                        new Passage(
                                1,
                                "body",
                                1,
                                3,
                                "원문",
                                2
                        )
                )
        ).isTrue();
        assertThat(
                SearchHybridWorkflow.exactPassage(
                        doc,
                        new Passage(
                                1,
                                "body",
                                1,
                                3,
                                "이전",
                                2
                        )
                )
        ).isFalse();
        assertThat(
                SearchHybridWorkflow.exactPassage(
                        doc,
                        new Passage(
                                1,
                                "summary",
                                0,
                                2,
                                "요약",
                                2
                        )
                )
        ).isTrue();
        assertThat(
                SearchHybridWorkflow.exactPassage(
                        doc,
                        new Passage(
                                2,
                                "summary",
                                0,
                                2,
                                "요약",
                                2
                        )
                )
        ).isFalse();
    }
}
