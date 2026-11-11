package com.knot.backend.search.infrastructure.llm;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import org.springframework.core.io.ClassPathResource;

public class SearchAnswerPrompt {
    private static final String SYSTEM = load("prompts/search-answer-system.md");
    private static final String GROUNDED = load("prompts/search-grounded-answer-system.md");

    private SearchAnswerPrompt() {}

    public static String system() {
        return SYSTEM;
    }

    public static String grounded() {
        return GROUNDED;
    }

    private static String load(String path) {
        try {
            return new ClassPathResource(path).getContentAsString(StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new UncheckedIOException(
                    "탐색 시스템 프롬프트를 읽을 수 없습니다",
                    exception
            );
        }
    }
}
