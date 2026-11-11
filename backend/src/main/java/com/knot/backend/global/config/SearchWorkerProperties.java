package com.knot.backend.global.config;

import com.knot.backend.search.domain.SearchErrorCode;
import com.knot.backend.search.domain.SearchException;
import java.time.Duration;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties(prefix = "knot.search.worker")
public class SearchWorkerProperties {

    private boolean enabled;
    private int concurrency = 2;
    private int lookupConcurrency;
    private int candidateLimit = 20;
    private Duration pollInterval = Duration.ofSeconds(1);
    private Duration executionLease = Duration.ofSeconds(150);
    private Duration streamPollInterval = Duration.ofMillis(100);
    private Duration streamTimeout = Duration.ofMinutes(3);
    private Duration streamHeartbeatInterval = Duration.ofSeconds(15);
    private int streamConcurrency = 8;

    public void validate() {
        validateConcurrency();
        validateCandidateLimit();
        validateStreamConcurrency();
        validateDuration(pollInterval);
        validateDuration(executionLease);
        validateDuration(streamPollInterval);
        validateDuration(streamTimeout);
        validateDuration(streamHeartbeatInterval);
    }

    private void validateConcurrency() {
        if (concurrency < 1 || concurrency > 16) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_CONFIGURATION);
        }
    }

    private void validateCandidateLimit() {
        if (lookupConcurrency < 0 || lookupConcurrency > 16) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_CONFIGURATION);
        }
        if (candidateLimit < Math.max(concurrency, lookupConcurrency) || candidateLimit > 100) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_CONFIGURATION);
        }
    }

    private void validateStreamConcurrency() {
        if (streamConcurrency < 1 || streamConcurrency > 100) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_CONFIGURATION);
        }
    }

    private void validateDuration(Duration duration) {
        if (duration == null || duration.isNegative() || duration.isZero() || duration.toMillis() < 1) {
            throw new SearchException(SearchErrorCode.INVALID_SEARCH_CONFIGURATION);
        }
    }
}
