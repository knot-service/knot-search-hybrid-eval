package com.knot.backend.global.config;

import com.knot.backend.search.application.SearchAnswerSnapshotService;
import com.knot.backend.global.logging.RequestIdTaskDecorator;
import com.knot.backend.search.presentation.stream.SearchAnswerStream;
import org.springframework.beans.factory.annotation.Qualifier;
import com.knot.backend.search.application.SearchGenerationAvailability;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import java.util.concurrent.ThreadPoolExecutor;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties({SearchWorkerProperties.class, SearchAgentProperties.class})
public class SearchWorkerConfig {

    @Bean
    @ConditionalOnProperty(prefix = "knot.search.worker", name = "enabled", havingValue = "true")
    public ThreadPoolTaskExecutor searchGenerationExecutor(SearchWorkerProperties properties) {
        properties.validate();
        if (properties.isShareGenerationSlots()) {
            return workerExecutor(properties.getConcurrency(), "search-priority-");
        }
        return properties.isSplitGenerationRoutes()
                ? workerExecutor(properties.getConcurrency() - 1, "search-question-")
                : workerExecutor(properties.getConcurrency(), "search-generation-");
    }

    @Bean
    @ConditionalOnProperty(prefix = "knot.search.worker", name = "enabled", havingValue = "true")
    public ThreadPoolTaskExecutor searchSynthesisExecutor() {
        return workerExecutor(1, "search-synthesis-");
    }

    @Bean
    @ConditionalOnProperty(prefix = "knot.search.worker", name = "enabled", havingValue = "true")
    public ThreadPoolTaskExecutor searchLookupExecutor(SearchWorkerProperties properties) {
        properties.validate();
        return workerExecutor(Math.max(1, properties.getLookupConcurrency()), "search-lookup-");
    }

    private ThreadPoolTaskExecutor workerExecutor(int concurrency, String threadNamePrefix) {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(concurrency);
        executor.setMaxPoolSize(concurrency);
        executor.setQueueCapacity(0);
        executor.setThreadNamePrefix(threadNamePrefix);
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.AbortPolicy());
        executor.setWaitForTasksToCompleteOnShutdown(false);
        return executor;
    }

    @Bean
    public ThreadPoolTaskExecutor searchStreamExecutor(SearchWorkerProperties properties) {
        properties.validate();
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(properties.getStreamConcurrency());
        executor.setMaxPoolSize(properties.getStreamConcurrency());
        executor.setQueueCapacity(properties.getStreamConcurrency());
        executor.setThreadNamePrefix("search-stream-");
        executor.setTaskDecorator(new RequestIdTaskDecorator());
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.AbortPolicy());
        executor.setWaitForTasksToCompleteOnShutdown(false);
        return executor;
    }

    @Bean
    @ConditionalOnProperty(prefix = "knot.search.worker", name = "enabled", havingValue = "true")
    public org.springframework.beans.factory.SmartInitializingSingleton searchGenerationReadiness(
            SearchGenerationAvailability availability
    ) {
        return availability::requireAvailable;
    }

    @Bean
    public SearchAnswerStream searchAnswerStream(
            SearchAnswerSnapshotService snapshots,
            @Qualifier("searchStreamExecutor") ThreadPoolTaskExecutor executor,
            SearchWorkerProperties properties
    ) {
        return new SearchAnswerStream(
                snapshots,
                executor,
                properties
        );
    }
}
