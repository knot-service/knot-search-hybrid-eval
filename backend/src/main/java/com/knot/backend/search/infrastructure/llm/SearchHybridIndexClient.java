package com.knot.backend.search.infrastructure.llm;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

@Component
@ConditionalOnProperty(name = "knot.search.hybrid.index-url")
public class SearchHybridIndexClient implements AutoCloseable {

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(2))
            .build();
    private final ObjectMapper mapper;
    private final URI endpoint;

    public SearchHybridIndexClient(
            ObjectMapper mapper,
            @Value("${knot.search.hybrid.index-url}") URI endpoint
    ) {
        this.mapper = mapper;
        this.endpoint = endpoint;
    }

    public Output rank(
            long workspaceId,
            String question,
            List<Long> allowedDocumentIds,
            String contextMode,
            int recentMeetings
    ) {
        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(Duration.ofSeconds(10))
                .header(
                        "Content-Type",
                        "application/json"
                )
                .POST(
                        HttpRequest.BodyPublishers.ofString(
                                mapper.writeValueAsString(
                                        new Input(
                                                workspaceId,
                                                question,
                                                allowedDocumentIds,
                                                contextMode,
                                                recentMeetings
                                        )
                                )
                        )
                )
                .build();
        try {
            var response = http.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Hybrid index returned HTTP " + response.statusCode());
            }
            return mapper.readValue(
                    response.body(),
                    Output.class
            );
        } catch (IOException failure) {
            throw new IllegalStateException(
                    "Hybrid index request failed",
                    failure
            );
        } catch (InterruptedException failure) {
            Thread.currentThread()
                    .interrupt();
            throw new IllegalStateException(
                    "Hybrid index request interrupted",
                    failure
            );
        }
    }

    @Override
    public void close() {
        http.close();
    }

    private record Input(
            long workspaceId,
            String question,
            List<Long> allowedDocumentIds,
            String contextMode,
            int recentMeetings
    ) {
    }

    public record Output(
            List<Long> documentIds,
            List<Passage> passages,
            double rerankMs,
            double contextMs
    ) {
    }

    public record Passage(
            long documentId,
            String field,
            int start,
            int end,
            String quote,
            int tokenCount
    ) {
    }
}
