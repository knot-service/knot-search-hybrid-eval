package com.knot.backend.search.application;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import tools.jackson.databind.ObjectMapper;

final class SearchHybridLoadReplay {
    private SearchHybridLoadReplay() {}

    static List<Sample> wave(
            HttpClient http,
            ObjectMapper mapper,
            List<HttpRequest> requests
    ) throws Exception {
        return wave(http, mapper, requests, Collections.nCopies(requests.size(), 0L));
    }

    static List<Sample> wave(
            HttpClient http,
            ObjectMapper mapper,
            List<HttpRequest> requests,
            List<Long> dispatchDelayMs
    ) throws Exception {
        return wave(http, mapper, requests, dispatchDelayMs, requests.size());
    }

    static List<Sample> wave(
            HttpClient http,
            ObjectMapper mapper,
            List<HttpRequest> requests,
            List<Long> dispatchDelayMs,
            int maxInFlight
    ) throws Exception {
        CountDownLatch ready = new CountDownLatch(requests.size());
        CountDownLatch start = new CountDownLatch(1);
        Semaphore inFlight = new Semaphore(maxInFlight, true);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<Sample>> results = new ArrayList<>();
            for (int index = 0; index < requests.size(); index++) {
                HttpRequest request = requests.get(index);
                long delay = dispatchDelayMs.get(index);
                results.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();
                    if (delay > 0) {
                        Thread.sleep(delay);
                    }
                    inFlight.acquire();
                    try {
                        return read(http, mapper, request);
                    } finally {
                        inFlight.release();
                    }
                }));
            }
            try {
                if (!ready.await(
                        10,
                        TimeUnit.SECONDS
                )) {
                    throw new IllegalStateException("Load clients did not reach the dispatch barrier");
                }
                start.countDown();
                List<Sample> samples = new ArrayList<>();
                for (Future<Sample> future : results) {
                    samples.add(
                            future.get(
                                    180,
                                    TimeUnit.SECONDS
                            )
                    );
                }
                return List.copyOf(samples);
            } finally {
                start.countDown();
                results.forEach(future -> future.cancel(true));
            }
        }
    }

    private static Sample read(
            HttpClient http,
            ObjectMapper mapper,
            HttpRequest request
    ) throws InterruptedException {
        long started = System.nanoTime();
        long answerId = 0;
        Long first = null;
        int status = 0;
        StringBuilder answer = new StringBuilder();
        try {
            var response = http.send(
                    request,
                    HttpResponse.BodyHandlers.ofLines()
            );
            status = response.statusCode();
            try (var lines = response.body()) {
                if (status != 200) {
                    return new Sample(
                            status,
                            answerId,
                            "http_error",
                            null,
                            null,
                            ""
                    );
                }
                String event = "";
                var stream = lines.iterator();
                while (stream.hasNext()) {
                    String line = stream.next();
                    if (line.startsWith("event:")) {
                        event = line.substring(6)
                                .strip();
                    }
                    if (!line.startsWith("data:")) {
                        continue;
                    }
                    var data = mapper.readTree(line.substring(5));
                    long received = System.nanoTime() - started;
                    if (data.has("answerMessageId")) {
                        answerId = data.path("answerMessageId")
                                .asLong();
                    }
                    if ("delta".equals(event)) {
                        String delta = data.path("text")
                                .asText();
                        answer.append(delta);
                        if (first == null && !delta.isBlank()) {
                            first = received;
                        }
                    }
                    if ("completed".equals(event)) {
                        return new Sample(
                                status,
                                answerId,
                                event,
                                first,
                                received,
                                answer.toString()
                        );
                    }
                    if ("failed".equals(event) || "stopped".equals(event)) {
                        return new Sample(
                                status,
                                answerId,
                                event,
                                first,
                                null,
                                answer.toString()
                        );
                    }
                }
            }
            return new Sample(
                    status,
                    answerId,
                    "incomplete_stream",
                    first,
                    null,
                    answer.toString()
            );
        } catch (IOException | java.io.UncheckedIOException error) {
            return new Sample(
                    status,
                    answerId,
                    "transport_error",
                    first,
                    null,
                    answer.toString()
            );
        }
    }

    record Sample(
            int httpStatus,
            long answerId,
            String terminal,
            Long firstNanos,
            Long completedNanos,
            String answer
    ) {
    }
}
