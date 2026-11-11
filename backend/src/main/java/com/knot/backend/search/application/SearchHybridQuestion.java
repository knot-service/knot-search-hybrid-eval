package com.knot.backend.search.application;

import com.knot.backend.search.domain.SearchToolException;
import java.text.Normalizer;
import java.time.Instant;
import java.time.LocalDate;
import java.time.YearMonth;
import java.time.ZoneId;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.regex.Pattern;

public record SearchHybridQuestion(
        Route route,
        List<DateRange> recordedDates,
        int recentMeetings,
        List<String> inferredComparisonDays
) {
    public SearchHybridQuestion {
        recordedDates = List.copyOf(recordedDates);
        inferredComparisonDays = List.copyOf(inferredComparisonDays);
    }

    private static final ZoneId KOREA = ZoneId.of("Asia/Seoul");
    private static final String NUMBERED_DOCUMENT = "문서\\s*#?\\s*([1-9][0-9]{0,18})(?![0-9]|\\s*(?:건|개|편|종|쪽|페이지))";
    private static final Pattern DOCUMENT_ID = Pattern.compile(NUMBERED_DOCUMENT);
    private static final String DOCUMENT = "(?:회의(?:록|\\s*문서)?|(?:" + NUMBERED_DOCUMENT + "|문서)|기록|자료|링크|목록)";
    private static final String REQUEST_END = "\\s*(?:[.!?…]+\\s*)?$";
    private static final Pattern DOCUMENT_REQUEST = Pattern.compile(
            DOCUMENT + "(?:를|을|이|가|은|는|도)?\\s*(?:좀\\s*)?" + "(?:(?:찾아|보여|띄워|열어|검색해)(?:\\s*(?:줘|주세요|줄래|줄 수 있어|주실래요))?"
                    + "|(?:뭐|어디)\\s*(?:야|예요|인가요|있어|있어요))" + REQUEST_END + "|(?:어느|어떤|어디)\\s*" + DOCUMENT
                    + "(?:에|이|가)?\\s*(?:있어|있어요|있나요|야|예요)" + REQUEST_END
                    + "|(?:회의록|문서|기록)\\s*(?:링크|목록)(?:를|을)?(?:\\s*(?:줘|주세요))?" + REQUEST_END
    );
    private static final Pattern MULTIPLE_MEETINGS = Pattern.compile(
            "(?<![\\p{L}\\d])(?:두|세|네|다섯|[2-9]|10)(?:\\s*번의)?\\s+(?:[가-힣A-Za-z]+\\s+){0,2}회의"
                    + "|회의\\s*(?:두|세|네|다섯|[2-9]|10)\\s*(?:건|회(?!의)|차례|개)"
    );
    private static final Pattern COMPARISON = Pattern.compile(
            "여러\\s*회의|회의\\s*(?:간|사이)|(?:이전|지난|첫|다음|최근)\\s*회의.{0,50}(?:비교|변경|달라|바뀌|달랐|바뀐)"
                    + "|(?:비교|달라|바뀌|달랐|바뀐|변화).{0,35}(?:회의|차례)|시간\\s*순"
                    + "|(?:처음|이전|지난|최근|이후).{0,40}(?:이후|최종|현재|달라|바뀌|변화)"
                    + "|어떻게\\s*(?:바뀌었|달라졌)|(?:부터|까지).{0,20}논의.{0,10}모아"
                    + "|회의들|(?:첫|앞|나중|다음)\\s*회의.{0,32}(?:차이|바꾼|바뀐|비교)" + "|(?:두|세|[2-9])\\s*번.{0,32}각각"
                    + "|\\d{1,2}일.{0,24}\\d{1,2}일.{0,20}(?:회의|발표)"
    );
    private static final Pattern COMPARED_DAYS = Pattern.compile(
            "[0-9]{1,2}일\\s*(?:과|와|이랑|랑|하고|및|,)\\s*(?:[0-9]{1,2}월\\s*)?[0-9]{1,2}일"
                    + ".{0,40}(?:각각|비교|같이\\s*보|함께\\s*보)"
    );
    private static final Pattern RECENT_MEETINGS = Pattern
            .compile("(?:최근|지난)\\s*(\\d{1,2}|한|두|세|네|다섯)\\s*(?:번|회(?!의)|차례|건)");
    private static final Pattern ISO_DATE = Pattern.compile("(?<!\\d)\\d{4}-\\d{2}-\\d{2}(?!\\d)");
    private static final Pattern MONTH_DAY = Pattern.compile("(?:(\\d{4})년\\s*)?(\\d{1,2})월\\s*(?:(\\d{1,2})일)?");
    private static final String DATE_LIST = "과|와|이랑|랑|하고|및|,";
    private static final Pattern TRAILING_DAY = Pattern.compile(
            "^\\s*(부터|~|–|-|에서|" + DATE_LIST + ")\\s*(\\d{1,2})일");
    private static final Pattern DATE_CONJUNCTION = Pattern.compile("\\s*(?:" + DATE_LIST + ")\\s*");
    private static final Pattern TEMPORAL_MONTH_SUBJECT = Pattern.compile("^이(?:랑|\\s*(?:지나|되|끝나|시작|오기|다가오))");
    private static final Pattern RELATIVE = Pattern.compile("(?:최근|지난)\\s*(\\d{1,2}|한|두|세|네|다섯)\\s*(일|주|개월|달)");

    public enum Route {
        DOCUMENT_FIND,
        MEETING_QUESTION,
        MULTI_MEETING_SYNTHESIS
    }

    public static SearchHybridQuestion parse(
            String raw,
            Instant reference
    ) {
        String question = Normalizer.normalize(
                raw,
                Normalizer.Form.NFKC
        );
        int count = recentCount(question);
        var days = dates(question, reference.atZone(KOREA).toLocalDate());
        Route initialRoute = baseRoute(question, count);
        boolean inferred = isInferredDateComparison(question, initialRoute);
        List<String> comparisonDays = inferred && days.size() == 2 && days.stream()
                .allMatch(range -> range.from().plus(1, ChronoUnit.DAYS).equals(range.before()))
                ? days.stream().map(range -> range.from().atZone(KOREA).toLocalDate().toString()).distinct().toList()
                : List.of();
        return new SearchHybridQuestion(inferred ? Route.MULTI_MEETING_SYNTHESIS : initialRoute,
                days, count, comparisonDays);
    }

    static Route classifyRoute(String raw) {
        String question = Normalizer.normalize(raw, Normalizer.Form.NFKC);
        Route initialRoute = baseRoute(question, recentCount(question));
        return isInferredDateComparison(question, initialRoute) ? Route.MULTI_MEETING_SYNTHESIS : initialRoute;
    }

    private static boolean isInferredDateComparison(String question, Route initialRoute) {
        return initialRoute == Route.MEETING_QUESTION && COMPARED_DAYS.matcher(question).find();
    }

    private static int recentCount(String question) {
        var recent = RECENT_MEETINGS.matcher(question);
        int count = 0;
        if (recent.find()) {
            count = Math.clamp(
                    number(recent.group(1)),
                    1,
                    10
            );
        }
        if (count == 0 && (question.contains("지난번") || question.contains("가장 최근"))) {
            count = 1;
        }
        return count;
    }

    static boolean isDocumentFind(String question) {
        return DOCUMENT_REQUEST.matcher(Normalizer.normalize(question, Normalizer.Form.NFKC)).find();
    }

    private static Route baseRoute(
            String question,
            int count
    ) {
        if (isDocumentFind(question)) {
            return Route.DOCUMENT_FIND;
        }
        if (count > 1 || MULTIPLE_MEETINGS.matcher(question)
                .find() || COMPARISON.matcher(question)
                        .find()
                || (question.contains("비교") && (question.contains("지난달") || question.contains("지난 달")))) {
            return Route.MULTI_MEETING_SYNTHESIS;
        }
        return Route.MEETING_QUESTION;
    }

    static Long explicitDocumentId(String raw) {
        var matcher = DOCUMENT_ID.matcher(Normalizer.normalize(raw, Normalizer.Form.NFKC));
        if (!matcher.find()) {
            return null;
        }
        String id = matcher.group(1);
        if (matcher.find()) {
            return null;
        }
        try {
            return Long.valueOf(id);
        } catch (NumberFormatException outOfRange) {
            return null;
        }
    }

    private static List<DateRange> dates(
            String question,
            LocalDate day
    ) {
        try {
            var iso = ISO_DATE.matcher(question);
            if (iso.find()) {
                LocalDate first = LocalDate.parse(iso.group());
                int firstEnd = iso.end();
                if (iso.find()) {
                    LocalDate last = LocalDate.parse(iso.group());
                    return combine(days(first, first.plusDays(1)), days(last, last.plusDays(1)),
                            question.substring(firstEnd, iso.start()));
                }
                return List.of(days(first, first.plusDays(1)));
            }
            var calendar = MONTH_DAY.matcher(question);
            while (calendar.find()) {
                String year = calendar.group(1);
                String month = calendar.group(2);
                String tail = question.substring(calendar.end());
                if (calendar.group(3) == null && tail.startsWith("이")
                        && !TEMPORAL_MONTH_SUBJECT.matcher(tail).find()) {
                    continue;
                }
                DateRange first = monthOrDay(
                        year,
                        month,
                        calendar.group(3),
                        day.getYear()
                );
                var trailing = TRAILING_DAY.matcher(tail);
                if (trailing.find()) {
                    DateRange last = monthOrDay(
                            year,
                            month,
                            trailing.group(2),
                            day.getYear()
                    );
                    return combine(first, last, trailing.group(1));
                }
                int firstEnd = calendar.end();
                if (!calendar.find()) {
                    return List.of(first);
                }
                DateRange last = monthOrDay(
                        calendar.group(1),
                        calendar.group(2),
                        calendar.group(3),
                        day.getYear()
                );
                return combine(first, last, question.substring(firstEnd, calendar.start()));
            }
            DateRange relative = relativeDates(question, day);
            return relative.from() == null ? List.of() : List.of(relative);
        } catch (java.time.DateTimeException | IllegalArgumentException invalidDate) {
            throw new SearchToolException("INVALID_TOOL_ARGUMENTS");
        }
    }

    private static List<DateRange> combine(DateRange first, DateRange last, String connector) {
        if (DATE_CONJUNCTION.matcher(connector).matches()) {
            return List.of(first, last);
        }
        return List.of(new DateRange(first.from(), last.before()));
    }

    private static DateRange monthOrDay(
            String year,
            String month,
            String day,
            int referenceYear
    ) {
        int parsedYear = referenceYear;
        if (year != null) {
            parsedYear = Integer.parseInt(year);
        }
        YearMonth value = YearMonth.of(
                parsedYear,
                Integer.parseInt(month)
        );
        if (day == null) {
            return days(
                    value.atDay(1),
                    value.plusMonths(1)
                            .atDay(1)
            );
        }
        LocalDate date = value.atDay(Integer.parseInt(day));
        return days(
                date,
                date.plusDays(1)
        );
    }

    private static DateRange relativeDates(
            String question,
            LocalDate day
    ) {
        var relative = RELATIVE.matcher(question);
        if (relative.find()) {
            int count = number(relative.group(1));
            LocalDate from = switch (relative.group(2)) {
                case "일" -> day.minusDays(count);
                case "주" -> day.minusWeeks(count);
                default -> day.minusMonths(count);
            };
            return days(
                    from,
                    day.plusDays(1)
            );
        }
        LocalDate firstOfMonth = day.withDayOfMonth(1);
        if (question.contains("지난달") || question.contains("지난 달")) {
            LocalDate before = firstOfMonth;
            if (question.contains("이번 달") || question.contains("이번달")) {
                before = firstOfMonth.plusMonths(1);
            }
            return days(
                    firstOfMonth.minusMonths(1),
                    before
            );
        }
        if (question.contains("이번 달") || question.contains("이번달")) {
            return days(
                    firstOfMonth,
                    firstOfMonth.plusMonths(1)
            );
        }
        LocalDate monday = day.minusDays(
                day.getDayOfWeek()
                        .getValue() - 1L
        );
        if (question.contains("지난주") || question.contains("지난 주")) {
            return days(
                    monday.minusWeeks(1),
                    monday
            );
        }
        if (question.contains("이번 주") || question.contains("이번주")) {
            return days(
                    monday,
                    monday.plusWeeks(1)
            );
        }
        if (question.contains("어제")) {
            return days(
                    day.minusDays(1),
                    day
            );
        }
        if (question.contains("오늘")) {
            return days(
                    day,
                    day.plusDays(1)
            );
        }
        return new DateRange(
                null,
                null
        );
    }

    private static int number(String value) {
        return switch (value) {
            case "한" -> 1;
            case "두" -> 2;
            case "세" -> 3;
            case "네" -> 4;
            case "다섯" -> 5;
            default -> Integer.parseInt(value);
        };
    }

    private static DateRange days(
            LocalDate from,
            LocalDate before
    ) {
        return new DateRange(
                from.atStartOfDay(KOREA)
                        .toInstant(),
                before.atStartOfDay(KOREA)
                        .toInstant()
        );
    }

    public record DateRange(
            Instant from,
            Instant before
    ) {
        public DateRange {
            if (from != null && before != null && !from.isBefore(before)) {
                throw new IllegalArgumentException("Reversed date range");
            }
        }
    }
}
