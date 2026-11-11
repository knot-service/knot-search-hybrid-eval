"""Reserve space for one additional status-bearing source window per represented meeting."""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Final

from transformers import PreTrainedTokenizerFast

from build_dataset import SourceDoc
from evidence_expansion import token_count
from experiment import tokens
from phase1 import Hit
from phase3 import Context, material
from same_document_evidence import SHORT_PASSAGE_COUNT, SHORT_PASSAGE_TOKENS, features, sections

STATUS: Final = re.compile(r"미정|미확정|미결정|보류|합의|확정|결정|결론|채택|제외|유지|잠정|검토|정하기로|하기로|않기로")
EDGE: Final = re.compile(r"\n+|[.!?。！？](?=\s|$)")

TOPIC_STOP: Final = frozenset("""
회의 첫 다음 이전 이후 최근 두 세 각 각각 차이 비교 이번 정리 관련 진행 방식 상태 상황 결정 미정
도입 여부 내용 논의 결과 확인 변경 어떻게 어떻케 뭐가 뭐였어 무엇 어떤 알려줘 말해줘 설명해줘 이것 하나만
다름 아니라 좀 더 따로 건데 물어보고 싶은 줘 해줘 있으면 없으면 전후 년 월 일
""".split())
PARTICLE: Final = re.compile(r"(?:으로|에서|까지|부터|에게|이나|라는|이라|보다|하고|이랑|[은는이가을를과와도만의로])$")


@dataclass(frozen=True, slots=True)
class Packing:
    passages: tuple[Context, ...]
    short: tuple[Context, ...]


def reserve_meeting_evidence(
    question: str,
    surrounding: tuple[Context, ...],
    short: tuple[Context, ...],
    source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast,
    *,
    budget: int,
    reclaim: int,
    require_topic_status: bool = False,
) -> Packing:
    """Give compact source windows first claim on the budget, then retain as much surrounding text as fits."""
    unchanged = Packing(surrounding, short)
    if not surrounding or len(short) >= SHORT_PASSAGE_COUNT:
        return unchanged
    covered = (*surrounding, *short)
    candidates = (
        topic_status_windows(question, covered, source, tokenizer)
        if require_topic_status else status_windows(question, covered, source, tokenizer)
    )
    if not candidates:
        return unchanged
    remaining = budget - sum(item.tokenCount for item in covered)
    spare = remaining
    reserved: list[Context] = []
    meetings: set[int] = set()
    for candidate in candidates:
        meeting = source[candidate.hit.documentId].meetingId
        if meeting in meetings or candidate.tokenCount > remaining + reclaim:
            continue
        reserved.append(candidate)
        meetings.add(meeting)
        remaining -= candidate.tokenCount
        if len(reserved) + len(short) == SHORT_PASSAGE_COUNT:
            break
    if not reserved:
        return unchanged
    if remaining >= 0:
        return Packing(surrounding, (*short, *reserved))

    # Reclaim only complete peripheral sentences with no question or status overlap.
    trimmed, saved = trim_surrounding(
        question, surrounding, source, tokenizer, -remaining, reclaim, preserve_punctuation=require_topic_status,
    )
    available = remaining + saved + sum(item.tokenCount for item in reserved)
    accepted: list[Context] = []
    for candidate in reserved:
        if candidate.tokenCount <= available:
            accepted.append(candidate)
            available -= candidate.tokenCount
    if not accepted:
        return unchanged
    if sum(item.tokenCount for item in accepted) <= spare:
        return Packing(surrounding, (*short, *accepted))
    return Packing(trimmed, (*short, *accepted))


def status_windows(
    question: str, covered: tuple[Context, ...], source: dict[int, SourceDoc], tokenizer: PreTrainedTokenizerFast,
) -> tuple[Context, ...]:
    """Find complete sentences near matching topic words and decision or uncertainty language."""
    query = features(question)
    words = set(tokens(question))
    by_document: dict[int, list[Hit]] = {}
    for item in covered:
        by_document.setdefault(item.hit.documentId, []).append(item.hit)
    options: list[tuple[float, Context]] = []
    for document, existing in by_document.items():
        body = source[document].body
        spans = sections(body, query)
        frequency = Counter(term for span in spans for term in span.terms)
        for span in spans:
            if len(span.terms) < 2 or any(
                hit.field == "body" and hit.start <= span.start and hit.end >= span.end for hit in existing
            ):
                continue
            excerpt = body[span.start:span.end]
            status = tuple(STATUS.finditer(excerpt))
            if not any(not any(
                hit.field == "body" and hit.start <= span.start + cue.start() and span.start + cue.end() <= hit.end
                for hit in existing
            ) for cue in status):
                continue
            hit = Hit(documentId=document, field="body", start=span.start, end=span.end)
            count = token_count(source, tokenizer, hit)
            if not 0 < count <= SHORT_PASSAGE_TOKENS:
                continue
            score = sum((2 if term in words else 1) / frequency[term] for term in sorted(span.terms))
            options.append((score, Context(hit=hit, tokenCount=count)))
    options.sort(key=lambda pair: (-pair[0], pair[1].tokenCount, pair[1].hit.documentId, pair[1].hit.start))
    return tuple(candidate for _, candidate in options)


def topic_words(text: str) -> tuple[str, ...]:
    """Keep meaningful words, normalizing common Korean particles on longer terms."""
    words = (PARTICLE.sub("", word) if len(word) >= 3 else word for word in tokens(text))
    return tuple(word for word in words if len(word) >= 2 and not word[0].isdigit() and word not in TOPIC_STOP)


def topic_status_windows(
    question: str, covered: tuple[Context, ...], source: dict[int, SourceDoc], tokenizer: PreTrainedTokenizerFast,
) -> tuple[Context, ...]:
    """Require a decision or uncertainty cue in the sentence that names the requested topic."""
    query = topic_words(question)
    terms = set(query)
    pairs = set(zip(query, query[1:]))
    if not terms:
        return ()
    by_document: dict[int, list[Hit]] = {}
    for item in covered:
        by_document.setdefault(item.hit.documentId, []).append(item.hit)
    options: list[tuple[float, Context]] = []
    for document, existing in by_document.items():
        body = source[document].body
        # Paragraph headings and neighboring sentences cannot lend their status to an unrelated sentence.
        boundaries = (0, *(m.end() for m in EDGE.finditer(body)), len(body))
        for left, right in zip(boundaries, boundaries[1:]):
            raw = body[left:right]
            excerpt = raw.strip()
            status = STATUS.search(excerpt)
            if status is None:
                continue
            words = topic_words(excerpt)
            matched = terms.intersection(words)
            matched_pairs = pairs.intersection(zip(words, words[1:]))
            if len(matched) < min(2, len(terms)):
                continue
            start = left + len(raw) - len(raw.lstrip())
            end = start + len(excerpt)
            if any(hit.field == "body" and hit.start <= start and end <= hit.end for hit in existing):
                continue
            hit = Hit(documentId=document, field="body", start=start, end=end)
            count = token_count(source, tokenizer, hit)
            if not 0 < count <= SHORT_PASSAGE_TOKENS:
                continue
            score = 6 * len(matched_pairs) + len(matched) + len(matched) / len(set(words))
            options.append((score, Context(hit=hit, tokenCount=count)))
    options.sort(key=lambda pair: (-pair[0], pair[1].tokenCount, pair[1].hit.documentId, pair[1].hit.start))
    return tuple(candidate for _, candidate in options)


def trim_surrounding(
    question: str, surrounding: tuple[Context, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, needed: int, limit: int, *, preserve_punctuation: bool = False,
) -> tuple[tuple[Context, ...], int]:
    """Remove low overlap edges without removing any representative or reducing it below one short window."""
    query = set(tokens(question))
    available = list(surrounding)
    saved = 0
    for index in reversed(range(len(surrounding))):
        item = surrounding[index]
        hit = item.hit
        if hit.field != "body" or item.tokenCount <= SHORT_PASSAGE_TOKENS:
            continue
        excerpt = material(source[hit.documentId], hit)[hit.start:hit.end]
        separators = list(EDGE.finditer(excerpt))
        if not separators:
            continue
        last = separators[-1]
        tail_start = last.end() if preserve_punctuation and last.group()[0] in ".!?。！？" else last.start()
        for start, end in ((separators[0].end(), len(excerpt)), (0, tail_start)):
            discarded = excerpt[:start] + excerpt[end:]
            if STATUS.search(discarded) or query.intersection(tokens(discarded)):
                continue
            crop = hit.model_copy(update={"start": hit.start + start, "end": hit.start + end})
            count = token_count(source, tokenizer, crop)
            cost = item.tokenCount - count
            if count >= SHORT_PASSAGE_TOKENS and 0 < cost <= limit - saved:
                available[index] = Context(hit=crop, tokenCount=count)
                saved += cost
                break
        if saved >= needed:
            break
    return tuple(available), saved
