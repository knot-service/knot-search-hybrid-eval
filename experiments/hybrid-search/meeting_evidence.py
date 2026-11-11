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
) -> Packing:
    """Give compact source windows first claim on the budget, then retain as much surrounding text as fits."""
    unchanged = Packing(surrounding, short)
    if not surrounding or len(short) >= SHORT_PASSAGE_COUNT:
        return unchanged
    covered = (*surrounding, *short)
    candidates = status_windows(question, covered, source, tokenizer)
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
    trimmed, saved = trim_surrounding(question, surrounding, source, tokenizer, -remaining, reclaim)
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


def trim_surrounding(
    question: str, surrounding: tuple[Context, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, needed: int, limit: int,
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
        for start, end in ((separators[0].end(), len(excerpt)), (0, separators[-1].start())):
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
