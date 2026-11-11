"""Use spare context tokens for query-related text in already selected documents."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from transformers import PreTrainedTokenizerFast

from build_dataset import SourceDoc
from evidence_expansion import token_count
from experiment import tokens
from phase1 import Hit
from phase3 import Context, material

SHORT_PASSAGE_COUNT = 4
SHORT_PASSAGE_TOKENS = 96


@dataclass(frozen=True, slots=True)
class Section:
    start: int
    end: int
    terms: frozenset[str]


def features(text: str) -> frozenset[str]:
    words = set(tokens(text))
    return frozenset(words | {word[i:i + 2] for word in words for i in range(len(word) - 1)})


def sections(text: str, query: frozenset[str]) -> list[Section]:
    """Return sentences or adjacent sentence pairs with query term overlap."""
    boundaries = [0, *(match.end() for match in re.finditer(r"\n+|[.!?。！？](?=\s|$)", text)), len(text)]
    result = []
    for index, start in enumerate(boundaries[:-1]):
        for end in boundaries[index + 1:index + 3]:
            raw = text[start:end]
            body = raw.strip()
            overlap = features(body) & query
            if not overlap:
                continue
            left = start + len(raw) - len(raw.lstrip())
            result.append(Section(left, left + len(body), overlap))
    return result


def select_short_passages(
    question: str, passages: tuple[Context, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, *, budget: int,
) -> tuple[Context, ...]:
    """Spend reserved tokens on short source excerpts from documents already selected."""
    query = features(question)
    existing: dict[int, list[Hit]] = defaultdict(list)
    for passage in passages:
        existing[passage.hit.documentId].append(passage.hit)
    options: list[tuple[float, Context]] = []
    for doc, windows in existing.items():
        body = source[doc].body
        candidates = sections(body, query)
        frequency = Counter(term for section in candidates for term in section.terms)
        covered = query.intersection(features(" ".join(
            material(source[doc], hit)[hit.start:hit.end] for hit in windows
        )))
        for section in candidates:
            if len(section.terms) < 2 or any(
                hit.field == "body" and hit.start <= section.start and hit.end >= section.end for hit in windows
            ):
                continue
            hit = Hit(documentId=doc, field="body", start=section.start, end=section.end)
            count = token_count(source, tokenizer, hit)
            if 0 < count <= min(budget, SHORT_PASSAGE_TOKENS):
                score = sum((2 if term not in covered else 1) / frequency[term] for term in sorted(section.terms))
                options.append((score, Context(hit=hit, tokenCount=count)))

    selected: list[Context] = []
    documents: set[int] = set()
    options.sort(key=lambda option: (
        -option[0], option[1].tokenCount, option[1].hit.documentId, option[1].hit.start, option[1].hit.end,
    ))
    for _, option in options:
        doc = option.hit.documentId
        if option.tokenCount <= budget and doc not in documents:
            selected.append(option)
            documents.add(doc)
            budget -= option.tokenCount
            if len(selected) == SHORT_PASSAGE_COUNT:
                break
    return tuple(selected)


def extend_same_document(
    question: str, passages: tuple[Context, ...], source: dict[int, SourceDoc],
    tokenizer: PreTrainedTokenizerFast, *, budget: int, max_added: int,
) -> tuple[Context, ...]:
    """Extend an existing body excerpt without removing any of its source text."""
    remaining = budget - sum(item.tokenCount for item in passages)
    query = features(question)
    if remaining < 24 or not query:
        return passages

    by_document: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(passages):
        if item.hit.field == "body":
            by_document[item.hit.documentId].append(index)

    options: list[tuple[float, int, int, Hit]] = []
    limit = min(remaining, max_added)
    for document, indices in by_document.items():
        candidates = sections(source[document].body, query)
        counts = Counter(term for section in candidates for term in section.terms)
        windows = [passages[index].hit for index in indices]
        for section in candidates:
            if any(hit.start <= section.start and hit.end >= section.end for hit in windows):
                continue
            score = sum(1 / counts[term] for term in sorted(section.terms))
            if len(section.terms) < 2:
                continue
            for index in indices:
                item = passages[index]
                hit = item.hit
                if max(section.start - hit.end, hit.start - section.end) > 900:
                    continue
                joined = hit.model_copy(update={"start": min(hit.start, section.start), "end": max(hit.end, section.end)})
                added = token_count(source, tokenizer, joined) - item.tokenCount
                if 0 < added <= limit:
                    options.append((score, added, index, joined))

    extended = list(passages)
    options.sort(key=lambda option: (
        -option[0], option[1], option[3].documentId, option[3].start, option[3].end,
        passages[option[2]].hit.start, passages[option[2]].hit.end,
    ))
    for _, added, index, joined in options:
        item = passages[index]
        if added > remaining or extended[index] != item:
            continue
        if any(other_index != index and extended[other_index].hit.start <= joined.start
               and extended[other_index].hit.end >= joined.end for other_index in by_document[item.hit.documentId]):
            continue
        extended[index] = Context(hit=joined, tokenCount=item.tokenCount + added)
        remaining -= added
    return tuple(extended)
