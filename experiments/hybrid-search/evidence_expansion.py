"""Add ranked source passages while preserving the original evidence."""
from collections import Counter
from typing import Literal

from transformers import PreTrainedTokenizerFast

from build_dataset import SourceDoc
from phase1 import Hit
from phase3 import Context, assemble_worker_context, material


def compact_overlaps(items: tuple[Context, ...], source: dict[int, SourceDoc],
                     tokenizer: PreTrainedTokenizerFast) -> tuple[Context, ...]:
    compacted = []
    for item in items:
        hit = item.hit
        for index, previous in enumerate(compacted):
            old = previous.hit
            if (hit.documentId, hit.field) != (old.documentId, old.field) or max(hit.start, old.start) > min(hit.end, old.end):
                continue
            joined = hit.model_copy(update={"start": min(old.start, hit.start), "end": max(old.end, hit.end)})
            count = token_count(source, tokenizer, joined)
            if count <= previous.tokenCount + item.tokenCount:
                compacted[index] = Context(hit=joined, tokenCount=count)
                break
        else:
            compacted.append(item)
    return tuple(compacted)


def token_count(source: dict[int, SourceDoc], tokenizer: PreTrainedTokenizerFast, hit: Hit) -> int:
    text = material(source[hit.documentId], hit)
    return len(tokenizer(text[hit.start:hit.end], add_special_tokens=False, verbose=False)["input_ids"])


def expand_evidence(
    question: str, contexts: tuple[Context, ...], supports: tuple[Context, ...],
    candidates: tuple[Hit, ...], source: dict[int, SourceDoc], tokenizer: PreTrainedTokenizerFast,
    *, budget: int, excerpt_tokens: int, passages_per_kind: int = 5, supports_per_meeting: int = 2,
    compact: bool = False, order: Literal["rank", "documents"] = "rank",
) -> tuple[tuple[Context, ...], tuple[Context, ...]]:
    primary = list(compact_overlaps(contexts, source, tokenizer) if compact else contexts)
    supporting = list(supports)
    primary_docs = {item.hit.documentId for item in primary}
    meetings = {source[doc].meetingId for doc in primary_docs}
    supporting_docs = {item.hit.documentId for item in supporting}
    support_counts = Counter(source[doc].meetingId for doc in supporting_docs)
    remaining = budget - sum(item.tokenCount for item in (*primary, *supporting))
    if order == "documents":
        candidates = sorted(candidates, key=lambda hit: hit.documentId in primary_docs | supporting_docs)
    for hit in candidates:
        if remaining < 24:
            break
        document = hit.documentId
        meeting = source[document].meetingId
        if hit.field != "body" or meeting not in meetings:
            continue
        existing = [(group, index, item) for group in (primary, supporting) for index, item in enumerate(group)
                    if item.hit.documentId == document and item.hit.field == hit.field]
        if any(item.hit.start <= hit.start and item.hit.end >= hit.end for _, _, item in existing):
            continue
        # Prefer a contiguous expansion when the selected window meets this candidate.
        overlap = [(group, index, item) for group, index, item in existing
                   if hit.start <= item.hit.end and item.hit.start <= hit.end]
        for group, index, item in overlap:
            old = item.hit
            joined = hit.model_copy(update={"start": min(old.start, hit.start), "end": max(old.end, hit.end)})
            count = token_count(source, tokenizer, joined)
            if count - item.tokenCount <= min(remaining, excerpt_tokens):
                group[index] = Context(hit=joined, tokenCount=count)
                remaining -= count - item.tokenCount
                break
        else:
            if overlap or document in supporting_docs:
                continue
            if document in primary_docs:
                group = primary
            elif support_counts[meeting] < supports_per_meeting:
                group = supporting
            else:
                continue
            if len(group) >= passages_per_kind:
                continue
            added = assemble_worker_context(
                (hit,), source, tokenizer, {}, budget=min(remaining, excerpt_tokens), focus=question,
            )
            if not added or any(item.hit.start <= added[0].hit.start and item.hit.end >= added[0].hit.end
                                for _, _, item in existing):
                continue
            group.extend(added)
            remaining -= added[0].tokenCount
            if group is supporting:
                supporting_docs.add(document)
                support_counts[meeting] += 1
    return tuple(primary), tuple(supporting)
