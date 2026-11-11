"""Add controlled Korean wording variants and register the local Workspace 1 evaluation set."""
from collections import Counter, defaultdict
from datetime import datetime
import json
import re
from zoneinfo import ZoneInfo

from psycopg.types.json import Jsonb

from database import LOCAL, connect

LABELS = {
    "01": "기본", "02": "구어체", "03": "오타", "04": "간투사",
    "05": "담화표지", "06": "군더더기", "07": "혼합형",
}
COMMON_TYPOS = (
    ("어떻게", "어떻케"), ("정했", "정햇"), ("됐", "됬"),
    ("했", "햇"), ("알려줘", "알려주ㅓ"), ("보여줘", "보여주ㅓ"),
    ("정리해", "정리헤"), ("무엇", "무엿"), ("확인", "확읜"),
    ("어떤", "어뗜"), ("인가요", "인까요"), ("뭐야", "뭐얌"),
    ("정책", "정첵"), ("맞아", "맟아"),
)
INTERJECTIONS = ("어, ", "음, ", "아, ", "어어, ")
DISCOURSE = ("근데 ", "그나저나 ", "그러면 ", "그런데 ")
VERBOSITY = (
    "제가 지금 그 부분 하나만 좀 확인하고 싶어서 그러는데, ",
    "다름이 아니라 이것 하나만 좀 물어보고 싶은 건데, ",
    "잠깐, 이거 하나만 좀 짚고 넘어가고 싶어서 그러는데, ",
    "뭐랄까, 제가 확인하고 싶은 게 하나 있어서 그러는데, ",
)


def typo(question: str, index: int) -> tuple[str, str]:
    matches = [(old, new) for old, new in COMMON_TYPOS if old in question]
    if matches:
        old, new = matches[index % len(matches)]
        return question.replace(old, new, 1), "글자 오타"
    spaces = tuple(re.finditer(r"(?<=[가-힣]) (?=[가-힣])", question))
    if spaces:
        position = spaces[index % len(spaces)].start()
        return question[:position] + question[position + 1:], "띄어쓰기 누락"
    return question.rstrip("?!") + "ㅏ" + question[len(question.rstrip("?!")):], "여분의 자모"


def main() -> None:
    path = LOCAL / "workspace1-evaluation-draft.json"
    draft = json.loads(path.read_bytes())
    if (draft["id"], draft["workspaceId"], draft["status"]) != ("knot-workspace-1", 1, "ai_draft"):
        raise SystemExit("Expected the private Workspace 1 evaluation draft.")

    grouped = defaultdict(dict)
    for case in draft["cases"]:
        if case["variant"] in ("01", "02"):
            grouped[case["scenario"]][case["variant"]] = case
    if not grouped or any(set(variants) != {"01", "02"} for variants in grouped.values()):
        raise SystemExit("Each scenario needs its original standard and colloquial variants.")

    expanded = []
    for index, (scenario, original) in enumerate(grouped.items()):
        base = original["01"]
        question = base["question"]
        misspelled, error = typo(question, index)
        interjection = INTERJECTIONS[index % len(INTERJECTIONS)]
        discourse = DISCOURSE[(index // 2) % len(DISCOURSE)]
        redundant = VERBOSITY[(index // 3) % len(VERBOSITY)]
        variants = {
            "01": (question, "표준 표현"),
            "02": (original["02"]["question"], "구어체 표현"),
            "03": (misspelled, error),
            "04": (interjection + question, "머뭇거림·감탄"),
            "05": (discourse + question, "화제 전환·연결 표현"),
            "06": (redundant + question, "불필요한 도입 표현"),
            "07": (interjection + discourse + redundant + misspelled, f"간투사·담화표지·군더더기·{error}"),
        }
        if len({text for text, _ in variants.values()}) != len(variants):
            raise SystemExit(f"Duplicate wording variants in {scenario}.")
        for variant, (text, detail) in variants.items():
            expanded.append({**base, "id": f"{scenario}-{variant}", "variant": variant,
                             "question": text, "variantLabel": LABELS[variant], "variantDetail": detail})

    draft["version"] = "0.2-draft"
    draft["cases"] = expanded
    draft["variantCounts"] = dict(Counter(case["variantLabel"] for case in expanded))
    draft["extendedAt"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

    with connect() as db:
        db.execute(
            """create table if not exists evaluation_datasets (
                dataset_id text primary key,
                workspace_id integer not null references workspaces(id),
                version text not null,
                artifact jsonb not null
            )""",
        )
        db.execute(
            """insert into evaluation_datasets values (%s,1,%s,%s)
               on conflict(dataset_id) do update
               set version=excluded.version, artifact=excluded.artifact""",
            (draft["id"], draft["version"], Jsonb(draft)),
        )
        row = db.execute(
            "select artifact from evaluation_datasets where dataset_id=%s and workspace_id=1",
            (draft["id"],),
        ).fetchone()
        if row is None or row[0] != draft:
            raise SystemExit("The registered dataset does not match the generated cases.")

    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(draft, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)
    print(f"registered {draft['id']}@{draft['version']} scenarios={len(grouped)} cases={len(expanded)}")
    print(json.dumps(draft["variantCounts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
