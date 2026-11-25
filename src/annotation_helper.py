"""
Công cụ hỗ trợ đọc từng đoạn ứng viên để gắn nhãn thủ công 2 vòng
và kiểm chứng exact substring cho evidence_sentence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from text_utils import split_sentences_vi

ROOT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT_DIR / "data" / "processed"
ANN_DIR = ROOT_DIR / "data" / "annotations"


def dump_batches() -> None:
    df = pd.read_csv(PROC_DIR / "annotation_pool.csv", encoding="utf-8-sig")
    ranges = [(0, 70, 1), (70, 140, 2), (140, 210, 3), (210, len(df), 4)]
    for s, e, b in ranges:
        lines = []
        for i in range(s, e):
            r = df.iloc[i]
            sents = split_sentences_vi(r["text"])
            lines.append(f"=== {i} | {r['item_id']} | {r['source_name']} | {str(r['published_at'])[:10]} ===")
            lines.append(f"TITLE: {r['title']}")
            for j, st in enumerate(sents):
                lines.append(f"  [{j}] {st}")
        out = PROC_DIR / f"_batch{b}.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out.name}: rows {s}..{e-1}")


def dump_pass2_blind() -> None:
    """
    Chọn ngẫu nhiên phân tầng 75 mẫu (65 VietOnlineNews + 10 UVN-1, ~26.8% tổng mẫu)
    và xuất file chỉ chứa văn bản (KHÔNG chứa nhãn Pass 1) để đọc hậu kiểm độc lập.
    """
    df = pd.read_csv(PROC_DIR / "annotation_pool.csv", encoding="utf-8-sig")
    von = df[df["dataset_source"] == "VietOnlineNews"].sample(n=65, random_state=2026)
    uvn = df[df["dataset_source"] == "UVN-1"].sample(n=10, random_state=2026)
    sub = pd.concat([von, uvn], ignore_index=True).sort_values("item_id").reset_index(drop=True)

    lines = []
    p2_ids = []
    for i, r in sub.iterrows():
        sents = split_sentences_vi(r["text"])
        p2_ids.append(r["item_id"])
        lines.append(f"=== P2_{i} | {r['item_id']} | {r['source_name']} ===")
        lines.append(f"TITLE: {r['title']}")
        for j, st in enumerate(sents):
            lines.append(f"  [{j}] {st}")
    (PROC_DIR / "_pass2_blind.txt").write_text("\n".join(lines), encoding="utf-8")
    (PROC_DIR / "_pass2_ids.json").write_text(json.dumps(p2_ids, indent=2), encoding="utf-8")
    print(f"Wrote _pass2_blind.txt with {len(p2_ids)} blind items.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--pass2", action="store_true")
    args = parser.parse_args()
    if args.dump:
        dump_batches()
    if args.pass2:
        dump_pass2_blind()


if __name__ == "__main__":
    main()
