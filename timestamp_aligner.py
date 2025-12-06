# For NLP Project

"""
LaBSE timing alignment  Chinese ↔ English  (no MT or spaCy)
for movie subtitles with SRT format
Output: aligned_windows.jsonl
"""

"""
Structure:

 script.py
 movie folder 1
 ├── zh.txt (input)
 ├── en.txt  (input)
 └── aligned_windows.jsonl (output)
 movie folder 2
 ├── zh.txt (input)
 ├── en.txt  (input)
 └── aligned_windows.jsonl (output)
"""

import sys, json, re, math, os
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

# ============================
# ---------- config ----------
# ============================
MIN_IOU   = 0.30 # temporal overlap
MIN_COS   = 0.70 # LaBSE cosine
BATCH     = 64
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# ---------- helpers ----------
# =============================
def read_stdin(prompt: str) -> str:
    print(prompt, file=sys.stderr)
    return sys.stdin.read().strip()

def clean_srt(raw: str) -> str:
    """Remove numbering, time-codes, [] sound cues, ,<i>, etc."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+", line):      # remove srt subtitle index
            continue
        if "-->" in line:                   # remove time line
            continue
        if re.fullmatch(r"\[.*\]", line):   # remove []
            continue
        line = re.sub(r"<[^>]+>", "", line) # remove <i>, <b>, </i>, etc.
        if line:
            lines.append(line)
    return " ".join(lines)

def make_blocks(text: str) -> List[Tuple[int, int, str]]:
    """
    pair subtitles if gap ≤ 1 s.
    Returns list of (start_ms, end_ms, text)
    """
    entries = [] # (start_ms, end_ms, text)
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3: # index / time / text
            continue
        time_line = lines[1]
        m = re.search(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})", time_line)
        if not m:
            continue
        def ms(h, m, s, c): return int(h)*3_600_000 + int(m)*60_000 + int(s)*1_000 + int(c)
        start = ms(*m.groups()[:4])
        end   = ms(*m.groups()[4:])
        txt   = " ".join(lines[2:]).strip()
        entries.append((start, end, txt))

    if not entries:
        return [(0, 0, clean_srt(text))] # fallback for plain text

    # merge
    merged = []
    cur_start, cur_end, cur_txt = entries[0]
    for start, end, txt in entries[1:]:
        if start - cur_end <= 1_000: # 1 s gap
            cur_end = end
            cur_txt += " " + txt
        else:
            merged.append((cur_start, cur_end, cur_txt))
            cur_start, cur_end, cur_txt = start, end, txt
    merged.append((cur_start, cur_end, cur_txt))
    return merged

def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a
    s2, e2 = b
    overlap = max(0, min(e1, e2) - max(s1, s2))
    union   = max(e1, e2) - min(s1, s2)
    return overlap / union if union else 0.0

# ==========================
# ---------- main ----------
# ==========================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Scanning:", base_dir, file=sys.stderr)

    folders = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]

    if not folders:
        print("No folders found.", file=sys.stderr)
        return

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        zh_path = os.path.join(folder_path, "zh.txt")
        en_path = os.path.join(folder_path, "en.txt")

        print(f"\n---- Folder: {folder} ----", file=sys.stderr)

        if not (os.path.isfile(zh_path) and os.path.isfile(en_path)):
            print("Skipping → missing zh.srt or en.srt", file=sys.stderr)
            continue

        with open(zh_path, encoding="utf-8") as f:
            zh_raw = f.read()
        with open(en_path, encoding="utf-8") as f:
            en_raw = f.read()

        zh_blocks = make_blocks(zh_raw)
        en_blocks = make_blocks(en_raw)

        print(f"ZH blocks: {len(zh_blocks)} | EN blocks: {len(en_blocks)}", file=sys.stderr)

        model = SentenceTransformer("sentence-transformers/LaBSE").to(DEVICE)
        zh_embs = model.encode([b[2] for b in zh_blocks], convert_to_tensor=True, device=DEVICE)
        en_embs = model.encode([b[2] for b in en_blocks], convert_to_tensor=True, device=DEVICE)

        cos_scores = util.cos_sim(en_embs, zh_embs)

        pairs = []
        for i, (en_start, en_end, _) in enumerate(en_blocks):
            for j, (zh_start, zh_end, _) in enumerate(zh_blocks):
                if iou((en_start, en_end), (zh_start, zh_end)) >= MIN_IOU and cos_scores[i, j] >= MIN_COS:
                    pairs.append((i, j, cos_scores[i, j].item()))

        pairs.sort(key=lambda x: x[2], reverse=True)
        used_en, used_zh = set(), set()
        final = []
        for i, j, score in pairs:
            if i not in used_en and j not in used_zh:
                final.append((i, j, score))
                used_en.add(i); used_zh.add(j)

        out_path = os.path.join(folder_path, "aligned_timestamps.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, j, score in final:
                en_block = en_blocks[i]
                zh_block = zh_blocks[j]
                f.write(json.dumps({
                    "start_ms": max(en_block[0], zh_block[0]),
                    "end_ms"  : min(en_block[1], zh_block[1]),
                    "iou"     : iou((en_block[0], en_block[1]), (zh_block[0], zh_block[1])),
                    "cosine"  : round(score, 3),
                    "zh_text" : zh_block[2],
                    "en_text" : en_block[2]
                }, ensure_ascii=False) + "\n")

        print(f"Done. {len(final)} pairs → {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
