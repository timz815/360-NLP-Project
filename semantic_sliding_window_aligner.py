#for NLP Project

"""
Many to many (paragraph-level) semantic sliding window alignment (Chinese ↔ English)

Pipeline:
1. Split Chinese and English text into sentences (spaCy)
2. Build sliding windows of 1–5 sentences for both EN and ZH
3. Translate each EN window into ZH (MT: Helsinki opus-mt-zh-en)
4. Encode translated EN windows and ZH windows with SBERT
5. Compute cosine similarity between all window pairs
6. Select high-similarity matches (greedy alignment)
7. Remove sentence overlaps (deduplication)
8. Merge nearby matches into bigger chunks

Output: aligned_windows.jsonl
"""


# Dependencies:
# - sacremoses
# - torch
# - torchvision
# - torchaudio
# - transformers
# - sentence-transformers
# - tf-keras
# - sentencepiece
# - spacy
"""
Structure:

 script.py
 movie folder 1
 ├── zh.txt (input)
 ├── en.txt  (input)
 └── aligned_timestamps.jsonl (output)
 movie folder 2
 ├── zh.txt (input)
 ├── en.txt  (input)
 └── aligned_timestamps.jsonl (output)
"""
import os
import sys, torch, json, re
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ================================================
# -------------------- config --------------------
# ================================================
TAU = 0.75
MAX_WINDOW = 5
BATCH = 32

# ================================================
# -------------------- models --------------------
# ================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

nlp_zh = spacy.load("zh_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])

for pipe in (nlp_zh, nlp_en):
    if "sentencizer" not in pipe.pipe_names:
        pipe.add_pipe("sentencizer")

tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
mt  = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(device)

sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# ==========================================================
# -------------------- subtitle cleanup --------------------
# ==========================================================
def clean_subtitles(text):
    """
    Clean SRT format (adapted for movie subtitles)
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        original = line
        line = line.strip()
        if not line:
            continue

        # Skip subtitle number lines
        if re.fullmatch(r"\d+", line):
            continue

        # Skip timestamp lines
        if re.fullmatch(
            r"\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}", line
        ):
            continue

        # Remove leading dash (dialogue indicator), but keep content
        line = re.sub(r"^-+\s*", "", line)

        # Remove all {...} formatting blocks
        line = re.sub(r"\{.*?\}", "", line)

        # Remove all <...> HTML-like tags
        line = re.sub(r"<.*?>", "", line)

        # Remove ALL [ ... ] cues (one or multiple)
        line = re.sub(r"\[.*?\]", "", line)

        # Strip whitespace
        line = line.strip()

        if line:
            cleaned.append(line)

    return "\n".join(cleaned)

def ensure_terminal_punct(text: str, lang: str) -> str:
    """
    Add a sentence punctuation mark if missing
    lang = 'zh' or 'en'
    """
    en_stops = {'.', '!', '?', '…'}
    zh_stops = {'。', '！', '？', '…'}
    stops = en_stops | zh_stops if lang == 'zh' else en_stops

    lines_out = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            lines_out.append(line)
            continue
        if line[-1] not in stops:
            line += '。' if lang == 'zh' else '.'
        lines_out.append(line)
    return '\n'.join(lines_out)

# =================================================
# -------------------- helpers --------------------
# =================================================
def encode(texts):
    return sbert.encode(texts, convert_to_tensor=True, device=device)

def window_spans(sents, max_w):
    for w in range(1, max_w + 1):
        for i in range(len(sents) - w + 1):
            text = " ".join(sents[i:i + w])
            yield (0, 0, text, i, i + w)

# ===================================================
# -------------------- alignment --------------------
# ===================================================
def align_windows(en_windows, zh_windows, tau=0.8):
    """
    English alignment:
    - Translate English windows → Chinese
    - Embed translated English and Chinese
    - Cosine similarity ≥ tau
    """
    en_texts = [w[2] for w in en_windows]
    zh_texts = [w[2] for w in zh_windows]

    # translate EN → ZH
    translated = []
    for i in range(0, len(en_texts), BATCH):
        batch = en_texts[i:i+BATCH]
        inp = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = mt.generate(**inp, num_beams=4, max_length=256)
        translated.extend(tok.batch_decode(out, skip_special_tokens=True))

    en_emb = encode(translated)
    zh_emb = encode(zh_texts)

    sim = util.cos_sim(en_emb, zh_emb)

    pairs = [
        (i, j, sim[i, j].item())
        for i in range(sim.shape[0])
        for j in range(sim.shape[1])
        if sim[i, j].item() >= tau
    ]

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Candidate alignments: {len(pairs)}")

    return pairs

# ==============================================================
# -------------------- sentence-level dedup --------------------
# ==============================================================
def dedup_sentences(pairs, zh_windows, en_windows):
    """
    Greedy sentence-level deduplication:
    Each sentence index may appear only once per side
    """
    used_zh = set()
    used_en = set()
    survivors = []

    for wi, wj, score in pairs:
        zh_i0, zh_i1 = zh_windows[wj][3], zh_windows[wj][4]
        en_i0, en_i1 = en_windows[wi][3], en_windows[wi][4]

        zh_range = set(range(zh_i0, zh_i1))
        en_range = set(range(en_i0, en_i1))

        # keep only if none of the sentences have been used before
        if zh_range.isdisjoint(used_zh) and en_range.isdisjoint(used_en):
            survivors.append((wi, wj, score))
            used_zh.update(zh_range)
            used_en.update(en_range)

    return survivors

# ==========================================================
# -------------------- merge contiguous --------------------
# ==========================================================
def merge_contiguous(pairs):
    """
    Merge pairs with consecutive window indices on both sides
    """
    if not pairs:
        return pairs

    pairs = sorted(pairs, key=lambda x: (x[1], x[0]))
    merged = [pairs[0]]

    for wi, wj, sc in pairs[1:]:
        last_wi, last_wj, last_sc = merged[-1]

        if wi == last_wi + 1 and wj == last_wj + 1:
            merged[-1] = (last_wi, last_wj, last_sc)
        else:
            merged.append((wi, wj, sc))

    return merged

# ===============================================
# -------------------- input --------------------
# ===============================================
def get_text(prompt):
    """
    Get user to input raw text in idle
    """
    print(prompt)
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines).strip()

# ==============================================
# -------------------- main --------------------
# ==============================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Scanning:", base_dir)

    folders = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not folders:
        print("No folders found.")
        return

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        zh_path = os.path.join(folder_path, "zh.txt")
        en_path = os.path.join(folder_path, "en.txt")

        print(f"\n---- Folder: {folder} ----")

        if not (os.path.exists(zh_path) and os.path.exists(en_path)):
            print("Skipping → missing zh.txt or en.txt")
            continue

        print("Found zh.txt & en.txt → running alignment...")

        # read files
        with open(zh_path, "r", encoding="utf-8") as f:
            zh_text = f.read().strip()

        with open(en_path, "r", encoding="utf-8") as f:
            en_text = f.read().strip()

        if not zh_text or not en_text:
            print("Skipping → one file is empty.")
            continue

        # clean
        zh_text = ensure_terminal_punct(clean_subtitles(zh_text), 'zh')
        en_text = ensure_terminal_punct(clean_subtitles(en_text), 'en')

        # process
        zh_sents = [line.strip() for line in zh_text.splitlines() if line.strip()]
        en_sents = [line.strip() for line in en_text.splitlines() if line.strip()]

        print(f"ZH sentences: {len(zh_sents)} | EN sentences: {len(en_sents)}")

        zh_windows = list(window_spans(zh_sents, MAX_WINDOW))
        en_windows = list(window_spans(en_sents, MAX_WINDOW))

        pairs = align_windows(en_windows, zh_windows, TAU)
        pairs = dedup_sentences(pairs, zh_windows, en_windows)
        final_pairs = merge_contiguous(pairs)

        # output path
        out_path = os.path.join(folder_path, "aligned_windows.jsonl")
        print("Writing:", out_path)

        with open(out_path, "w", encoding="utf-8") as f:
            for wi, wj, score in final_pairs:
                zh_start, zh_end, zh_txt, zh_i0, zh_i1 = zh_windows[wj]
                en_start, en_end, en_txt, en_i0, en_i1 = en_windows[wi]

                f.write(json.dumps({
                    "chinese_span": zh_txt,
                    "english_span": en_txt,
                    "score": round(score, 3),
                    "chinese_start": zh_start,
                    "chinese_end": zh_end,
                    "english_start": en_start,
                    "english_end": en_end,
                    "chinese_sent_indices": list(range(zh_i0, zh_i1)),
                    "english_sent_indices": list(range(en_i0, en_i1))
                }, ensure_ascii=False) + "\n")

        print(f"Done → {len(final_pairs)} alignments")

    print("\nAll folders processed.")

if __name__ == "__main__":
    main()
