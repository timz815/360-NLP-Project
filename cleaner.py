# adapted for aligned_timestamps.jsonl
# see timestamp_aligner.py
# originally for semantic_sliding_window_aligner.py

"""
cleaner.py

Overview:
- Find all folders in same location
- process aligned_timestamps.jsonl in each folder
- Outputs in each folder:
  movie_dialogue.jsonl (output, ready for finetuning)
      kept pairs in {"english": "...", "chinese": "..."} format
  rejected.jsonl (output)

Pipeline:
  - split sentence, both sides (pairs) with spaCy
  - early reject using input cosine threshold before SBERT
  - SBERT veto using tau and length-ratio checks to clean dataset

"""

import os
import sys
import json
import math
import glob
import torch
from typing import List
from sentence_transformers import SentenceTransformer, util
import spacy

# ================================================
# -------------------- config --------------------
# ================================================
INPUT_FILENAME = "aligned_timestamps.jsonl"
OUTPUT_FILENAME = "movie_dialogue.jsonl"
REJECTED_FILENAME = "rejected.jsonl"

# timing cosine early-reject threshold
TIMING_COSINE_THRESHOLD = 0.65

# semantic cleaning thresholds
TAU_CLEAN = 0.68 # minimum SBERT similarity for a "match"
MAX_RATIO = 3.0  # max length ratio allowed
MIN_RATIO = 1.0 / MAX_RATIO


# ================================================
# -------------------- models --------------------
# ================================================
SBERT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load SBERT
sbert = SentenceTransformer(SBERT_MODEL, device=device)

# load spaCy sentencizers
nlp_zh = spacy.load("zh_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])

for p in (nlp_zh, nlp_en):
    if "sentencizer" not in p.pipe_names:
        p.add_pipe("sentencizer")

# ===============================================
# ------------------ functions ------------------
# ===============================================
def normalize_text(s: str) -> str:
    """basic whitespace normalization and strip"""
    if s is None:
        return ""
    s = s.strip()
    # remove leading dash bullet if present
    if s.startswith("-"):
        s = s.lstrip("-").strip()
    # collapse multiple spaces
    s = " ".join(s.split())
    return s

def sent_split(text: str, nlp) -> List[str]:
    """split text into sentences using spaCy sentencizer"""
    if not text:
        return []
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

@torch.no_grad()
def encode(texts):
    """encode texts to SBERT embeddings"""
    if not texts:
        # return empty tensor shaped (0, dim)
        return sbert.encode([], convert_to_tensor=True, device=device)
    return sbert.encode(texts, convert_to_tensor=True, device=device)


def veto(cn_sents: List[str], en_sents: List[str]) -> (bool, dict):
    """apply veto rules
    return (keep_bool, debug_info)
    """
    debug = {}
    if not cn_sents or not en_sents:
        debug["fail"] = "empty_side"
        return False, debug

    cn_emb = encode(cn_sents)
    en_emb = encode(en_sents)

    sim = util.cos_sim(cn_emb, en_emb)  # (len(cn), len(en))

    # every CN must have ≥ TAU_CLEAN pair and pass length ratio test
    for i in range(sim.shape[0]):
        best = sim[i].max().item()
        best_j = int(sim[i].argmax().item())
        debug.setdefault("cn_best", []).append({"i": i, "best_sim": best, "best_j": best_j})
        if best < TAU_CLEAN:
            debug["fail"] = f"cn_no_good_match_i{i}"
            return False, debug

        # length ratio check: compare char lengths for cn and word count for en best match
        cn_len = len(cn_sents[i])
        en_len_words = max(1, len(en_sents[best_j].split()))
        ratio = cn_len / en_len_words
        debug.setdefault("ratios", []).append(ratio)
        if ratio > MAX_RATIO or ratio < MIN_RATIO:
            debug["fail"] = f"cn_en_ratio_out_of_bounds_i{i}_ratio{ratio:.2f}"
            return False, debug

    # every EN must have ≥ TAU_CLEAN pair
    for j in range(sim.shape[1]):
        best = sim[:, j].max().item()
        debug.setdefault("en_best", []).append({"j": j, "best_sim": best})
        if best < TAU_CLEAN:
            debug["fail"] = f"en_no_good_match_j{j}"
            return False, debug

    # passed all checks
    debug["pass"] = True
    return True, debug


def process_all_folders(base_path: str):
    """Find all folders in base_path and process each one, similar to timestamp aligner"""
    if not os.path.exists(base_path):
        print(f"Base path not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    
    folders = [f for f in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, f))]
    
    print(f"Found {len(folders)} folders to process", file=sys.stderr)
    
    for folder_name in folders:
        folder_path = os.path.join(base_path, folder_name)
        input_file = os.path.join(folder_path, INPUT_FILENAME)
        
        if not os.path.exists(input_file):
            print(f"Skipping {folder_name}: {INPUT_FILENAME} not found", file=sys.stderr)
            continue
            
        print(f"\nProcessing folder: {folder_name}", file=sys.stderr)
        process_file(input_file, folder_path)


def process_file(input_path: str, output_folder: str):
    """read input file and produce kept and rejected outputs in the given folder"""
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return

    total = 0
    kept = 0
    rejected = 0

    # Use the provided output folder for output files
    output_keep_path = os.path.join(output_folder, OUTPUT_FILENAME)
    output_reject_path = os.path.join(output_folder, REJECTED_FILENAME)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_keep_path, "w", encoding="utf-8") as fout_keep, \
         open(output_reject_path, "w", encoding="utf-8") as fout_rej:

        for raw in fin:
            total += 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception as e:
                # malformed JSON -> record as rejected
                rej = {"original": raw, "reason": "json_parse_error", "error": str(e)}
                fout_rej.write(json.dumps(rej, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            # required fields
            zh = normalize_text(rec.get("zh_text", ""))
            en = normalize_text(rec.get("en_text", ""))
            input_cosine = rec.get("cosine", None)

            # early filters
            if not zh or not en:
                rej = {"original": rec, "reason": "empty_text"}
                fout_rej.write(json.dumps(rej, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            # use input cosine as early reject
            if input_cosine is not None:
                try:
                    cval = float(input_cosine)
                    if cval < TIMING_COSINE_THRESHOLD:
                        rej = {"original": rec, "reason": "timing_cosine_below_threshold", "timing_cosine": cval}
                        fout_rej.write(json.dumps(rej, ensure_ascii=False) + "\n")
                        rejected += 1
                        continue
                except Exception:
                    # if parsing fails, continue to SBERT
                    pass

            # split into sentences
            cn_sents = sent_split(zh, nlp_zh)
            en_sents = sent_split(en, nlp_en)

            # if splitting yields single long segments it still works
            keep, debug = veto(cn_sents, en_sents)

            if keep:
                out = {"english": en, "chinese": zh}
                fout_keep.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rej = {"original": rec, "reason": debug.get("fail", "veto_failed"), "debug": debug}
                fout_rej.write(json.dumps(rej, ensure_ascii=False) + "\n")
                rejected += 1

            # print progress
            if total % 100 == 0:
                print(f"[{os.path.basename(output_folder)} progress] processed={total} kept={kept} rejected={rejected}", file=sys.stderr)

    # print report
    print(f"\n========== CLEANING REPORT for {os.path.basename(output_folder)} ==========", file=sys.stderr)
    print(f"Input file: {input_path}", file=sys.stderr)
    print(f"Total records processed: {total}", file=sys.stderr)
    print(f"Kept (written to {output_keep_path}): {kept}", file=sys.stderr)
    print(f"Rejected (written to {output_reject_path}): {rejected}", file=sys.stderr)
    print("=====================================\n", file=sys.stderr)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_all_folders(script_dir)
