import os
import sys
import json
import re
import csv
import unicodedata
import torch
import torchaudio
import soundfile as sf
import numpy as np
import uroman as ur
from torchaudio.pipelines import MMS_FA as bundle

# 1. Global initialization
device = torch.device("cpu")
_model = bundle.get_model().to(device).eval()
_tokenizer = bundle.get_tokenizer()
_aligner = bundle.get_aligner()
_labels = bundle.get_labels()
_uroman = ur.Uroman()

def verify_and_log_model_parity():
    """Confirms CTC emission dimension matches labels and logs star token info."""
    dummy_wav = torch.zeros(1, 16000, device=device)
    with torch.inference_mode():
        dummy_emission, _ = _model(dummy_wav)
    vocab_dim = dummy_emission.shape[-1]
    labels_len = len(_labels)
    has_star = '*' in _labels
    print("=== MODEL & TOKENIZER SANITY CHECK ===")
    print(f"Emission Vocab Dim: {vocab_dim} | Labels Len: {labels_len} | Star Token in Labels: {has_star}")
    print(f"Labels Sample: {_labels[:10]} ... {_labels[-5:]}")
    if vocab_dim != labels_len:
        raise RuntimeError(f"Vocabulary mismatch! Emission ({vocab_dim}) != Labels ({labels_len})")
    print("Parity confirmed: MMS_FA CTC emissions align with tokenizer.\n")

def normalize_text_for_match(text):
    """Unicode NFC normalization and casefold for exact matching."""
    norm = unicodedata.normalize("NFC", str(text)).casefold()
    clean = re.sub(r"[।,\-?!.\s]+", " ", norm).strip()
    return clean

def split_preserving_original(text):
    """Splits text on whitespace AND embedded punctuation, caching both forms."""
    norm = unicodedata.normalize("NFC", str(text))
    spaced = re.sub(r"[।,?!.\-]+", " ", norm)
    tokens = [w for w in spaced.split() if w.strip()]
    return [{"raw": w, "norm": normalize_text_for_match(w)} for w in tokens]

def romanize_word(w):
    """Romanizes Hindi word with language code 'hin' for schwa-deletion."""
    rom = _uroman.romanize_string(w, lcode="hin").lower()
    return re.sub(r"[^a-z']", "", rom)

def find_subsequence(full_list, sub_list, start_idx=0, end_idx=None):
    """Finds contiguous sub_list inside bounded slice full_list[start_idx:end_idx]."""
    if end_idx is None:
        end_idx = len(full_list)
    n_sub = len(sub_list)
    if n_sub == 0 or n_sub > (end_idx - start_idx):
        return -1
    for i in range(start_idx, end_idx - n_sub + 1):
        if full_list[i:i + n_sub] == sub_list:
            return i
    return -1

def resolve_breakers_bounded(aligned_words, record_metadata, is_fallback=False):
    words_norm = [w["w_norm"] for w in aligned_words]
    annotated = [dict(w, is_breaker=False, status="normal") for w in aligned_words]
    
    breakers_info = record_metadata.get("breakers", [])
    search_cursor = 0
    resolved_breakers = []
    unmatched_markers = []
    unmatched_phrases = []
    
    for meta_idx, b in enumerate(breakers_info):
        marker_phrase = normalize_text_for_match(b.get("marker", "")).split()
        abandoned_phrase = normalize_text_for_match(b.get("abandoned", "")).split()
        corrected_phrase = normalize_text_for_match(b.get("corrected", "")).split()
        
        # 1. Try combined tripartite search (abandoned + marker + corrected) first
        is_ambiguous_marker = False
        m_idx = -1
        ab_idx = -1
        corr_idx = -1
        
        combined_phrase = abandoned_phrase + marker_phrase + corrected_phrase
        if abandoned_phrase and corrected_phrase:
            comb_idx = find_subsequence(words_norm, combined_phrase, start_idx=search_cursor)
            if comb_idx != -1:
                ab_idx = comb_idx
                m_idx = comb_idx + len(abandoned_phrase)
                corr_idx = m_idx + len(marker_phrase)
                
        # Fallback to marker alone if combined match fails
        if m_idx == -1:
            m_idx = find_subsequence(words_norm, marker_phrase, start_idx=search_cursor)
            if m_idx == -1:
                unmatched_markers.append({"meta_idx": meta_idx+1, "phrase": " ".join(marker_phrase)})
                continue
            is_ambiguous_marker = True
            
        m_end_idx = m_idx + len(marker_phrase)
        for i in range(m_idx, m_end_idx):
            annotated[i]["is_breaker"] = True
            annotated[i]["status"] = "breaker_marker"
            
        # 2. Match abandoned phrase if not already set by combined
        ab_end_idx = m_idx
        if abandoned_phrase and ab_idx == -1:
            ab_search_start = max(search_cursor, m_idx - len(abandoned_phrase) - 4)
            ab_idx = find_subsequence(words_norm, abandoned_phrase, start_idx=ab_search_start, end_idx=m_idx)
            if ab_idx != -1:
                ab_end_idx = ab_idx + len(abandoned_phrase)
                for i in range(ab_idx, ab_end_idx):
                    annotated[i]["status"] = "abandoned_entity"
            else:
                unmatched_phrases.append({"meta_idx": meta_idx+1, "type": "abandoned", "phrase": " ".join(abandoned_phrase)})
        elif ab_idx != -1:
            ab_end_idx = ab_idx + len(abandoned_phrase)
            for i in range(ab_idx, ab_end_idx):
                annotated[i]["status"] = "abandoned_entity"
                
        # 3. Match corrected phrase if not already set by combined
        corr_end_idx = m_end_idx
        if corrected_phrase and corr_idx == -1:
            window_end = min(len(words_norm), m_end_idx + len(corrected_phrase) + 5)
            corr_idx = find_subsequence(words_norm, corrected_phrase, start_idx=m_end_idx, end_idx=window_end)
            if corr_idx != -1:
                corr_end_idx = corr_idx + len(corrected_phrase)
                for i in range(corr_idx, corr_end_idx):
                    annotated[i]["status"] = "committed_entity"
            else:
                unmatched_phrases.append({"meta_idx": meta_idx+1, "type": "corrected", "phrase": " ".join(corrected_phrase)})
        elif corr_idx != -1:
            corr_end_idx = corr_idx + len(corrected_phrase)
            for i in range(corr_idx, corr_end_idx):
                annotated[i]["status"] = "committed_entity"

        # 4. Check for unaligned words across ALL regions (abandoned + marker + corrected)
        breaker_span_indices = []
        if ab_idx != -1:
            breaker_span_indices.extend(range(ab_idx, ab_end_idx))
        breaker_span_indices.extend(range(m_idx, m_end_idx))
        if corr_idx != -1:
            breaker_span_indices.extend(range(corr_idx, corr_end_idx))
            
        has_unaligned_word = any(not annotated[k]["aligned"] for k in breaker_span_indices)
        aligned_scores = [annotated[k]["score"] for k in breaker_span_indices if annotated[k]["aligned"]]
        local_score = round(float(np.mean(aligned_scores)), 3) if aligned_scores else 0.0

        # 5. Safe-to-speak timestamp & confidence derivation
        phrase_missing = (abandoned_phrase and ab_idx == -1) or (corrected_phrase and corr_idx == -1)
        time_key = "safe_to_speak_sec"
        timing_note = "exact_committed_end"
        
        if is_fallback:
            safe_confidence = "none"
            safe_time = annotated[m_end_idx - 1]["end"]
            timing_note = "unverified_fallback"
        elif phrase_missing:
            safe_confidence = "low_phrase_not_found"
            safe_time = annotated[m_end_idx - 1]["end"]
            timing_note = "fallback_marker_end"
        elif has_unaligned_word:
            safe_confidence = "low_unaligned_word"
            time_key = "safe_to_speak_min_sec"
            timing_note = "lower_bound_estimate_due_to_unaligned_token"
            if corr_idx != -1:
                valid_ends = [annotated[k]["end"] for k in range(corr_idx, corr_end_idx) if annotated[k]["aligned"]]
                safe_time = valid_ends[-1] if valid_ends else annotated[m_end_idx - 1]["end"]
            else:
                safe_time = annotated[m_end_idx - 1]["end"]
        elif is_ambiguous_marker:
            safe_confidence = "low_marker_ambiguous"
            safe_time = annotated[corr_end_idx - 1]["end"] if corr_idx != -1 else annotated[m_end_idx - 1]["end"]
        else:
            safe_confidence = "high"
            safe_time = annotated[corr_end_idx - 1]["end"] if corr_idx != -1 else annotated[m_end_idx - 1]["end"]

        resolved_breakers.append({
            "meta_idx": meta_idx + 1,
            "marker": " ".join(marker_phrase),
            "marker_start": annotated[m_idx]["start"],
            "marker_end": annotated[m_end_idx - 1]["end"],
            time_key: round(safe_time, 3),
            "safe_confidence": safe_confidence,
            "timing_note": timing_note,
            "breaker_acoustic_score": local_score,
            "has_unaligned_word": has_unaligned_word
        })
        search_cursor = corr_end_idx
        
    return annotated, resolved_breakers, unmatched_markers, unmatched_phrases

def align_clip(audio_path, record, output_json, cuts_dir=None, cuts_manifest_rows=None):
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).to(device)
    
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
        
    duration_sec = round(waveform.shape[1] / bundle.sample_rate, 3)
    word_tokens = split_preserving_original(record["hi_verbatim"])
    if len(word_tokens) == 0:
        raise ValueError(f"Empty transcript in record {record['id']}")
        
    with torch.inference_mode():
        emission, _ = _model(waveform)
        
    ratio = waveform.size(1) / emission.size(1) / bundle.sample_rate
    alignment_method = "mms_ctc_trellis"
    
    # Precompute romanization ONCE per word and cache
    for w in word_tokens:
        w["roman"] = romanize_word(w["raw"])
        
    kept_indices = [i for i, w in enumerate(word_tokens) if w["roman"]]
    aligned_words = [None] * len(word_tokens)
    
    if len(kept_indices) > 0:
        try:
            roman_list = [word_tokens[i]["roman"] for i in kept_indices]
            tokens = _tokenizer(roman_list)
            spans = _aligner(emission[0], tokens)
            
            # Diagnostic check on first clip
            if record["id"] == "rec_000001":
                print(f"[DIAGNOSTIC] Clip 1 Tokenized Words: {len(kept_indices)} | Aligner Output Spans: {len(spans)}")
                
            if len(spans) != len(kept_indices):
                raise RuntimeError(f"Span count mismatch: {len(spans)} spans vs {len(kept_indices)} tokens")
                
            for idx, s in zip(kept_indices, spans):
                if len(s) == 0:
                    continue
                w_start = round(s[0].start * ratio, 3)
                w_end = round(s[-1].end * ratio, 3)
                score = round(float(sum(t.score for t in s) / len(s)), 3)
                aligned_words[idx] = {
                    "w": word_tokens[idx]["raw"],
                    "w_norm": word_tokens[idx]["norm"],
                    "roman": word_tokens[idx]["roman"],
                    "aligned": True,
                    "start": min(w_start, duration_sec),
                    "end": min(max(w_end, w_start), duration_sec),
                    "score": score
                }
        except Exception as e:
            print(f"  [ERROR] CTC Trellis failed ({e}). Tagging as uniform_fallback.")
            alignment_method = "uniform_fallback"
    else:
        alignment_method = "uniform_fallback"

    # Interpolate unaligned words
    prev_end = 0.0
    for i in range(len(aligned_words)):
        if aligned_words[i] is None:
            if alignment_method == "uniform_fallback":
                step = duration_sec / len(word_tokens)
                w_start = round(i * step, 3)
                w_end = round((i + 1) * step, 3)
            else:
                w_start = prev_end
                w_end = prev_end
            aligned_words[i] = {
                "w": word_tokens[i]["raw"],
                "w_norm": word_tokens[i]["norm"],
                "roman": word_tokens[i]["roman"],
                "aligned": False,
                "start": w_start,
                "end": w_end,
                "score": 0.0
            }
        prev_end = aligned_words[i]["end"]

    # Invariant validation
    for i in range(len(aligned_words)):
        if aligned_words[i]["start"] > aligned_words[i]["end"]:
            raise ValueError(f"Invariant: start > end at {aligned_words[i]}")
        if i > 0 and aligned_words[i]["start"] < aligned_words[i-1]["start"]:
            raise ValueError(f"Invariant: non-monotonic start at {aligned_words[i]}")
    if aligned_words[-1]["end"] > duration_sec + 0.05:
        raise ValueError(f"Invariant: end time exceeds clip duration ({duration_sec})")

    # Resolve breakers
    is_fallback = (alignment_method == "uniform_fallback")
    annotated_words, resolved_breakers, unmatched_m, unmatched_p = resolve_breakers_bounded(
        aligned_words, record, is_fallback=is_fallback
    )
    
    # Export cuts ONLY if alignment is real
    if cuts_dir and resolved_breakers and not is_fallback:
        os.makedirs(cuts_dir, exist_ok=True)
        for b in resolved_breakers:
            pad = 1.2 if b["has_unaligned_word"] else 0.8
            time_val = b.get("safe_to_speak_sec", b.get("safe_to_speak_min_sec", b["marker_end"]))
            t1 = max(0.0, b["marker_start"] - pad)
            t2 = min(duration_sec, time_val + pad)
            s1, s2 = int(t1 * bundle.sample_rate), int(t2 * bundle.sample_rate)
            if s2 > s1:
                snippet = waveform[0, s1:s2].cpu().numpy()
                cut_filename = f"{record['id']}_b{b['meta_idx']}_{b['safe_confidence']}.wav"
                cut_path = os.path.join(cuts_dir, cut_filename)
                sf.write(cut_path, snippet, bundle.sample_rate)
                if cuts_manifest_rows is not None:
                    cuts_manifest_rows.append({
                        "clip_id": record["id"],
                        "breaker_idx": b["meta_idx"],
                        "marker": b["marker"],
                        "breaker_score": b["breaker_acoustic_score"],
                        "safe_to_speak_val": time_val,
                        "confidence": b["safe_confidence"],
                        "has_unaligned": b["has_unaligned_word"],
                        "cut_file": cut_filename
                    })

    scores = [w["score"] for w in annotated_words if w["aligned"]]
    mean_score = round(float(np.mean(scores)), 3) if scores else 0.0
    
    output_data = {
        "id": record["id"],
        "clip_file": record["clip_file"],
        "duration_sec": duration_sec,
        "alignment_method": alignment_method,
        "mean_acoustic_score": mean_score,
        "total_words": len(annotated_words),
        "aligned_words_count": len(scores),
        "unaligned_words_count": len(annotated_words) - len(scores),
        "breakers": resolved_breakers,
        "unmatched_markers": unmatched_m,
        "unmatched_phrases": unmatched_p,
        "words": annotated_words
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed: {record['clip_file']}")
    print(f"  • Method: {alignment_method} | Mean Score: {mean_score} | Words: {len(annotated_words)} (Unaligned: {len(annotated_words)-len(scores)})")
    for b in resolved_breakers:
        time_key = "safe_to_speak_min_sec" if b["has_unaligned_word"] else "safe_to_speak_sec"
        print(f"  • Breaker #{b['meta_idx']} '{b['marker']}': [{b['marker_start']}s -> {b['marker_end']}s] => {time_key}: {b[time_key]}s (Score: {b['breaker_acoustic_score']}, Conf: {b['safe_confidence']})")
    print(f"  • JSON: {output_json}\n")
    return output_data, unmatched_m, unmatched_p

def main():
    base_dir = "code" if os.path.exists("code") else "."
    data_dir = os.path.join(base_dir, "data")
    aligned_dir = os.path.join(data_dir, "aligned")
    cuts_dir = os.path.join(data_dir, "verification_cuts")
    manifest_path = os.path.join(data_dir, "manifest.json")
    standardized_dir = os.path.join(data_dir, "standardized")
    
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(cuts_dir, exist_ok=True)
    
    if not os.path.exists(manifest_path):
        print(f"[FATAL] Manifest not found at: {manifest_path}", file=sys.stderr)
        sys.exit(1)
        
    verify_and_log_model_parity()
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print("=== EXECUTING RIGOROUS FORCED ALIGNMENT & BREAKER AUDIT ===\n")
    results = []
    cuts_manifest_rows = []
    total_unmatched_markers = []
    total_unmatched_phrases = []
    
    for rec in manifest:
        audio_path = os.path.join(standardized_dir, rec["clip_file"])
        out_json = os.path.join(aligned_dir, rec["clip_file"].replace(".wav", "_aligned.json"))
        try:
            res, un_m, un_p = align_clip(audio_path, rec, out_json, cuts_dir=cuts_dir, cuts_manifest_rows=cuts_manifest_rows)
            if res:
                results.append(res)
                if un_m: total_unmatched_markers.extend([(rec["id"], m) for m in un_m])
                if un_p: total_unmatched_phrases.extend([(rec["id"], p) for p in un_p])
        except Exception as e:
            err_type = type(e).__name__
            print(f"[FAIL] Clip {rec['id']} failed ({err_type}): {e} [Audio: {audio_path}]\n", file=sys.stderr)

    # Always ensure sidecar verification cuts CSV exists
    csv_path = os.path.join(cuts_dir, "cuts_manifest.csv")
    fieldnames = ["clip_id", "breaker_idx", "marker", "breaker_score", "safe_to_speak_val", "confidence", "has_unaligned", "cut_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        if cuts_manifest_rows:
            writer.writerows(cuts_manifest_rows)
    print(f"Exported cuts manifest table -> {csv_path}\n")

    # Summary Audit Table
    ctc_count = sum(1 for r in results if r["alignment_method"] == "mms_ctc_trellis")
    fb_count = sum(1 for r in results if r["alignment_method"] == "uniform_fallback")
    all_scores = [r["mean_acoustic_score"] for r in results if r["alignment_method"] == "mms_ctc_trellis"]
    avg_score = np.mean(all_scores) if all_scores else 0.0

    print("==========================================================")
    print("                 PIPELINE AUDIT SUMMARY                   ")
    print("==========================================================")
    print(f"Total Clips Processed: {len(results)} / {len(manifest)}")
    print(f"Real CTC Aligned Clips: {ctc_count}")
    print(f"Fallback Clips (Unverified): {fb_count}")
    print(f"Overall Mean Acoustic Token Score: {avg_score:.3f}")
    
    if total_unmatched_markers:
        print(f"[ERROR] Unmatched Breaker Markers ({len(total_unmatched_markers)}): {total_unmatched_markers}")
    if total_unmatched_phrases:
        print(f"[WARN] Unmatched Breaker Phrases ({len(total_unmatched_phrases)}): {total_unmatched_phrases}")
        
    unaligned_clips = [r["id"] for r in results if r["unaligned_words_count"] > 0]
    if unaligned_clips:
        print(f"Clips with unaligned words (interpolated): {unaligned_clips}")
    else:
        print("100% token coverage: Zero unaligned words across all clips.")
        
    print(f"Verification cuts and CSV manifest saved in: {cuts_dir}")
    print("==========================================================")
    
    if len(results) < len(manifest) or len(total_unmatched_markers) > 0:
        print("[FATAL] Pipeline completed with missing clips or broken markers.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()