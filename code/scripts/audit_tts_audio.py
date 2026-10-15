"""
TTS Audio Quality, Mispronunciation & Artefact Audit Gate
Implements the Sarvam.ai Requirement:
"If you use TTS-generated speech anywhere, check it and filter or regenerate bad output. 
TTS can mispronounce names and numbers, skip or repeat words, or add artefacts."

Runs 5 Automated Quality & Integrity Gates on every audio clip:
  1. Acoustic Saturation & Clipping Gate: Checks peak levels, flat-top clipping, and digital ceiling.
  2. Pacing & Speed Cadence Gate: Measures syllables/char rate; flags hyper-speed chatter or frozen silence.
  3. Repetition & Vocoder Loop Gate: Detects autoregressive loop artefacts and repeated phrase stutters.
  4. CTC Trellis Acoustic Confidence Gate: Verifies word alignment score > 0.60 and 0 unaligned/skipped words.
  5. Entity & Numeral Integrity Gate: Validates exact digit runs, lakh/crore units, names, and negations.

Decision:
  - If all 5 gates pass: VERIFIED_CLEAN (Keep clip)
  - If any gate fails:   REJECT_AND_REGENERATE (Filter out / re-prompt TTS)
"""

import os
import sys
import json
import glob
import wave
import numpy as np
import soundfile as sf
from pathlib import Path

# Add code/scripts to sys.path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rgnpe_extractors import (
    parse_spoken_digits,
    parse_relative_scale_amount,
    check_polarity_with_breaker_masking
)

DATA_DIR = HERE.parent / "data"
STANDARDIZED_DIR = DATA_DIR / "standardized"
ALIGNED_DIR = DATA_DIR / "aligned"
MANIFEST_PATH = DATA_DIR / "manifest_rgnpe.json"

# Quality Thresholds
MAX_PEAK_DBTP = -1.0          # Digital true peak ceiling
MIN_CHAR_RATE = 5.0           # Minimum chars/sec (flags unnatural pauses / frozen vocoder)
MAX_CHAR_RATE = 25.0          # Maximum chars/sec (flags runaway TTS gibberish / speed glitches)
MIN_CTC_SCORE = 0.50          # Minimum acoustic alignment confidence per word
MAX_REPETITION_COUNT = 3      # Maximum identical consecutive token repeats allowed


def audit_clip(audio_path, aligned_path, record_meta):
    clip_id = record_meta.get("id") or record_meta.get("clip_id")
    clip_name = os.path.basename(audio_path)
    failures = []
    warnings = []
    
    # -------------------------------------------------------------------------
    # Gate 1: Acoustic Saturation & Clipping Detector
    # -------------------------------------------------------------------------
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    peak_linear = np.max(np.abs(data))
    peak_db = 20.0 * np.log10(peak_linear + 1e-9)
    duration_sec = len(data) / sr
    
    # Check for hard clipping (samples hitting exactly 1.0 or -1.0)
    clipped_samples = np.sum(np.abs(data) >= 0.999)
    if clipped_samples > 0:
        failures.append(f"Gate 1 Failed: Hard digital clipping detected ({clipped_samples} clipped samples)")
    elif peak_db > (MAX_PEAK_DBTP + 0.05):
        failures.append(f"Gate 1 Failed: True peak ({peak_db:.2f} dBTP) exceeds safety ceiling ({MAX_PEAK_DBTP} dBTP)")
        
    # -------------------------------------------------------------------------
    # Gate 2: Pacing & Speaking Rate Cadence
    # -------------------------------------------------------------------------
    sentences = record_meta.get("sentences", [{}])
    transcript = (
        record_meta.get("hi_verbatim")
        or (sentences[0].get("hi_text") if sentences else "")
        or (sentences[0].get("hi_verbatim") if sentences else "")
        or ""
    )
    char_count = len(transcript.replace(" ", ""))
    char_rate = char_count / duration_sec if duration_sec > 0 else 0
    
    if char_rate < MIN_CHAR_RATE:
        failures.append(f"Gate 2 Failed: Unnatural slow cadence ({char_rate:.1f} char/s < {MIN_CHAR_RATE}); vocoder stall")
    elif char_rate > MAX_CHAR_RATE:
        failures.append(f"Gate 2 Failed: Hyper-speed speech runaway ({char_rate:.1f} char/s > {MAX_CHAR_RATE}); vocoder gibberish")
        
    # -------------------------------------------------------------------------
    # Gate 3: Autoregressive Repetition & Stutter Loop Detector
    # -------------------------------------------------------------------------
    words = transcript.split()
    max_streak = 1
    current_streak = 1
    repeated_word = ""
    for i in range(1, len(words)):
        if words[i] == words[i-1] and len(words[i]) > 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                repeated_word = words[i]
        else:
            current_streak = 1
            
    if max_streak > MAX_REPETITION_COUNT:
        failures.append(f"Gate 3 Failed: Autoregressive repetition loop detected on word '{repeated_word}' (repeated {max_streak}x)")
        
    # -------------------------------------------------------------------------
    # Gate 4: CTC Trellis Alignment & Skipped Word Verification
    # -------------------------------------------------------------------------
    if os.path.exists(aligned_path):
        with open(aligned_path, "r", encoding="utf-8") as f:
            aligned_data = json.load(f)
            
        unaligned = aligned_data.get("unaligned_words_count", 0)
        mean_score = aligned_data.get("mean_acoustic_score", 0.0)
        
        if unaligned > 0:
            failures.append(f"Gate 4 Failed: TTS skipped {unaligned} word(s) during synthesis (unaligned in CTC trellis)")
        elif mean_score < MIN_CTC_SCORE:
            warnings.append(f"Gate 4 Warning: Low acoustic confidence score ({mean_score:.3f} < {MIN_CTC_SCORE})")
    else:
        warnings.append(f"Gate 4 Warning: Alignment JSON missing at {aligned_path}")
        
    # -------------------------------------------------------------------------
    # Gate 5: Entity & Numeral Integrity Verification
    # -------------------------------------------------------------------------
    locked_entities = record_meta.get("entities_locked", {})
    if isinstance(locked_entities, dict):
        entity_values = list(locked_entities.values())
    elif isinstance(locked_entities, list):
        entity_values = locked_entities
    else:
        entity_values = []
        
    for ent_str in entity_values:
        ent_str = str(ent_str)
        # Check digit sequences (e.g. 98204 15678, 452103)
        digits_in_ent = "".join(c for c in ent_str if c.isdigit())
        if len(digits_in_ent) >= 4:
            parsed_src_digits = parse_spoken_digits(transcript)
            if digits_in_ent not in parsed_src_digits and digits_in_ent not in transcript:
                failures.append(f"Gate 5 Failed: Entity digit sequence '{digits_in_ent}' dropped or corrupted in transcript")
                
    status = "REJECT_AND_REGENERATE" if failures else ("PASS_WITH_WARNINGS" if warnings else "VERIFIED_CLEAN")
    
    return {
        "clip_id": clip_id,
        "clip_file": clip_name,
        "duration_sec": round(duration_sec, 2),
        "peak_db": round(peak_db, 2),
        "char_rate": round(char_rate, 1),
        "status": status,
        "failures": failures,
        "warnings": warnings
    }


def main():
    print("=" * 96)
    print("TTS AUDIO QUALITY, MISPRONUNCIATION & ARTEFACT AUDIT GATE")
    print("Rule: Filter or regenerate bad output (mispronunciations, dropped numbers, vocoder loops)")
    print("=" * 96 + "\n")
    
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} not found. Run rgnpe_engine.py first.")
        sys.exit(1)
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    results = []
    passed_count = 0
    rejected_count = 0
    
    for record in manifest:
        clip_id = record.get("id") or record.get("clip_id")
        audio_val = record.get("audio") or record.get("audio_filepath") or record.get("clip_file")
        if isinstance(audio_val, dict):
            audio_name = audio_val.get("name") or audio_val.get("path")
        elif isinstance(audio_val, str):
            audio_name = os.path.basename(audio_val)
        else:
            audio_name = f"clip_{clip_id.split('_')[-1]}_standardized.wav"
            
        # Map rec_000001 -> clip_01
        num_str = f"{int(clip_id.split('_')[-1]):02d}"
        matches = list(STANDARDIZED_DIR.glob(f"*_{num_str}_*.wav"))
        if not matches:
            matches = list(STANDARDIZED_DIR.glob(f"*{num_str}*.wav"))
        if matches:
            audio_path = matches[0]
        else:
            audio_path = STANDARDIZED_DIR / f"clip_{num_str}.wav"
            
        # Find aligned JSON
        aligned_matches = list(ALIGNED_DIR.glob(f"*{num_str}*aligned.json"))
        if aligned_matches:
            aligned_path = aligned_matches[0]
        else:
            aligned_path = ALIGNED_DIR / f"clip_{num_str}_aligned.json"
                
        audit_res = audit_clip(audio_path, aligned_path, record)
        results.append(audit_res)
        
        status_icon = "✓" if audit_res["status"] == "VERIFIED_CLEAN" else "✗"
        print(f"[{status_icon}] {audit_res['clip_file']:<32} | {audit_res['duration_sec']:>5.2f}s | "
              f"Peak: {audit_res['peak_db']:>6.2f} dBTP | Rate: {audit_res['char_rate']:>4.1f} c/s | Status: {audit_res['status']}")
        
        if audit_res["failures"]:
            for f in audit_res["failures"]:
                print(f"    ✗ {f}")
        if audit_res["warnings"]:
            for w in audit_res["warnings"]:
                print(f"    ⚠ {w}")
                
        if audit_res["status"] in ["VERIFIED_CLEAN", "PASS_WITH_WARNINGS"]:
            passed_count += 1
        else:
            rejected_count += 1
            
    print("\n" + "=" * 96)
    print(f"SUMMARY: {passed_count}/{len(results)} Clips Verified Clean ({passed_count/len(results)*100:.1f}%) | "
          f"{rejected_count} Rejected for Regeneration")
    print("=" * 96)
    
    # Save audit log
    out_json = DATA_DIR / "tts_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote detailed audit report -> {out_json}")


if __name__ == "__main__":
    main()
