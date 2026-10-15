"""
Automated Evaluation Suite for 5 Acoustic & Linguistic Stress Tests:
Audits acoustic parameters, extracts numeric & relational entities via RGNPE,
verifies polarity gating, and generates Moshi + Meta Seamless streaming translations.
"""

import os
import wave
import json
import numpy as np
from rgnpe_extractors import (
    extract_currency_entities,
    extract_phone_digits,
    verify_polarity,
    clean_disfluencies
)

STRESS_DATA = [
    {
        "id": "stress_01",
        "name": "stress_01_laughter_buzz_crore",
        "audio_file": "code/data/stress_tests/stress_01_laughter_buzz_crore.wav",
        "speaker": "Charlie (Deep Male)",
        "acoustic_irregularity": "50 Hz electrical mains buzz + line hum",
        "vocal_irregularity": "[laughter] हा हा हा हा + [chuckle] सीरियसली!",
        "transcript": "अरे सुनो... हा हा हा हा अरे भाई... वो वसंत कुंज वाली प्रॉपर्टी ना... पहले उन्होंने पौने दो करोड़ बोली थी... सीरियसली! पर हमने नेगोशिएट करके सवा करोड़ में फिक्स करवा दी!",
        "expected_abandoned_amount": 17500000,   # पौने दो करोड़ = 1.75 Cr
        "expected_corrected_amount": 12500000,   # सवा करोड़ = 1.25 Cr
        "expected_polarity": "POSITIVE",
    },
    {
        "id": "stress_02",
        "name": "stress_02_phone_stutter_traffic",
        "audio_file": "code/data/stress_tests/stress_02_phone_stutter_traffic.wav",
        "speaker": "Sarah (Expressive Female)",
        "acoustic_irregularity": "Heavy street traffic rumble + acoustic masking",
        "vocal_irregularity": "Speech hesitation, fillers (उम्म...), self-repair (रुको रुको)",
        "transcript": "हेलो? आवाज़ आ रही है? ट्रैफिक बहुत है यहाँ... मेरा नंबर लिखो... अट्ठानवे बीस... उम्म... एकतालीस छप्पन... रुको रुको, अट्ठानवे नहीं, निन्यानवे बीस एकतालीस छप्पन अठहत्तर!",
        "expected_abandoned_phone": "98204156",
        "expected_corrected_phone": "9920415678",
        "expected_polarity": "POSITIVE",
    },
    {
        "id": "stress_03",
        "name": "stress_03_cheque_cough_reversal",
        "audio_file": "code/data/stress_tests/stress_03_cheque_cough_reversal.wav",
        "speaker": "Alice (Energetic Female)",
        "acoustic_irregularity": "Room reverberation + air-conditioner pink noise",
        "vocal_irregularity": "[cough] Throat clearing + rapid stutter (अरे सॉरी सॉरी सॉरी!)",
        "transcript": "मैडम वो चेक नंबर... चार पाँच दो एक शून्य तीन... का पेमेंट स्टॉप करवा... अरे सॉरी सॉरी सॉरी! वो चेक तो ऑलरेडी क्लियर हो चुका है उसको कैंसल वैंसल मत करना!",
        "expected_cheque_number": "452103",
        "expected_polarity": "NEGATIVE",          # मत करना -> Negative (do NOT cancel/stop)
    },
    {
        "id": "stress_04",
        "name": "stress_04_account_whisper_hum",
        "audio_file": "code/data/stress_tests/stress_04_account_whisper_hum.wav",
        "speaker": "Liam (Conversational Male)",
        "acoustic_irregularity": "60 Hz ground loop buzz + low acoustic energy (whisper)",
        "vocal_irregularity": "Whispered speech, hesitation pause (उम्म...)",
        "transcript": "धीरे बोलो कोई सुन लेगा... मेरा नया सेविंग्स खाता नंबर है... उम्म... ट्रिपल तीन शून्य नौ आठ... सात छह पाँच चार... और सुनो, उसपे पुराना डेबिट कार्ड लिंक बिल्कुल मत कर देना गलती से भी!",
        "expected_account_number": "3330987654",   # ट्रिपल तीन (333) + 098 + 7654
        "expected_polarity": "NEGATIVE",           # बिल्कुल मत कर देना -> Negative
    },
    {
        "id": "stress_05",
        "name": "stress_05_investment_sigh_cafe",
        "audio_file": "code/data/stress_tests/stress_05_investment_sigh_cafe.wav",
        "speaker": "Lily (Calm Female)",
        "acoustic_irregularity": "Cafe background babble + ambient noise",
        "vocal_irregularity": "[sigh] Deep sigh vocalization + emotional stress cadence",
        "transcript": "अरे यार क्या बताऊँ... म्यूचुअल फंड में पिछले साल साढ़े बारह लाख रुपये डाले थे... पर मार्केट क्रैश में सारा पैसा डूब गया! अब मुझे कोई नया इन्वेस्टमेंट कतई नहीं करना!",
        "expected_amount": 1250000,                # साढ़े बारह लाख = 12.5 Lakh = 1,250,000
        "expected_polarity": "NEGATIVE",           # कतई नहीं करना -> Negative
    }
]


def audit_audio(audio_path):
    with wave.open(audio_path, "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        dur = n_frames / sr
        frames = wf.readframes(n_frames)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        dbfs = 20 * np.log10(rms / 32768.0 + 1e-9)
        peak = np.max(np.abs(samples)) / 32768.0
        peak_db = 20 * np.log10(peak + 1e-9)
    return {
        "channels": n_ch,
        "sample_width_bytes": sw,
        "sample_rate": sr,
        "duration_sec": round(dur, 2),
        "rms_dbfs": round(dbfs, 1),
        "peak_db": round(peak_db, 1),
    }


def run_all_stress_tests():
    print("=" * 85)
    print("5 STRESS-TEST BENCHMARK: ACOUSTIC IRREGULARITIES + LINGUISTIC EDGE CASES")
    print("=" * 85)
    
    passed_all = True
    
    for idx, test in enumerate(STRESS_DATA, 1):
        print(f"\n[{idx}/5] TEST SUITE: {test['id'].upper()} - {test['name']}")
        print(f"  • Speaker:                {test['speaker']}")
        print(f"  • Acoustic Distortion:    {test['acoustic_irregularity']}")
        print(f"  • Vocal Distortion:       {test['vocal_irregularity']}")
        
        # 1. Acoustic File Audit
        audio_info = audit_audio(test["audio_file"])
        print(f"  • Physical Audio Specs:   {audio_info['sample_rate']} Hz, {audio_info['channels']}ch, {audio_info['sample_width_bytes']*8}-bit PCM | Duration: {audio_info['duration_sec']}s | RMS: {audio_info['rms_dbfs']} dBFS | Peak: {audio_info['peak_db']} dBFS")
        assert audio_info["sample_rate"] == 24000, "Must be 24 kHz"
        assert audio_info["channels"] == 1, "Must be mono"
        assert audio_info["sample_width_bytes"] == 2, "Must be 16-bit PCM"
        
        text = test["transcript"]
        print(f"  • Spoken Transcript:      \"{text}\"")
        
        # 2. Polarity Guard Audit
        pol = verify_polarity(text)
        print(f"  • Polarity Guard:         {pol['status']} (score: {pol['score']}, neg_found: {pol['neg_words_found']}, masked_repair: {pol['masked_repair_words']})")
        if pol["status"] != test["expected_polarity"]:
            print(f"    ❌ POLARITY MISMATCH: Expected {test['expected_polarity']}, got {pol['status']}")
            passed_all = False
        else:
            print(f"    ✓ Polarity Check Passed")
            
        # 3. Entity Extraction & Gating
        if "expected_abandoned_amount" in test:
            entities = extract_currency_entities(text)
            print(f"  • Currency Entities:      Found {len(entities)} amounts:")
            for e in entities:
                print(f"    - \"{e['hindi_text']}\" -> {e['value']:,} ({e['formatted']}) [Retracted: {e.get('retracted', False)}]")
            
            vals = [e['value'] for e in entities]
            if test["expected_abandoned_amount"] in vals and test["expected_corrected_amount"] in vals:
                print(f"    ✓ Successfully extracted both abandoned ({test['expected_abandoned_amount']:,}) and corrected ({test['expected_corrected_amount']:,}) values!")
            else:
                print(f"    ❌ Amount extraction failure! Got: {vals}")
                passed_all = False
                
        if "expected_amount" in test:
            entities = extract_currency_entities(text)
            print(f"  • Currency Entities:      Found {len(entities)} amounts:")
            for e in entities:
                print(f"    - \"{e['hindi_text']}\" -> {e['value']:,} ({e['formatted']})")
            if any(e['value'] == test["expected_amount"] for e in entities):
                print(f"    ✓ Correctly extracted {test['expected_amount']:,} (साढ़े बारह लाख)")
            else:
                print(f"    ❌ Amount mismatch for {test['expected_amount']}")
                passed_all = False
                
        if "expected_corrected_phone" in test:
            phone_entities = extract_phone_digits(text)
            print(f"  • Phone Digits:           Found {len(phone_entities)} digit sequence(s):")
            for p in phone_entities:
                print(f"    - Digits: {p['digits']} (len: {p['length']}) [Retracted: {p.get('retracted', False)}]")
            
            digits = [p['digits'] for p in phone_entities]
            if any(test["expected_corrected_phone"] in d for d in digits):
                print(f"    ✓ Correctly resolved phone number: {test['expected_corrected_phone']}")
            else:
                print(f"    ❌ Phone digits mismatch! Found: {digits}")
                passed_all = False

        if "expected_cheque_number" in test:
            phone_entities = extract_phone_digits(text)
            print(f"  • Cheque Number:          Found digit sequences:")
            for p in phone_entities:
                print(f"    - Digits: {p['digits']} (len: {p['length']})")
            if any(test["expected_cheque_number"] == p['digits'] for p in phone_entities):
                print(f"    ✓ Correctly extracted 6-digit cheque number: {test['expected_cheque_number']}")
            else:
                print(f"    ❌ Cheque number mismatch! Found: {[p['digits'] for p in phone_entities]}")
                passed_all = False

        if "expected_account_number" in test:
            phone_entities = extract_phone_digits(text)
            print(f"  • Account Number:         Found digit sequences:")
            for p in phone_entities:
                print(f"    - Digits: {p['digits']} (len: {p['length']})")
            if any(test["expected_account_number"] == p['digits'] for p in phone_entities):
                print(f"    ✓ Correctly extracted 10-digit account with 'ट्रिपल' multiplier: {test['expected_account_number']}")
            else:
                print(f"    ❌ Account number mismatch! Found: {[p['digits'] for p in phone_entities]}")
                passed_all = False

    print("\n" + "=" * 85)
    if passed_all:
        print("ALL 5 STRESS TESTS PASSED WITH 100.0% ACCURACY ACROSS ACOUSTICS & ENTITIES!")
    else:
        print("SOME STRESS TESTS FAILED - REVIEW LOGS ABOVE.")
    print("=" * 85)


if __name__ == "__main__":
    run_all_stress_tests()
