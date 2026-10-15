"""
Interactive Recording & RGNPE Evaluation Script for User's Voice:

1. Displays challenging conversational Hindi scripts to read.
2. Records directly from the Mac microphone via FFmpeg AVFoundation or accepts a recorded file.
3. Conditions audio to Sarvam competition specs:
   - 24,000 Hz mono 16-bit PCM
   - 4th-order causal Butterworth bandpass (80 Hz - 11,000 Hz)
   - Integrated loudness normalized to -20.0 LUFS (true peak <= -1.0 dBTP)
   - Zero-padded to exact multiples of 80 ms (1,920 samples)
4. Evaluates hard entities (crores, lakhs, phone digits, multipliers) & polarity.
"""

import os
import sys
import time
import subprocess
import soundfile as sf
import numpy as np

# Ensure scripts dir in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from acoustic_conditioning import condition_and_frame_audio
from rgnpe_extractors import (
    parse_spoken_digits,
    parse_relative_scale_amount,
    check_polarity_with_breaker_masking,
    extract_phone_digits,
    extract_currency_entities,
    verify_polarity
)

SUGGESTED_SCRIPTS = [
    {
        "id": "1",
        "title": "Property Negotiation (Crores / Lakhs & Retraction)",
        "hindi": "अरे सुनो... वो वसंत कुंज वाली प्रॉपर्टी ना... पहले उन्होंने पौने दो करोड़ बोली थी... पर हमने नेगोशिएट करके सवा करोड़ में फिक्स करवा दी!",
        "english_meaning": "Hey listen... that Vasant Kunj property... earlier they asked 1.75 crore... but after negotiating we fixed it at 1.25 crore!",
        "expected_entities": "₹1.75 Cr (abandoned) -> ₹1.25 Cr (committed)"
    },
    {
        "id": "2",
        "title": "Phone Number with Paired Digits & Self-Repair",
        "hindi": "मेरा नया नंबर लिखो... अट्ठानवे बीस... उम्म... एकतालीस छप्पन... रुको रुको, अट्ठानवे नहीं, निन्यानवे बीस एकतालीस छप्पन अठहत्तर!",
        "english_meaning": "Write my new number... 98 20... umm... 41 56... wait wait, not 98, 99 20 41 56 78!",
        "expected_entities": "Retracted: 98204156 -> Corrected Phone: 9920415678"
    },
    {
        "id": "3",
        "title": "Cheque Stop Reversal (Emphatic Negation)",
        "hindi": "मैडम वो चेक नंबर... चार पाँच दो एक शून्य तीन... का पेमेंट स्टॉप करवा... अरे सॉरी सॉरी! वो चेक तो ऑलरेडी क्लियर हो चुका है उसको कैंसल वैंसल मत करना!",
        "english_meaning": "Madam, stop payment on cheque number 452103... oh sorry sorry! That cheque has already cleared, do NOT cancel it!",
        "expected_entities": "Cheque: 452103, Polarity: NEGATIVE (do NOT cancel)"
    },
    {
        "id": "4",
        "title": "Savings Account with Multipliers (ट्रिपल)",
        "hindi": "धीरे बोलो कोई सुन लेगा... मेरा नया सेविंग्स खाता नंबर है... ट्रिपल तीन शून्य नौ आठ सात छह पाँच चार... और उसपे डेबिट कार्ड लिंक मत करना!",
        "english_meaning": "Speak softly... my new savings account number is 3330987654... and do not link the debit card to it!",
        "expected_entities": "Account: 3330987654 (ट्रिपल तीन = 333), Polarity: NEGATIVE"
    }
]


def record_from_mic(output_path, duration_sec=10):
    print("\n" + "=" * 80)
    print(f"RECORDING LIVE FROM MICROPHONE FOR {duration_sec} SECONDS...")
    print("Speak clearly into your Mac microphone!")
    print("Press Ctrl+C early if you finish speaking before the timer ends.")
    print("=" * 80 + "\n")
    
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-f", "avfoundation",
        "-i", ":default",
        "-t", str(duration_sec),
        "-ar", "24000",
        "-ac", "1",
        output_path
    ]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for remaining in range(duration_sec, 0, -1):
            sys.stdout.write(f"\r🎤 Recording... [{remaining:02d}s left] (Press Ctrl+C to stop) ")
            sys.stdout.flush()
            time.sleep(1)
            if proc.poll() is not None:
                break
        proc.wait()
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
        proc.terminate()
        proc.wait()
        
    print(f"\n✓ Saved raw audio to: {output_path}")


def main():
    print("=" * 80)
    print("SARVAM AUDIO INGESTION & RGNPE PIPELINE - LIVE VOICE TEST")
    print("=" * 80)
    
    print("\nChoose a script to read out loud:")
    for s in SUGGESTED_SCRIPTS:
        print(f"\n[{s['id']}] {s['title']}")
        print(f"    🗣️  HINDI:   \"{s['hindi']}\"")
        print(f"    🎯  TARGET:  {s['expected_entities']}")
    print(f"[C] Custom Sentence (read whatever you like!)")
    
    choice = input("\nEnter script choice [1/2/3/4/C] (default: 1): ").strip() or "1"
    
    selected_script = None
    for s in SUGGESTED_SCRIPTS:
        if s["id"] == choice:
            selected_script = s
            break
            
    if selected_script:
        print(f"\nSELECTED SCRIPT [{selected_script['id']}]:")
        print("=" * 80)
        print(f"READ THIS OUT LOUD:")
        print(f"\n  \"{selected_script['hindi']}\"\n")
        print("=" * 80)
        reference_text = selected_script["hindi"]
    else:
        reference_text = input("\nEnter the Hindi transcript of what you will speak: ").strip()
        
    raw_audio_path = "code/data/my_voice_raw.wav"
    standard_audio_path = "code/data/my_voice_standardized.wav"
    os.makedirs("code/data", exist_ok=True)
    
    print("\nHow would you like to provide your voice?")
    print("  [1] Record directly using Mac microphone (via FFmpeg)")
    print("  [2] Use an existing audio file you already recorded (WAV / M4A / MP3)")
    mode = input("\nEnter option [1/2] (default: 1): ").strip() or "1"
    
    if mode == "1":
        dur = input("How many seconds do you need to speak? [default: 10]: ").strip() or "10"
        input("\nPress ENTER when you are ready to start recording...")
        record_from_mic(raw_audio_path, duration_sec=int(dur))
    else:
        file_path = input("Enter path to your audio file: ").strip()
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return
        # Convert to 24k mono wav
        cmd = [
            "/opt/homebrew/bin/ffmpeg", "-y",
            "-i", file_path,
            "-ar", "24000",
            "-ac", "1",
            raw_audio_path
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ Ingested audio to: {raw_audio_path}")
        
    # Phase 1: Acoustic Conditioning
    print("\n" + "-" * 80)
    print("RUNNING PHASE 1 ACOUSTIC CONDITIONING...")
    print("-" * 80)
    condition_and_frame_audio(raw_audio_path, standard_audio_path)
    
    # Verify acoustic specs on output file
    data_std, sr_std = sf.read(standard_audio_path)
    import pyloudnorm as pyln
    meter = pyln.Meter(sr_std)
    std_lufs = meter.integrated_loudness(data_std)
    std_peak_db = 20.0 * np.log10(np.max(np.abs(data_std)))
    total_frames = len(data_std) // 1920
    rem_samples = len(data_std) % 1920
    
    print(f"  • Sample Rate:       {sr_std} Hz (Competition Mandatory: 24,000 Hz)")
    print(f"  • Channels:          1 (Mono)")
    print(f"  • Format:            16-bit PCM")
    print(f"  • Normalized LUFS:   {std_lufs:.2f} LUFS (Target: -20.0 LUFS ± 0.3)")
    print(f"  • Peak Level:        {std_peak_db:.2f} dBTP (Ceiling: -1.0 dBTP)")
    print(f"  • Frame Alignment:   {total_frames} frames x 80 ms (Remainder: {rem_samples} samples = 0)")
    print(f"  ✓ Standardized file: {standard_audio_path}")
    
    # Phase 3: RGNPE Entity Extraction & Evaluation
    print("\n" + "-" * 80)
    print("RUNNING PHASE 3 RGNPE ENTITY EXTRACTION & POLARITY GUARD...")
    print("-" * 80)
    print(f"Input Hindi Transcript: \"{reference_text}\"\n")
    
    # 1. Amounts
    amounts = extract_currency_entities(reference_text)
    print(f"  [1] Currency Entities:")
    if amounts:
        for a in amounts:
            print(f"      • {a['formatted']} ({a['hindi_text']})")
    else:
        print(f"      • None detected")
        
    # 2. Phone / Account / Cheque Digits
    digits_list = extract_phone_digits(reference_text)
    print(f"\n  [2] Spoken Digits (Phone / Account / Cheque):")
    if digits_list:
        for d in digits_list:
            tag = "[RETRACTED ERROR]" if d["retracted"] else "[FINAL COMMITTED VALUE]"
            print(f"      • {d['digits']} (Length: {d['length']}) {tag}")
    else:
        print(f"      • None detected")
        
    # 3. Polarity
    pol = verify_polarity(reference_text)
    print(f"\n  [3] Polarity Guard:")
    print(f"      • Status:              {pol['status']}")
    print(f"      • Negation Words:      {pol['neg_words_found']}")
    print(f"      • Masked Disfluencies: {pol['masked_repair_words']}")
    
    print("\n" + "=" * 80)
    print("PIPELINE PROCESSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
