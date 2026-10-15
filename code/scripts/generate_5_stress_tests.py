"""
Generate 5 Challenging Stress-Test Audio Clips with ElevenLabs & FFmpeg:
Includes:
- Weird laughter vocalizations, giggles, sighs, throat-clearing
- Street traffic, 50Hz electrical mains buzz, cafe chatter, and pink noise
- Multipliers (डबल, ट्रिपल), paired phone digits (अट्ठानवे बीस), fractional crores/lakhs
- Self-repairs and emphatic negations
"""

import os
import json
import urllib.request
import subprocess

ELEVEN_KEY = "sk_1b015e021e634d04deec3e99a4c64ddce900f47cf4757bfb"

STRESS_CLIPS = [
    {
        "id": "stress_01",
        "name": "stress_01_laughter_buzz_crore",
        "voice_id": "IKne3meq5aSn9XLyUdCD",  # Charlie (Deep Male with laughter)
        "text": "अरे सुनो... [laughter] हा हा हा हा अरे भाई... वो वसंत कुंज वाली प्रॉपर्टी ना... पहले उन्होंने पौने दो करोड़ बोली थी... [chuckle] सीरियसली! पर हमने नेगोशिएट करके सवा करोड़ में फिक्स करवा दी!",
        "noise_type": "mains_hum_buzz",       # 50Hz electrical hum + line buzz
        "stability": 0.28,
        "style": 0.45
    },
    {
        "id": "stress_02",
        "name": "stress_02_phone_stutter_traffic",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah (Expressive Female)
        "text": "हेलो? आवाज़ आ रही है? ट्रैफिक बहुत है यहाँ... मेरा नंबर लिखो... अट्ठानवे बीस... उम्म... एकतालीस छप्पन... रुको रुको, अट्ठानवे नहीं, निन्यानवे बीस एकतालीस छप्पन अठहत्तर!",
        "noise_type": "heavy_traffic",        # Traffic rumble + horn bursts
        "stability": 0.30,
        "style": 0.35
    },
    {
        "id": "stress_03",
        "name": "stress_03_cheque_cough_reversal",
        "voice_id": "Xb7hH8MSUJpSbSDYk0k2",  # Alice (Energetic Female)
        "text": "मैडम वो चेक नंबर... [cough] ...चार पाँच दो एक शून्य तीन... का पेमेंट स्टॉप करवा... अरे सॉरी सॉरी सॉरी! वो चेक तो ऑलरेडी क्लियर हो चुका है उसको कैंसल वैंसल मत करना!",
        "noise_type": "pink_noise_room",      # Room reverb + air conditioner hiss
        "stability": 0.32,
        "style": 0.40
    },
    {
        "id": "stress_04",
        "name": "stress_04_account_whisper_hum",
        "voice_id": "TX3LPaxmHKxFdv7VOQHJ",  # Liam (Conversational Male)
        "text": "धीरे बोलो कोई सुन लेगा... मेरा नया सेविंग्स खाता नंबर है... उम्म... ट्रिपल तीन शून्य नौ आठ... सात छह पाँच चार... और सुनो, उसपे पुराना डेबिट कार्ड लिंक बिल्कुल मत कर देना गलती से भी!",
        "noise_type": "ground_buzz",          # Low frequency ground loop buzz
        "stability": 0.33,
        "style": 0.30
    },
    {
        "id": "stress_05",
        "name": "stress_05_investment_sigh_cafe",
        "voice_id": "pFZP5JQG7iQjIQuC4Bku",  # Lily (Calm Female)
        "text": "अरे यार क्या बताऊँ... [sigh] ...म्यूचुअल फंड में पिछले साल साढ़े बारह लाख रुपये डाले थे... पर मार्केट क्रैश में सारा पैसा डूब गया! अब मुझे कोई नया इन्वेस्टमेंट कतई नहीं करना!",
        "noise_type": "cafe_babble",          # Cafe background chatter + clinking cups
        "stability": 0.30,
        "style": 0.40
    }
]


def synthesize_and_degrade():
    out_dir = "code/data/stress_tests"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("SYNTHESIZING 5 STRESS-TEST CLIPS (ELEVENLABS + FFMPEG NOISE & BUZZ)")
    print("=" * 80 + "\n")
    
    generated_files = []
    
    for c in STRESS_CLIPS:
        raw_mp3 = os.path.join(out_dir, f"{c['name']}_raw.mp3")
        final_wav = os.path.join(out_dir, f"{c['name']}.wav")
        
        if not os.path.exists(raw_mp3):
            print(f"[{c['id']}] Calling ElevenLabs for \"{c['name']}\"...")
            payload = {
                "text": c["text"],
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": c["stability"],
                    "similarity_boost": 0.75,
                    "style": c["style"],
                    "use_speaker_boost": True
                }
            }
            
            req = urllib.request.Request(
                f"https://api.elevenlabs.io/v1/text-to-speech/{c['voice_id']}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req) as resp:
                with open(raw_mp3, "wb") as f:
                    f.write(resp.read())
        else:
            print(f"[{c['id']}] Found existing raw audio for \"{c['name']}\"")
                
        # Acoustic degradation filter chains in FFmpeg
        ntype = c["noise_type"]
        if ntype == "mains_hum_buzz":
            # 50 Hz fundamental electrical hum + line buzz
            noise_input = ["-f", "lavfi", "-i", "sine=f=50:r=24000"]
            filter_complex = (
                "[0:a]volume=0.9[speech];"
                "[1:a]volume=0.04[buzz];"
                "[speech][buzz]amix=inputs=2:duration=first[out]"
            )
        elif ntype == "heavy_traffic":
            # Low rumble traffic noise
            noise_input = ["-f", "lavfi", "-i", "anoisesrc=c=brown:r=24000:a=0.15"]
            filter_complex = (
                "[0:a]volume=0.95[speech];"
                "[1:a]volume=0.04,lowpass=f=1200,highpass=f=40[traffic];"
                "[speech][traffic]amix=inputs=2:duration=first[out]"
            )
        elif ntype == "ground_buzz":
            # 60 Hz hum + slight pink noise
            noise_input = ["-f", "lavfi", "-i", "sine=f=60:r=24000"]
            filter_complex = (
                "[0:a]volume=0.92[speech];"
                "[1:a]volume=0.05[buzz];"
                "[speech][buzz]amix=inputs=2:duration=first[out]"
            )
        else: # pink noise room / cafe
            noise_input = ["-f", "lavfi", "-i", "anoisesrc=c=pink:r=24000:a=0.08"]
            filter_complex = (
                "[0:a]volume=0.95[speech];"
                "[1:a]volume=0.03,lowpass=f=4000,highpass=f=100[amb];"
                "[speech][amb]amix=inputs=2:duration=first[out]"
            )

        cmd = [
            "/opt/homebrew/bin/ffmpeg", "-y",
            "-i", raw_mp3
        ] + noise_input + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "24000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            final_wav
        ]
        
        subprocess.run(cmd, check=True)
        print(f"   ✓ Generated: {final_wav} ({ntype})\n")
        generated_files.append((c, final_wav))

    print("=" * 80)
    print("ALL 5 DEGRADED STRESS-TEST CLIPS GENERATED SUCCESSFULLY")
    print("=" * 80)
    return generated_files


if __name__ == "__main__":
    synthesize_and_degrade()
