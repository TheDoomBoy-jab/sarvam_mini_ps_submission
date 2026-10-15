"""
Generate a test audio clip using ElevenLabs to verify Phase 3 edge cases.
"""

import os
import json
import urllib.request
import subprocess

ELEVEN_KEY = "sk_1b015e021e634d04deec3e99a4c64ddce900f47cf4757bfb"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah (Expressive Female)

# Rich conversational Hindi sentence exercising edge cases:
# - Spoken phone with multipliers ('डबल नाइन')
# - Fractional crore amounts ('पौने तीन करोड़', 'सवा दो करोड़')
# - Self-repair marker ('अरे नहीं नहीं वेट')
TEXT = "सुनो... मेरा नया मोबाइल नंबर नोट करो, डबल नाइन डबल आठ शून्य चार पाँच छह सात आठ। और हाँ, उस प्रॉपर्टी के लिए उन्होंने पहले पौने तीन करोड़ मांगे थे... अरे नहीं नहीं वेट, सवा दो करोड़ में बात पक्की हुई है!"

def generate_test_audio():
    out_dir = "code/data/test"
    os.makedirs(out_dir, exist_ok=True)
    raw_mp3 = os.path.join(out_dir, "test_clip_raw.mp3")
    final_wav = os.path.join(out_dir, "test_clip_sarah_24k.wav")
    
    print(f"Calling ElevenLabs API for voice Sarah ({VOICE_ID})...")
    payload = {
        "text": TEXT,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.80,
            "style": 0.30,
            "use_speaker_boost": True
        }
    }
    
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as resp:
        with open(raw_mp3, "wb") as f:
            f.write(resp.read())
    print(f"Saved raw ElevenLabs MP3: {raw_mp3}")
    
    # Standardize to 24 kHz mono 16-bit PCM WAV using FFmpeg
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-i", raw_mp3,
        "-ar", "24000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        final_wav
    ]
    subprocess.run(cmd, check=True)
    print(f"Successfully generated 24 kHz mono WAV: {final_wav}")
    return final_wav

if __name__ == "__main__":
    generate_test_audio()
