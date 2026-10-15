import os
import json
import urllib.request
import subprocess

ELEVEN_KEY = "sk_1b015e021e634d04deec3e99a4c64ddce900f47cf4757bfb"

# Voice assignments with varied personalities
CLIPS = [
    {
        "id": "clip_01",
        "name": "clip_01_male_transfer",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb", # George (Warm/Expressive Male)
        "text": "अरे सुनो... म-म-मुझे ना राहुल शर्मा के अकाउंट में... मतलब... पच्चीस लाख... अरे नहीं नहीं, वेट! पच्चीस नहीं, पैंतीस लाख रुपये ट्रांसफर करवाने थे... लेकिन मेरे फोन पे ना कोई ओटीपी आया ही नहीं।",
        "has_chatter": True
    },
    {
        "id": "clip_02",
        "name": "clip_02_female_phone",
        "voice_id": "EXAVITQu4vr4xnSDxMaL", # Sarah (Expressive Female)
        "text": "मेरा रजिस्टर्ड मोबाइल नंबर... अह्ह... नोट करो... नौ आठ दो शून्य चार... एक पाँच छह सात आठ। पर बैंक वालों ने ना मेरा खाता अनफ्रीज करवाने के बाद भी... [sigh] ...कोई कन्फर्मेशन मैसेज नहीं भेजा।",
        "has_chatter": False
    },
    {
        "id": "clip_03",
        "name": "clip_03_male_property",
        "voice_id": "IKne3meq5aSn9XLyUdCD", # Charlie (Deep Male with laughter)
        "text": "उन्होंने वसंत कुंज वाली प्रॉपर्टी का ना... पहले डेढ़ करोड़ रुपये कोट किया था... [laughter] हा हा, अरे सीरियसली! पर बाद में हमने नेगोशिएट करके एक करोड़ बीस लाख में डील फाइनल करवा ली।",
        "has_chatter": True
    },
    {
        "id": "clip_04",
        "name": "clip_04_female_cheque",
        "voice_id": "Xb7hH8MSUJpSbSDYk0k2", # Alice (Energetic Female)
        "text": "प्लीज वो चेक नंबर... चार पाँच दो एक शून्य तीन का पेमेंट स्टॉप करवा... अरे सॉरी सॉरी! वो चेक तो ऑलरेडी क्लियर हो चुका है, उसको कैंसल-वैंसल मत करना!",
        "has_chatter": True
    },
    {
        "id": "clip_05",
        "name": "clip_05_male_account",
        "voice_id": "TX3LPaxmHKxFdv7VOQHJ", # Liam (Conversational Male)
        "text": "मेरा नया सेविंग्स खाता नंबर है... उम्म... तीन शून्य नौ आठ... सात छह पाँच चार... तीन दो एक शून्य। और सुनो, उसपे पुराना डेबिट कार्ड लिंक मत कर देना गलती से भी।",
        "has_chatter": False
    },
    {
        "id": "clip_06",
        "name": "clip_06_female_investment",
        "voice_id": "pFZP5JQG7iQjIQuC4Bku", # Lily (Calm Female)
        "text": "म्यूचुअल फंड्स में ना... बेसिकली मैंने पिछले महीने दस लाख रुपये इन्वेस्ट-वाए थे... [sigh] ...पर इस बार मार्केट डाउन है, तो मुझे कोई नया इन्वेस्टमेंट बिल्कुल नहीं करना।",
        "has_chatter": False
    }
]

def synthesize_clip(clip_info):
    raw_mp3 = f"code/data/raw/{clip_info['name']}_speech.mp3"
    final_wav = f"code/data/raw/{clip_info['name']}.wav"
    
    print(f"Calling ElevenLabs for {clip_info['id']} ({clip_info['name']})...")
    payload = {
        "text": clip_info["text"],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.32,          # Lower stability creates realistic human inflection & pauses
            "similarity_boost": 0.78,
            "style": 0.35,              # Expressive style for laughter/sighs
            "use_speaker_boost": True
        }
    }
    
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{clip_info['voice_id']}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as resp:
        with open(raw_mp3, "wb") as f:
            f.write(resp.read())
            
    # Mix with realistic acoustic traffic/ambient street layer via FFmpeg
    # anoisesrc creates subtle traffic rumble, aeval adds low frequency road buzz
    # volume=0.04 keeps ambient noise realistic (-24dB SNR) so speech is clean but grounded in noise
    filter_complex = (
        "[1:a]volume=0.035,lowpass=f=2500,highpass=f=60[noise];"
        "[0:a][noise]amix=inputs=2:duration=first:dropout_transition=2,"
        "highpass=f=80,lowpass=f=11000[out]"
    )
    
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-i", raw_mp3,
        "-f", "lavfi", "-i", "anoisesrc=c=pink:r=24000:a=0.08",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "24000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        final_wav
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(raw_mp3)
    print(f"  -> Generated & Mixed: {final_wav}")

def main():
    os.makedirs("code/data/raw", exist_ok=True)
    for c in CLIPS:
        synthesize_clip(c)
    print("\nPhase 1 Complete: All 6 stress-tested clips generated!")

if __name__ == "__main__":
    main()
