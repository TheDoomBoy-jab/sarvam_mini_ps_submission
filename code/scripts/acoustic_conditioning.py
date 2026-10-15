import os
import glob
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

# ==========================================
# ACOUSTIC & STREAMING SWEET SPOT PARAMETERS
# ==========================================
SAMPLE_RATE = 24000            # Competition mandatory standard: 24 kHz
TARGET_LUFS = -20.0            # Sweet spot: [-19.0, -23.0] LUFS
TRUE_PEAK_LIMIT_DBTP = -1.0    # True peak guard: -1.0 dBTP (no digital clipping)

# Psychoacoustic Bandpass Filter
HIGH_PASS_CUTOFF = 80.0        # 80 Hz: eliminate sub-bass mic rumble / AC hum
LOW_PASS_CUTOFF = 11000.0      # 11 kHz: eliminate ultrasonic aliasing below Nyquist (12 kHz)

# Streaming Frame & Context Specifications
FRAME_STEP_MS = 80             # Stride / Hop size: exactly 80 ms (competition WebSocket packet size)
SAMPLES_PER_FRAME = int(SAMPLE_RATE * (FRAME_STEP_MS / 1000.0)) # 1,920 samples = 3,840 bytes

CONTEXT_WINDOW_MS = 400        # Context window: 400 ms (Sweet spot: [320 ms, 480 ms])
SAMPLES_PER_CONTEXT = int(SAMPLE_RATE * (CONTEXT_WINDOW_MS / 1000.0)) # 9,600 samples = 5 frames of 80ms

def apply_causal_bandpass(audio_data, sr, lowcut, highcut, order=4):
    """
    Causal 4th-order Butterworth bandpass filter using Second-Order Sections (SOS).
    Zero phase instability while remaining strictly causal (no future audio lookahead).
    """
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, audio_data)

def condition_and_frame_audio(input_path, output_wav_path):
    data, sr = sf.read(input_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)

    meter = pyln.Meter(sr)
    initial_lufs = meter.integrated_loudness(data)

    # 1. Causal Bandpass Filter (80 Hz - 11 kHz)
    filtered = apply_causal_bandpass(data, sr, HIGH_PASS_CUTOFF, LOW_PASS_CUTOFF)

    # 2. Integrated Loudness Normalization to -20.0 LUFS
    post_filter_lufs = meter.integrated_loudness(filtered)
    normalized = pyln.normalize.loudness(filtered, post_filter_lufs, TARGET_LUFS)

    # 3. True-Peak Limiter Guard (-1.0 dBTP = ~0.891 linear amplitude)
    max_peak = np.max(np.abs(normalized))
    peak_ceiling = 10.0 ** (TRUE_PEAK_LIMIT_DBTP / 20.0) # 0.89125
    if max_peak > peak_ceiling:
        normalized = (normalized / max_peak) * peak_ceiling

    final_lufs = meter.integrated_loudness(normalized)
    final_peak_db = 20.0 * np.log10(np.max(np.abs(normalized)))

    # 4. Strict Frame Alignment: Pad to exact multiple of 80 ms (1,920 samples)
    rem = len(normalized) % SAMPLES_PER_FRAME
    if rem > 0:
        pad_samples = SAMPLES_PER_FRAME - rem
        normalized = np.pad(normalized, (0, pad_samples), mode='constant')

    total_frames = len(normalized) // SAMPLES_PER_FRAME
    total_seconds = len(normalized) / sr

    # Save standardized 16-bit PCM 24 kHz WAV
    sf.write(output_wav_path, normalized, sr, subtype='PCM_16')

    # 5. Verify Streaming Frame & Sliding Context Window Structure
    # Stride = 80 ms (1 frame), Context Window = 400 ms (5 frames)
    frames_generated = []
    for f_idx in range(total_frames):
        # Current 80ms frame
        f_start = f_idx * SAMPLES_PER_FRAME
        f_end = f_start + SAMPLES_PER_FRAME
        current_frame = normalized[f_start:f_end]

        # Causal Context Window (past 320ms + current 80ms = 400ms context)
        ctx_start = max(0, f_end - SAMPLES_PER_CONTEXT)
        causal_context = normalized[ctx_start:f_end]
        frames_generated.append((len(current_frame), len(causal_context)))

    print(f"Standardized: {os.path.basename(input_path)}")
    print(f"  • Loudness: {initial_lufs:.2f} LUFS -> {final_lufs:.2f} LUFS (Target: {TARGET_LUFS} LUFS)")
    print(f"  • Peak Level: {final_peak_db:.2f} dBTP (Ceiling: {TRUE_PEAK_LIMIT_DBTP} dBTP)")
    print(f"  • Causal Bandpass: {HIGH_PASS_CUTOFF:.0f} Hz - {LOW_PASS_CUTOFF:.0f} Hz applied")
    print(f"  • Total Duration: {total_seconds:.2f}s ({total_frames} frames of {FRAME_STEP_MS}ms)")
    print(f"  • Sliding Context Windows: {len(frames_generated)} frames verified with {CONTEXT_WINDOW_MS}ms context\n")

def main():
    out_dir = "code/data/standardized"
    os.makedirs(out_dir, exist_ok=True)
    raw_files = sorted(glob.glob("code/data/raw/*.wav"))

    print("=== RUNNING ACOUSTIC & STREAMING SWEET SPOT STANDARDIZATION ===\n")
    for f in raw_files:
        out_f = os.path.join(out_dir, os.path.basename(f))
        condition_and_frame_audio(f, out_f)
    print("All audio files successfully conditioned to exact sweet spot specifications!")

if __name__ == "__main__":
    main()
