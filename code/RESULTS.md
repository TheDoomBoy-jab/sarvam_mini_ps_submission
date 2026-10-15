# Reproduction Guide & Benchmark Results: Simultaneous S2ST for Conversational Hindi-English

This document provides exact, single-command deterministic instructions to reproduce every measurement, statistical parameter, unit test, and benchmark figure reported in the submission for the **Inter IIT Tech Meet 15.0 – Sarvam.ai Challenge (Mini Problem Statement)**.

All commands run deterministically using standard Python 3 / `uv`. Every reported number is backed by physical audio data, CTC trellis alignments, or measured streaming clock timings.

---

## 0. Quick-Start: Single-Command Full Pipeline Reproduction

From the repository root (`/Users/aahannayak/sarvam`):

```bash
# 1. Verify environment & dependencies
export PYTHONPATH="code:code/scripts:/Users/aahannayak/sarvam/.venv/lib/python3.14/site-packages"

# 2. Phase 1: Acoustic Conditioning & DSP Sweet Spot
python3 code/scripts/acoustic_conditioning.py
python3 code/scripts/audit_tts_audio.py

# 3. Phase 2: Word Boundary Hand Audit (30 boundaries, bootstrap CI, signed bias)
python3 code/scripts/audit_boundaries.py report

# 4. Phase 3: RGNPE Engine, 42-Edge-Case Test Suite, & 5 Acoustic Stress Tests
python3 code/scripts/rgnpe_engine.py
python3 code/tests/test_rgnpe_edge_cases.py
python3 code/scripts/run_stress_tests.py

# 5. Phase 4: Wait-k Simulation, Pareto Frontier Plot, & Measured Streaming Rollout
python3 code/scripts/wait_k_simulation.py
python3 code/scripts/streaming_rollout.py
```

---

## Phase 1: Audio Ingestion & Acoustic Conditioning

### Objective & Constraints
Standardize raw conversational Hindi-English speech into competition-compliant physical audio:
- **Sampling Rate:** $24,000\text{ Hz}$ mono, 16-bit linear PCM (`PCM_16`).
- **Integrated Loudness Sweet Spot:** $-20.00\text{ LUFS} \pm 0.3\text{ LUFS}$ (ITU-R BS.1770-4 standard).
- **True Peak Ceiling:** $\le -1.0\text{ dBTP}$ ($0.891$ linear amplitude ceiling to prevent DAC inter-sample clipping).
- **Causal Bandpass Filter:** 4th-order Butterworth bandpass ($80\text{ Hz} - 11,000\text{ Hz}$) implemented using Second-Order Sections (SOS) with state vector ($z_i$) carried across frames (strictly causal, zero lookahead).
- **Streaming Alignment:** Zero-padded to exact multiples of $80\text{ ms}$ ($1,920\text{ samples}$ / $3,840\text{ bytes}$ per frame). **Zero remainder samples** across all files.

### Command to Reproduce
```bash
python3 code/scripts/acoustic_conditioning.py
```

### Empirical Results Table

| Clip ID | Master Filename | Speaker Profile | Duration (s) | Raw LUFS | Conditioned LUFS | Peak Level (dBTP) | 80 ms Frames | Remainder Samples |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rec_000001` | `clip_01_male_transfer.wav` | Male (Deep) | $19.68$ | $-24.32$ | $\mathbf{-20.01}$ | $-1.06$ | $246$ | $\mathbf{0}$ |
| `rec_000002` | `clip_02_female_phone.wav` | Female (Fast) | $15.04$ | $-22.18$ | $\mathbf{-20.00}$ | $-1.12$ | $188$ | $\mathbf{0}$ |
| `rec_000003` | `clip_03_male_property.wav` | Male (Colloquial) | $16.56$ | $-23.85$ | $\mathbf{-20.01}$ | $-1.18$ | $207$ | $\mathbf{0}$ |
| `rec_000004` | `clip_04_female_cheque.wav` | Female (Urgent) | $11.28$ | $-25.10$ | $\mathbf{-20.00}$ | $-1.45$ | $141$ | $\mathbf{0}$ |
| `rec_000005` | `clip_05_male_account.wav` | Male (Formal) | $13.76$ | $-24.05$ | $\mathbf{-20.01}$ | $-1.21$ | $172$ | $\mathbf{0}$ |
| `rec_000006` | `clip_06_female_investment.wav` | Female (Casual) | $10.24$ | $-21.94$ | $\mathbf{-20.02}$ | $-1.84$ | $128$ | $\mathbf{0}$ |

*Output directory:* `code/data/standardized/`

### TTS Audio Quality, Mispronunciation & Artefact Gate
To satisfy Sarvam's explicit rule (*"If you use TTS-generated speech anywhere, check it and filter or regenerate bad output. TTS can mispronounce names and numbers, skip or repeat words, or add artefacts."*), we run an automated 5-gate audit across all generated audio:
```bash
python3 code/scripts/audit_tts_audio.py
```
1. **Acoustic Saturation Gate:** Confirms zero clipped samples and true peak $\le -1.0\text{ dBTP}$.
2. **Pacing & Cadence Gate:** Measures character speaking rate ($7.0 - 12.0\text{ chars/sec}$); flags vocoder stalls ($< 5\text{ c/s}$) or hyper-speed runaway ($> 25\text{ c/s}$).
3. **Autoregressive Loop Gate:** Detects repeating token loops and stutter artefacts (threshold: $\le 3$ repeats).
4. **CTC Trellis Word Alignment Gate:** Verifies $0$ skipped/unaligned words across the entire synthesis.
5. **Entity & Numeral Integrity Gate:** Validates exact digit strings, phone sequences, and lakh/crore units.

*Result:* **6/6 Clips Verified Clean (100.0%)**, 0 rejected for regeneration (`code/data/tts_audit_results.json`).

> [!NOTE]
> **Physical Audio vs. 200h Dataset Proposal:** The 6 physical master audio clips in `code/data/standardized/` strictly preserve wideband broadcast audio ($80\text{ Hz} - 11\text{ kHz}$, $-20.00\text{ LUFS}$, true peak $\le -1.06\text{ dBTP}$). The telephone-band simulation ($300 - 3,400\text{ Hz}$ G.711) and Room Impulse Response (RIR) convolution ($0.2 - 0.5\text{s}$ RT60 with SI-SDR) are part of our **theoretical 200-hour dataset architecture proposal (Part C)** to answer how to bridge the synthetic-to-real gap.

---

## Phase 2: Verbatim Transcripts, MMS-FA CTC Trellis & Boundary Audit

### Objective & Architecture
- **Transcription Standard:** IndicVoices Level 1 verbatim style (captures disfluencies, repetitions, stutters, loanwords, and conversational fillers).
- **Forced Alignment Model:** Meta MMS 300M multilingual CTC model (`torchaudio.pipelines.MMS_FA`) with `uroman` romanizer (`lcode="hin"` for accurate schwa deletion).
- **Model Versioning & DVC:** The 1.2 GB Meta MMS weights are tracked via DVC (`code/models/mms_fa/mms_300m.pt.dvc`) to keep the Git repository lightweight (< 5 MB). Run `python3 code/scripts/fetch_model.py` to verify or audit the model artifact.
- **Trellis Alignment Performance:** $\mathbf{193 / 193\text{ words aligned}}$ ($\mathbf{100.0\%}$ alignment success rate; $\mathbf{0}$ uniform-fallback clips).
- **Tripartite Breaker Resolution:** Resolves speech repairs via contiguous `abandoned + marker + corrected` acoustic segmentation, yielding physical `safe_to_speak_sec` timestamps.

### Command to Reproduce Audit
```bash
python3 code/scripts/audit_boundaries.py report
```

### Empirical 30-Boundary Hand Audit Statistics (`code/data/boundary_audit.csv`)
- **Boundary Criterion:** Acoustic energy cessation of word $w_1$ (decay into silence closure or unvoiced consonant burst).
- **Sample Size ($N$):** $30$ randomly sampled word transitions (seed $42$).
- **Mean Absolute Error (MAE):** $\mathbf{19.8\text{ ms}}$ (sample standard deviation $15.3\text{ ms}$, $\text{ddof}=1$).
- **95% Bootstrap Confidence Interval on MAE:** $\mathbf{[14.6, 25.4]\text{ ms}}$ ($10,000$ Monte Carlo resamples).
- **Median Absolute Error:** $20.0\text{ ms}$ | **P90 Error:** $42.2\text{ ms}$ | **Max Error:** $59.0\text{ ms}$.
- **Signed Bias:** $\mathbf{-11.8\text{ ms}}$, 95% Bootstrap CI $[-19.8, -3.8]\text{ ms}$.
  - *Linguistic cause:* CTC acoustic posteriors exhibit sharp peaks mid-phone, leading to early boundary predictions prior to vocalic energy cessation.
  - Aligner was early in $20/30$ boundaries ($66.7\%$), late in $10/30$ boundaries ($33.3\%$).
- **Streaming Frame Compliance:**
  - Within $20\text{ ms}$ ($1$ CTC frame): $43.3\%$
  - Within $40\text{ ms}$ ($0.5$ streaming frame): $\mathbf{86.7\%}$
  - Within $80\text{ ms}$ ($1.0$ streaming frame): $\mathbf{100.0\%}$
  - *MAE is $\mathbf{4.0\times}$ finer than the $80\text{ ms}$ streaming frame.*
- **Measurement Resolution:** $60\%$ of readings fall on a $5\text{ ms}$ grid; effective human reading resolution is $\sim 5\text{ ms}$.
- **Independent Spot-Check (Audacity):** $7$ independent boundaries re-measured: mean absolute difference $\mathbf{3.9\text{ ms}}$, max difference $6.0\text{ ms}$, $100\%$ within $10\text{ ms}$.

---

## Phase 3: Translation Targets & Hard Content / RGNPE Engine

### Objective & Architecture
Rule-Guided Neural Post-Editing (RGNPE) enforces deterministic entity preservation and self-repair handling:
1. **Symbolic Hard-Content Locking:** AI4Bharat XML Tag Masking (`<span translate="no">`) isolates numbers, currencies, phone digits, and account numbers from stochastic neural translation.
2. **Contextual Polarity Guard:** Distinguishes conversational tag particles (*'ना'* as filler) from correlative negation (*'ना...ना'*), and masks repair markers (*'नहीं नहीं'*) before polarity evaluation.
3. **Dual-Target Semantics:** Generates `en_text` (committing only the corrected entity as fact) and `en_alt` (preserving the monotonic streaming clause order).

### Research Basis: Movie Captions Technique for Target Alignment
In cinema subtitling (e.g., Netflix and BBC subtitling guidelines), subtitlers solve the cross-lingual SOV $\to$ SVO word-order mismatch under strict reading cadence limits ($12 - 17\text{ char/s}$, closely matching our $7 - 12\text{ char/s}$ audio cadence gate). Subtitlers segment dialogue along natural syntactic clause boundaries (prepositional and noun phrases), preventing compound numbers or verbs from being unnaturally severed. In our Part C dataset research proposal, we recommend mining bilingual movie dialogue subtitles to curate thousands of code-mixed, conversational training pairs pre-aligned for monotonic streaming.

### Commands to Reproduce
```bash
# 1. Run RGNPE engine over master manifest
python3 code/scripts/rgnpe_engine.py

# 2. Run the 42-edge-case unit test suite
python3 code/tests/test_rgnpe_edge_cases.py

# 3. Run the 5 real-world acoustic stress tests
python3 code/scripts/run_stress_tests.py
```

### Edge-Case Unit Test Suite Results (`code/tests/test_rgnpe_edge_cases.py`)
- **Total Test Cases:** $42$ | **Passed:** $\mathbf{42 / 42}$ ($\mathbf{100.0\%}$)
  - *Group 1 (Fractional Scales):* *पौने दो लाख* ($1,75,000$), *सवा दो करोड़* ($2,25,00,000$), *साढ़े तीन लाख* ($3,50,000$), *डेढ़ करोड़* ($1,50,00,000$), *ढाई लाख* ($2,50,000$) $\to$ **10/10 Passed**.
  - *Group 2 (Compound Amounts):* *एक करोड़ बीस लाख* ($1.2\text{ Cr}$), *दो करोड़ पचास लाख* ($2.5\text{ Cr}$), *पैंतीस लाख* ($35\text{ L}$) $\to$ **5/5 Passed**.
  - *Group 3 (Spoken Digits & Multipliers):* *डबल नाइन* ($99$), *ट्रिपल पाँच शून्य* ($5550$), Devanagari English loanwords (*'फोर फाइव टू वन'*) $\to$ **10/10 Passed**.
  - *Group 4 (Polarity & Breakers):* End-of-sentence negation, repair marker *'नहीं'* masked, filler *'ना'* disambiguated $\to$ **9/9 Passed**.
  - *Group 5 (Self-Repair Semantics):* Dual targets, breaker metadata formatting, slot index safety $\to$ **8/8 Passed**.

### 5-Stress-Test Benchmark (`code/scripts/run_stress_tests.py`)
Tested against real acoustic corruptions synthesized into 24 kHz WAVs (`code/data/stress_tests/`):
- **STRESS_01 (50 Hz Electrical Mains Buzz + Laughter):** *पौने दो करोड़* abandoned, *सवा करोड़* locked $\to$ **100% Passed**.
- **STRESS_02 (Street Traffic Rumble + Speech Stutter):** *98204156* retracted, *9920415678* locked $\to$ **100% Passed**.
- **STRESS_03 (Room Reverb + Throat Clearing Cough + Reversal):** Cheque *452103* locked, negative cancellation locked $\to$ **100% Passed**.
- **STRESS_04 (60 Hz Ground Loop Buzz + Whisper Cadence):** Account *3330987654* with 'ट्रिपल' locked, negative polarity locked $\to$ **100% Passed**.
- **STRESS_05 (Cafe Babble + Deep Sigh Vocalization):** *साढ़े बारह लाख* ($1,250,000$) locked, negative polarity locked $\to$ **100% Passed**.

---

## Phase 4: Streaming Latency, Wait-$k$ Simulation & Meta-Seamless Topicalization

### Benchmark Utterance
- **Source (Hindi SOV, $|x|=10$):** *"मैंने कल अपना पुराना बैंक खाता बंद करवा दिया ।"*
- **Target (English SVO, $|y|=8$, $\gamma=0.80$):** *"I closed my old bank account yesterday ."*
- **Topicalized Target (Our Architecture, $|y|=8$):** *"My old bank account yesterday I closed ."*

### Research Basis: Human Brain Cognitive Parallel (Gile's Effort Model & Working Memory)
Why does Fixed Wait-$k$ fail on Hindi? In cognitive psychology and simultaneous conference interpretation, human working memory (phonological loop) cannot store more than 3–5 seconds of incoming speech without cognitive overload (Gile's Effort Model). 

When translating from a verb-final language (Hindi SOV) to a verb-medial language (English SVO), human interpreters never wait for the sentence-final verb because their working memory would overflow. Instead, the human brain dynamically fronts the topic (*"Regarding my old bank account... yesterday... I got it closed"*), maintaining an Ear-Voice Span (EVS) of ~2–3 seconds. Our proposed Meta-Seamless topicalization directly mirrors this human cognitive architecture, dropping latency to $\text{AL} = 666.2\text{ ms}$ with $0.0\%$ premature error.

### Commands to Reproduce
```bash
# 1. Run mathematical schedule derivation & plot generation
python3 code/scripts/wait_k_simulation.py

# 2. Run real-time streaming rollout replay with audio-clock measurements
python3 code/scripts/streaming_rollout.py
```

### Latency & Pareto Benchmark Results (`code/results/phase4_latency_metrics.json`)

| Policy Architecture | Token AL | Time AL (ms) | AP | Premature Emission (%) | Verb Hallucination? | On Pareto Frontier? | Role in Evaluation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed Wait-1** | $0.125$ | $51.2$ | $0.450$ | $25.0\%$ | YES (Emits at step 2, verb at 7) | Yes (Low Latency) | Baseline (STACL) |
| **Fixed Wait-2** | $1.125$ | $461.2$ | $0.550$ | $25.0\%$ | YES (Emits at step 3, verb at 7) | Yes | Baseline (STACL) |
| **Fixed Wait-3** | $\mathbf{2.125}$ | $\mathbf{871.2}$ | $\mathbf{0.650}$ | $\mathbf{12.5\%}$ | **YES (Verb Cliff Failure)** | No (Dominated) | Mini PS Mandatory Baseline |
| **Fixed Wait-4** | $3.250$ | $1332.5$ | $0.738$ | $12.5\%$ | YES | No | Baseline (STACL) |
| **Fixed Wait-5** | $4.375$ | $1793.8$ | $0.812$ | $12.5\%$ | YES | No | Baseline (STACL) |
| **Fixed Wait-6** | $5.500$ | $2255.0$ | $0.875$ | $12.5\%$ | YES | No | Baseline (STACL) |
| **Fixed Wait-7** | $6.625$ | $2716.2$ | $0.925$ | $0.0\%$ | NO (Waited for verb) | Yes | Baseline (STACL) |
| **Fixed Wait-8** | $7.750$ | $3177.5$ | $0.963$ | $0.0\%$ | NO | Yes | Baseline (STACL) |
| **Fixed Wait-9** | $8.875$ | $3638.8$ | $0.988$ | $0.0\%$ | NO | Yes | Baseline (STACL) |
| **StreamSpeech Adaptive** | $\mathbf{3.125}$ | $\mathbf{1281.2}$ | $\mathbf{0.750}$ | $\mathbf{0.0\%}$ | **NO (Semantic Gated)** | **Yes (Pareto Optimal)** | Literature Benchmark (ACL 2024) |
| **Meta-Seamless Topicalization** | $\mathbf{1.625}$ | $\mathbf{666.2}$ | $\mathbf{0.600}$ | $\mathbf{0.0\%}$ | **NO (Topicalized Fronting)**| **Yes (Pareto Optimal)** | **OUR SUBMITTED ARCHITECTURE** |

### Measured Physical Streaming Rollout (`code/results/phase4_streaming_rollout.json`)
Measured dynamically over real-time 80 ms audio frames in `code/scripts/streaming_rollout.py`:
- **Policy:** Meta-Seamless Topicalization (**Our Submitted Architecture**)
- **Scheduled AL:** $\mathbf{666.2\text{ ms}}$
- **Observed AL (Audio Clock):** $\mathbf{696.2\text{ ms}}$ (includes $80\text{ ms}$ hop quantization)
- **First Emission:** $1,280\text{ ms}$ of audio
- **Final Emission:** $4,160\text{ ms}$ of audio
- **Processing Overhead (Wall - Audio Clock):** Mean $\mathbf{3.1\text{ ms}}$, Max $\mathbf{5.5\text{ ms}}$
- **Total Mouth-to-Ear Latency:** $\mathbf{798.4\text{ ms}}$ ($696.2\text{ ms}$ observed AL + $80\text{ ms}$ frame step + $3.1\text{ ms}$ processing lag + $19.1\text{ ms}$ streaming neural vocoder synthesis)

---

## Artifact & Output File Inventory

| Output File Path | Description | Generated By |
| :--- | :--- | :--- |
| `code/data/standardized/*.wav` | 6 standardized master audio clips (24 kHz, -20 LUFS, -1 dBTP) | `acoustic_conditioning.py` |
| `code/data/aligned/*_aligned.json` | 6 CTC trellis word alignments with timestamps and breaker annotations | `forced_alignment.py` |
| `code/data/boundary_audit.csv` | 30 hand-measured acoustic boundaries with notes and errors | `audit_boundaries.py` |
| `code/data/manifest_rgnpe.json` | Master manifest with locked/retracted entities and verified targets | `rgnpe_engine.py` |
| `code/results/read_write_matrix.csv` | Step-by-step reading and emission schedule matrix across all policies | `wait_k_simulation.py` |
| `code/results/phase4_latency_metrics.json`| Exact token AL, time AL, AP, premature rates, and Pareto flags | `wait_k_simulation.py` |
| `code/results/latency_vs_accuracy.png` | Publication-quality 300 DPI plot of AL vs Quality & Premature Emission | `wait_k_simulation.py` |
| `code/results/phase4_streaming_rollout.json`| Physical audio-clock and wall-clock emission timestamps | `streaming_rollout.py` |

---

## Mathematical Formulations Used

### 1. Token-Based Average Lagging (AL) [Ma et al., ACL 2019 / SimulEval]
$$\text{AL} = \frac{1}{\tau} \sum_{t=1}^{\tau} \left( g(t) - \frac{t-1}{\gamma} \right)$$
where $\gamma = \frac{|y|}{|x|}$, and $\tau = \min \{ t \mid g(t) = |x| \}$.

### 2. Time-Based Average Lagging [SimulEval Standard]
$$\text{AL}_{\text{time}} = \frac{1}{\tau} \sum_{t=1}^{\tau} \left( d(t) - \frac{t-1}{|y|} \cdot T \right)$$
where $d(t)$ is the physical arrival timestamp of the last source word read before target token $t$ is emitted, and $T$ is the total source audio duration.

### 3. Average Proportion (AP) [Cho & Esipova, 2016]
$$\text{AP} = \frac{1}{|x| \cdot |y|} \sum_{t=1}^{|y|} g(t)$$
 measures the absolute fraction of the input space consumed before emission.
