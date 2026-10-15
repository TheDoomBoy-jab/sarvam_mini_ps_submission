# Simultaneous Speech-to-Speech Translation (Hindi → English)
### Inter IIT Tech Meet 15.0 – Sarvam.ai Challenge (Mini Problem Statement)
**Author:** Aahan Nayak | **Submission Artifact:** [`aahan_nayak_mini_ps.pdf`](aahan_nayak_mini_ps.pdf) (4-page camera-ready report)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Evaluation](https://img.shields.io/badge/Evaluation-Inter_IIT_Tech_Meet_15.0-success.svg)](https://sarvam.ai)
[![MMS CTC Alignment](https://img.shields.io/badge/Meta_MMS_Alignment-100%25_(193%2F193)-brightgreen.svg)](code/data/aligned/)
[![RGNPE Tests](https://img.shields.io/badge/RGNPE_Unit_Tests-42%2F42_Passed-brightgreen.svg)](code/tests/test_rgnpe_edge_cases.py)
[![Latency](https://img.shields.io/badge/Observed_AL-696.2_ms-orange.svg)](code/results/phase4_streaming_rollout.json)

---

## Executive Summary

This repository contains the complete, production-grade submission for the **Sarvam.ai Mini Problem Statement (Inter IIT Tech Meet 15.0)**: a causal, simultaneous speech-to-speech translation (S2ST) pipeline for conversational Hindi-English speech with strict Indian numeral entity locking, self-correction handling, and sub-second streaming latency.

### Core Breakthroughs & Empirical Highlights
1. **The Hindi SOV "Verb Cliff" Proof:** Proved mathematically that Fixed Wait-$k$ ($k \le 5$) collapses on Hindi $\to$ English (forcing a **$74.1\%$ premature verb hallucination rate**). Paralleled this to **Gile's Effort Model** in human conference interpretation (working memory limits $\le 3\text{--}5\text{s}$).
2. **Meta-Seamless Topicalization:** Achieves **Scheduled $\text{AL} = 666.2\text{ ms}$**, **Observed $\text{AL} = 696.2\text{ ms}$**, and **Mean Processing Lag $= 3.1\text{ ms}$** over real-time 80 ms audio frames with **$0.0\%$ premature emission**.
3. **Causal Acoustic Conditioning:** 4th-order Butterworth bandpass ($80\text{ Hz} - 11\text{ kHz}$) using Second-Order Sections (SOS) with frame-to-frame filter state vector ($z_i$) preservation, ITU-R BS.1770-4 loudness compliance ($-20.00\text{ LUFS} \pm 0.02$), true peak $\le -1.06\text{ dBTP}$, and **zero remainder samples** across all 80 ms frames.
4. **CTC Trellis Forced Alignment & Hand Audit:** Meta MMS 300M multilingual model aligned $193/193$ words ($100.0\%$). Hand audit of $30$ acoustic boundaries shows $\text{MAE} = 19.8\text{ ms}$ ($95\%$ bootstrap CI $[14.6, 25.4]\text{ ms}$, signed bias $-11.8\text{ ms}$), with **$100\%$ within the 80 ms streaming frame**.
5. **Rule-Guided Neural Post-Editing (RGNPE):** $42/42$ unit tests passed ($100.0\%$), $5/5$ acoustic stress tests passed ($100.0\%$), and $6/6$ manifest checks passed with zero entity drops on *lakh*, *crore*, and 10-digit phone numbers.

---

##  Quick-Start: Single-Command Full Pipeline Reproduction

Every metric, figure, and table reported in the 4-page submission PDF can be reproduced deterministically with standard Python 3:

```bash
# 1. Environment setup (Python 3.10+ / uv)
# uv sync   # or: pip install -e code/

# 2. Set PYTHONPATH (from repository root)
export PYTHONPATH=".:code:code/scripts:$PYTHONPATH"

# 3. Run all phases end-to-end
python3 code/scripts/acoustic_conditioning.py
python3 code/scripts/audit_tts_audio.py
python3 code/scripts/audit_boundaries.py report
python3 code/scripts/rgnpe_engine.py
python3 code/tests/test_rgnpe_edge_cases.py
python3 code/scripts/run_stress_tests.py
python3 code/scripts/wait_k_simulation.py
python3 code/scripts/streaming_rollout.py
```

For complete per-metric reproduction commands and mathematical proofs, see **[`code/RESULTS.md`](code/RESULTS.md)**.

---

## Part A Mandatory Handwritten Appendix Scan

Per the problem statement specification (*"In the appendix, handwritten: for 'मैंने कल अपना पुराना बैंक खाता बंद करवा दिया', write the wait-3 output schedule and compute its Average Lagging"*):

- **High-Resolution Scan:** [`code/results/appendix_wait3_handwritten.png`](code/results/appendix_wait3_handwritten.png)
- **Mathematical Summary:**
  - Utterance: *"मैंने कल अपना पुराना बैंक खाता बंद करवा दिया ।"* ($|x| = 10$ words, MMS CTC end times: $410\text{ ms}$ to $4,100\text{ ms}$)
  - Target (SVO): *"I closed my old bank account yesterday ."* ($|y| = 8$ words, $\gamma = |y|/|x| = 0.80$)
  - Schedule $g(t)$: $g = [3, 4, 5, 6, 7, 8, 9, 10]$, $\tau = 8$.
  - **Token $\text{AL} = \mathbf{2.125\text{ tokens}}$** | **Time $\text{AL} = \mathbf{871.2\text{ ms}}$** | **$\text{AP} = \mathbf{0.650}$**.
  - **SOV Verb Cliff Failure:** English token 2 (*"closed"*) is emitted at step 4 (after hearing *"पुराना"*), forcing a premature verb guess because the Hindi verb *"बंद करवाना"* does not arrive until words 7–8.

---

##  Repository Structure

```
.
├── README.md                         # Top-level project landing page & quick reproduction
├── RESULTS.md                        # Master reproduction guide with commands & metrics
├── aahan_nayak_mini_ps.pdf           # 4-page camera-ready camera PDF report
├── code/
│   ├── README.md                     # Codebase documentation & architecture guide
│   ├── RESULTS.md                    # Detailed benchmark reproduction log
│   ├── data/
│   │   ├── raw/                      # Raw conversational recordings
│   │   ├── standardized/             # 24 kHz mono, -20 LUFS, -1 dBTP, 80 ms framed WAVs
│   │   ├── aligned/                  # Meta MMS-FA CTC trellis word timestamps & breaker metadata
│   │   ├── boundary_audit.csv        # 30 hand-measured acoustic boundaries with notes
│   │   ├── boundary_audit_plots/     # 30 boundary waveform/spectrogram plots & HTML viewer
│   │   ├── manifest.json             # Canonical metadata manifest
│   │   ├── manifest_rgnpe.json       # Enriched manifest with locked & retracted entities
│   │   ├── rgnpe_authored_targets.json # Authored English targets & clause chunks
│   │   ├── stress_tests/             # 5 physical acoustic corruptions (hum, traffic, cough, cafe)
│   │   └── test/                     # Sarah 24k test clip for live interview demonstration
│   ├── models/
│   │   └── mms_fa/
│   │       ├── mms_300m.pt.dvc       # DVC tracking pointer for 1.2 GB Meta MMS 300M weights
│   │       └── .gitignore            # Excludes 1.2 GB checkpoint from Git commits
│   ├── results/
│   │   ├── read_write_matrix.csv     # Step-by-step reading and emission schedule matrix
│   │   ├── phase4_latency_metrics.json # Computed AL, AP, and Pareto metrics across 11 policies
│   │   ├── latency_vs_accuracy.png   # Publication-quality 300 DPI Pareto frontier plot
│   │   ├── appendix_wait3_handwritten.png # Handwritten Wait-3 schedule & AL proof
│   │   ├── pdf_preview/              # Rendered 150 DPI page preview PNGs
│   │   └── phase4_streaming_rollout.json # Measured audio-clock & wall-clock streaming rollout
│   ├── scripts/
│   │   ├── fetch_model.py            # DVC model verification & torchaudio initialization utility
│   │   ├── acoustic_conditioning.py  # Causal 4th-order Butterworth bandpass & LUFS normalizer
│   │   ├── audit_tts_audio.py        # 5-gate TTS audio quality, clipping & cadence filter
│   │   ├── forced_alignment.py       # Meta MMS 300M CTC trellis aligner with uroman
│   │   ├── audit_boundaries.py       # Statistical boundary audit, bootstrap CI, signed bias
│   │   ├── rgnpe_extractors.py       # Hindi numeral, multiplier, and polarity parsers
│   │   ├── rgnpe_engine.py           # Entity locking, XML placeholder masking, and verification
│   │   ├── run_stress_tests.py       # 5 acoustic stress tests (hum, traffic, reverb, whisper, cafe)
│   │   ├── wait_k_simulation.py      # SimulEval AL/AP latency & Pareto simulation
│   │   ├── streaming_rollout.py      # Real-time streaming rollout replay with measured lag
│   │   ├── run_test_clip_suite.py    # Automated end-to-end verification suite on test clip
│   │   ├── interactive_test.py       # Interactive live-testing playground for any Hindi sentence
│   │   └── generate_pdf_report.py    # Generates 4-page camera-ready aahan_nayak_mini_ps.pdf
│   ├── tests/
│   │   └── test_rgnpe_edge_cases.py  # 42 unit tests covering all linguistic and numeral edge cases
│   └── pyproject.toml                # Project dependencies and configuration
```

---

##  Live Interview Demonstration & Testing on New Clips

The challenge rules state: *"in the interview you will run your code on a new clip and explain every part of it."*

We provide dedicated live-testing utilities:

### 1. Automated Pipeline Test on Sarah 24k Audio Clip
```bash
python3 code/scripts/run_test_clip_suite.py
```
*Runs all 5 stages end-to-end: Acoustic DSP $\to$ Meta MMS CTC Alignment $\to$ RGNPE Entity Extraction $\to$ Contextual Polarity Guard $\to$ Streaming Schedule.*

### 2. Interactive Sentence Playground
Test any arbitrary conversational Hindi or Hinglish sentence on the fly:
```bash
python3 code/scripts/interactive_test.py "मेरा नया मोबाइल नंबर नोट करो +91 डबल नाइन डबल आठ शून्य चार पाँच छह सात आठ"
python3 code/scripts/interactive_test.py "उन्होंने प्रॉपर्टी के लिए पौने तीन करोड़ रुपये मांगे हैं पर बजट सिर्फ सवा दो करोड़ है"
python3 code/scripts/interactive_test.py "अकाउंट में पच्चीस लाख अरे नहीं नहीं वेट पैंतीस लाख ट्रांसफर करने हैं"
```

---

##  Benchmark Summary: Policy Latency & Pareto Frontier

Evaluated across 11 streaming policies on the benchmark utterance (*"मैंने कल अपना पुराना बैंक खाता बंद करवा दिया ।"*):

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

---

##  License & Attribution
- Built for **Inter IIT Tech Meet 15.0 – Sarvam.ai Challenge**.
- Code released under the [MIT License](LICENSE).
