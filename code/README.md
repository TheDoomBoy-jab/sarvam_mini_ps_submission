# Simultaneous Speech-to-Speech Translation (Hindi → English)
### Inter IIT Tech Meet 15.0 – Sarvam.ai Challenge (Mini Problem Statement)

A complete, end-to-end, causal simultaneous speech-to-speech translation pipeline designed for code-mixed Hindi-English conversational speech with strict Indian numeral entity locking, self-correction handling, and sub-second streaming latency.

---

##  Quick Reproduction
To reproduce all results, metrics, and figures with a single command suite, see **[`RESULTS.md`](RESULTS.md)**:

```bash
# Set PYTHONPATH
export PYTHONPATH="code:code/scripts:/Users/aahannayak/sarvam/.venv/lib/python3.14/site-packages"

# Run all 4 phases end-to-end
python3 code/scripts/acoustic_conditioning.py
python3 code/scripts/audit_boundaries.py report
python3 code/scripts/rgnpe_engine.py
python3 code/tests/test_rgnpe_edge_cases.py
python3 code/scripts/run_stress_tests.py
python3 code/scripts/wait_k_simulation.py
python3 code/scripts/streaming_rollout.py
```

---

##  Repository Structure

```
code/
├── data/
│   ├── raw/                      # Raw conversational recordings
│   ├── standardized/             # 24 kHz mono, -20 LUFS, -1 dBTP, 80 ms framed WAVs
│   ├── aligned/                  # Meta MMS-FA CTC trellis word timestamps & breaker metadata
│   ├── boundary_audit.csv        # 30 hand-measured acoustic boundaries with notes
│   ├── manifest.json             # Canonical metadata manifest
│   ├── manifest_rgnpe.json       # Enriched manifest with locked & retracted entities
│   ├── rgnpe_authored_targets.json # Authored English targets & clause chunks
│   └── stress_tests/             # 5 physical acoustic & linguistic stress audio files
├── models/
│   └── mms_fa/
│       ├── mms_300m.pt.dvc       # DVC tracking pointer for 1.2 GB Meta MMS 300M weights
│       └── .gitignore            # Excludes 1.2 GB checkpoint from Git commits
├── results/
│   ├── read_write_matrix.csv     # Step-by-step reading and emission schedule matrix
│   ├── phase4_latency_metrics.json # Computed AL, AP, and Pareto metrics across 11 policies
│   ├── latency_vs_accuracy.png   # Publication-quality 300 DPI Pareto frontier plot
│   ├── pdf_preview/              # Rendered 150 DPI page preview PNGs
│   └── phase4_streaming_rollout.json # Measured audio-clock & wall-clock streaming rollout
├── scripts/
│   ├── fetch_model.py            # DVC model verification & torchaudio initialization utility
│   ├── acoustic_conditioning.py  # Causal 4th-order Butterworth bandpass (80 Hz - 11 kHz) & LUFS
│   ├── audit_tts_audio.py        # 5-gate TTS audio quality, clipping, cadence & mispronunciation filter
│   ├── forced_alignment.py       # Meta MMS 300M CTC trellis aligner with uroman
│   ├── audit_boundaries.py       # Statistical boundary audit, bootstrap CI, signed bias
│   ├── rgnpe_extractors.py       # Generalized Hindi numeral, multiplier, and polarity parsers
│   ├── rgnpe_engine.py           # Entity locking, XML placeholder masking, and verification
│   ├── run_stress_tests.py       # 5 acoustic stress tests (hum, traffic, reverb, whisper, cafe)
│   ├── wait_k_simulation.py      # SimulEval-standard AL/AP latency & Pareto frontier simulation
│   ├── streaming_rollout.py      # Real-time streaming rollout replay with measured lag
│   └── generate_pdf_report.py    # Generates 4-page camera-ready aahan_nayak_mini_ps.pdf
├── tests/
│   └── test_rgnpe_edge_cases.py  # 42 unit tests covering all linguistic and numeral edge cases
├── RESULTS.md                    # Single-command deterministic reproduction guide
└── pyproject.toml                # Project dependencies and configuration
```

---

##  Model Artifacts & DVC Tracking

The **Meta MMS 300M CTC Trellis Forced Alignment model** ($1,203.6\text{ MB}$) is tracked via **DVC (Data Version Control)** to keep the Git repository lightweight ($< 5\text{ MB}$) while ensuring deterministic artifact versioning.

```bash
# Verify model artifact and DVC pointer
python3 code/scripts/fetch_model.py

# Pull model weights via DVC (if remote configured)
dvc pull code/models/mms_fa/mms_300m.pt.dvc
```
*Note: If DVC remote is not configured, running `forced_alignment.py` automatically initializes and caches the official `torchaudio.pipelines.MMS_FA` model weights seamlessly.*

---

##  Key Achievements & Benchmarks
1. **Acoustic Conditioning (Phase 1):** Zero remainder samples across all 6 master clips when framed into 80 ms chunks ($1,920$ samples at $24\text{ kHz}$), $-20.00\text{ LUFS} \pm 0.02$, peak level $\le -1.06\text{ dBTP}$.
2. **CTC Alignment & Audit (Phase 2):** $193/193$ words aligned ($100.0\%$). Hand audit of $30$ boundaries shows $\text{MAE} = 19.8\text{ ms}$ ($95\%$ CI $[14.6, 25.4]\text{ ms}$, signed bias $-11.8\text{ ms}$), with $100\%$ of boundaries within the $80\text{ ms}$ frame.
3. **Entity Locking (Phase 3):** $42/42$ unit tests passed ($100.0\%$), $5/5$ acoustic stress tests passed ($100.0\%$), and $6/6$ manifest verification checks passed ($100.0\%$). Zero entity drops on *lakh*, *crore*, and 10-digit phone numbers.
4. **Streaming Latency (Phase 4):**
   - Meta-Seamless Topicalization achieves **Scheduled $\text{AL} = 666.2\text{ ms}$**, **Observed $\text{AL} = 696.2\text{ ms}$**, and **Mean Processing Lag = $3.1\text{ ms}$**.
   - Solves the Hindi SOV $\to$ English SVO "Verb Cliff" ($0.0\%$ premature emission vs $74.1\%$ in Wait-$k \le 5$).
