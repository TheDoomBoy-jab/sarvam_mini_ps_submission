"""
Streaming rollout replay (Phase 4 companion): causal front-end + schedule-driven emission + measured latency.

WHAT THIS IS
  A real-time *replay* of one Phase 4 read/write schedule on the benchmark sentence
  ("मैंने कल अपना पुराना बैंक खाता बंद करवा दिया ।"). It is not a translator: the English chunks come from the
  Phase 4 policy's target/schedule, and "source words read" comes from per-word end timestamps.
  Feeding it a different utterance will still emit the benchmark target (a warning is raised when the
  audio length does not match the benchmark).

PIPELINE (producer / consumer threads)
  1. CausalStreamingFilter  - causal Butterworth bandpass (order=4 per band edge -> 8th order overall), SOS
                              state carried across 80 ms frames, no lookahead (a causal IIR still has group delay).
  2. AudioRingBuffer        - thread-safe circular buffer; the consumer reads the latest 400 ms of context
                              (logged as RMS level; nothing in this replay's policy depends on it).
  3. ScheduleStreamer       - groups target tokens that share a read count into chunks and releases a chunk once
                              that many source words have been read.
  4. ParallelPipelinedEngine- paced producer (a frame is released only when its last sample would exist in real
                              time), consumer that emits chunks, and a shared clock for wall-time measurements.

WHAT IS MEASURED (nothing is hard-coded)
  scheduled AL   : the policy's schedule x word timestamps (should equal the Phase 4 AL)
  observed AL    : the same AL formula on the audio-clock time at which each token was actually emitted
                   (differs from scheduled by 80 ms hop quantisation)
  processing lag : wall-clock emission time minus audio-clock time (pipeline overhead)

Usage:
  python streaming_rollout.py                       # synthetic noise, benchmark timing, Meta_Seamless policy
  python streaming_rollout.py --audio clip.wav --policy StreamSpeech_Adaptive --speed 1
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 24000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1,920
LOW_HZ, HIGH_HZ, FILTER_ORDER = 80.0, 11000.0, 4

# Phase 2/4 benchmark: end time (ms) of each of the 10 source words (~410 ms/word).
DEFAULT_WORD_END_MS = [410.0 * i for i in range(1, 11)]
# Built-in copy of the Phase 4 Meta_Seamless policy, used only if the Phase 4 JSON is unavailable.
BUILTIN_POLICY = "Meta_Seamless"
BUILTIN_TARGET = ["My", "old", "bank", "account", "yesterday", "I", "closed", "."]
BUILTIN_SCHEDULE = [3, 4, 5, 6, 6, 6, 8, 10]


# ---------------------------------------------------------------------------
# 1. Causal streaming filter
# ---------------------------------------------------------------------------
class CausalStreamingFilter:
    """Causal SOS bandpass with state carried across frames."""

    def __init__(self, sr: int = SAMPLE_RATE, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ, order: int = FILTER_ORDER):
        self.sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
        self._zi_unit = sosfilt_zi(self.sos)
        self.zi: Optional[np.ndarray] = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float64)
        if self.zi is None:  # start as if the signal had always been frame[0]: no startup click
            self.zi = self._zi_unit * frame[0]
        out, self.zi = sosfilt(self.sos, frame, zi=self.zi)
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Ring buffer
# ---------------------------------------------------------------------------
class AudioRingBuffer:
    """Thread-safe circular buffer of the most recent audio."""

    def __init__(self, capacity_frames: int = 128):
        self.capacity = capacity_frames * FRAME_SAMPLES
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_ptr = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write_chunk(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n > self.capacity:
            raise ValueError("chunk larger than ring buffer capacity")
        with self.lock:
            end = self.write_ptr + n
            if end <= self.capacity:
                self.buffer[self.write_ptr:end] = samples
            else:
                first = self.capacity - self.write_ptr
                self.buffer[self.write_ptr:] = samples[:first]
                self.buffer[:n - first] = samples[first:]
            self.write_ptr = end % self.capacity
            self.total_written += n

    def get_latest_context(self, context_ms: int = 400) -> np.ndarray:
        with self.lock:
            n = min(int(SAMPLE_RATE * context_ms / 1000.0), self.total_written, self.capacity)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self.write_ptr - n) % self.capacity
            if start + n <= self.capacity:
                return self.buffer[start:start + n].copy()
            return np.concatenate([self.buffer[start:], self.buffer[:n - (self.capacity - start)]])


# ---------------------------------------------------------------------------
# 3. Schedule-driven streamer
# ---------------------------------------------------------------------------
def detok(tokens: List[str]) -> str:
    return re.sub(r"\s+([.,!?])", r"\1", " ".join(tokens))


def build_chunks(target: List[str], schedule: List[int]) -> List[Dict[str, Any]]:
    """Group consecutive target tokens that share the same read count g(t) into one emitted chunk."""
    if len(target) != len(schedule):
        raise ValueError("target and schedule must have the same length")
    if any(b < a for a, b in zip(schedule, schedule[1:])):
        raise ValueError("schedule must be non-decreasing")
    chunks: List[Dict[str, Any]] = []
    for i, (word, g) in enumerate(zip(target, schedule)):
        if chunks and chunks[-1]["required_source_words"] == g:
            chunks[-1]["tokens"].append(word)
            chunks[-1]["token_idx"].append(i)
        else:
            chunks.append({"required_source_words": g, "tokens": [word], "token_idx": [i]})
    for n, c in enumerate(chunks, 1):
        c["clause_id"], c["text"] = n, detok(c["tokens"])
    return chunks


class ScheduleStreamer:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks, self.emitted = chunks, 0

    def check_emissions(self, words_read: int, flush: bool = False) -> List[Dict[str, Any]]:
        ready = []
        while self.emitted < len(self.chunks):
            c = self.chunks[self.emitted]
            if flush or words_read >= c["required_source_words"]:
                ready.append(c)
                self.emitted += 1
            else:
                break
        return ready


# ---------------------------------------------------------------------------
# 4. Pipelined engine
# ---------------------------------------------------------------------------
class ParallelPipelinedEngine:
    def __init__(self, samples: np.ndarray, word_end_ms: List[float], chunks: List[Dict[str, Any]],
                 speed: float = 1.0, audio_filter: Optional[CausalStreamingFilter] = None, verbose: bool = True):
        self.samples = samples
        self.word_end_ms = np.asarray(word_end_ms, dtype=np.float64)
        self.filter = audio_filter or CausalStreamingFilter()
        self.ring = AudioRingBuffer()
        self.streamer = ScheduleStreamer(chunks)
        self.speed, self.verbose = speed, verbose
        self.q: "queue.Queue[Optional[int]]" = queue.Queue()
        self.stop = threading.Event()
        self.rollout_log: List[Dict[str, Any]] = []
        self.error: Optional[BaseException] = None
        self.t0 = 0.0

    def _producer(self) -> None:
        try:
            n_frames = math.ceil(len(self.samples) / FRAME_SAMPLES)  # keep the partial tail frame
            for f in range(n_frames):
                if self.stop.is_set():
                    break
                audio_ms = (f + 1) * FRAME_MS
                # A frame exists only once its last sample has been captured: wait, then process.
                delay = self.t0 + audio_ms / 1000.0 / self.speed - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                frame = self.samples[f * FRAME_SAMPLES:(f + 1) * FRAME_SAMPLES]
                if len(frame) < FRAME_SAMPLES:
                    frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
                self.ring.write_chunk(self.filter.process_frame(frame))
                self.q.put(audio_ms)
        except BaseException as exc:  # noqa: BLE001 - re-raised in run()
            self.error = exc
        finally:
            self.q.put(None)  # sentinel is always sent, so the consumer can never hang

    def _consumer(self) -> None:
        last_audio_ms, words_read = 0, 0
        try:
            while True:
                item = self.q.get()
                eos = item is None
                if not eos:
                    last_audio_ms = int(item)
                    words_read = int(np.searchsorted(self.word_end_ms, last_audio_ms, side="right"))
                ready = self.streamer.check_emissions(words_read, flush=eos)
                for c in ready:
                    self._log(c, last_audio_ms, words_read, flushed=eos)
                if eos:
                    break
        except BaseException as exc:  # noqa: BLE001
            self.error = self.error or exc
            self.stop.set()

    def _log(self, chunk: Dict[str, Any], audio_ms: int, words_read: int, flushed: bool) -> None:
        wall_ms = (time.perf_counter() - self.t0) * 1000.0
        ctx = self.ring.get_latest_context(400)
        rms = float(np.sqrt(np.mean(ctx.astype(np.float64) ** 2))) if len(ctx) else 0.0
        entry = {
            "clause_id": chunk["clause_id"],
            "english_chunk": chunk["text"],
            "token_idx": chunk["token_idx"],
            "required_source_words": chunk["required_source_words"],
            "source_words_read": words_read,
            "audio_ms": audio_ms,
            "wall_ms": round(wall_ms, 1),
            "processing_lag_ms": round(wall_ms - audio_ms / self.speed, 1),
            "context_rms_dbfs": round(20 * math.log10(max(rms, 1e-9)), 1),
            "flushed_at_end_of_stream": flushed,
        }
        self.rollout_log.append(entry)
        if self.verbose:
            tag = " (flushed at end of stream)" if flushed else ""
            print(f"  [audio {audio_ms:6d} ms | wall {wall_ms:8.1f} ms] chunk #{chunk['clause_id']}: "
                  f"\"{chunk['text']}\"  (needs {chunk['required_source_words']} words, read {words_read}){tag}")

    def run(self) -> None:
        self.t0 = time.perf_counter()
        tp = threading.Thread(target=self._producer, name="producer")
        tc = threading.Thread(target=self._consumer, name="consumer")
        tp.start()
        tc.start()
        tp.join()
        tc.join()
        if self.error is not None:
            raise RuntimeError(f"streaming pipeline failed: {self.error!r}") from self.error


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def al_ms(d_ms: List[float], read_counts: List[int], total_ms: float, lx: int) -> float:
    """Time-based AL: (1/tau) * sum_{t<=tau} ( d(t) - (t-1) * T/|y| ); tau = first token emitted after the
    full source was read (falls back to |y|)."""
    ly = len(d_ms)
    tau = next((t for t, r in enumerate(read_counts, 1) if r >= lx), ly)
    return sum(d_ms[t - 1] - (t - 1) * total_ms / ly for t in range(1, tau + 1)) / tau


def compute_metrics(engine: ParallelPipelinedEngine, schedule: List[int], word_end_ms: List[float],
                    phase4_al: Optional[float]) -> Dict[str, Any]:
    lx, ly, total = len(word_end_ms), len(schedule), word_end_ms[-1]
    scheduled = al_ms([word_end_ms[g - 1] for g in schedule], schedule, total, lx)

    d: List[Optional[float]] = [None] * ly
    reads: List[Optional[int]] = [None] * ly
    for e in engine.rollout_log:
        for i in e["token_idx"]:
            d[i], reads[i] = e["audio_ms"], e["source_words_read"]
    if any(v is None for v in d):
        raise RuntimeError("not every target token was emitted")
    # If the stream ended before every required word was read, flushed tokens carry the end-of-stream
    # time, so an observed AL would be meaningless: report n/a instead.
    flushed = any(e["flushed_at_end_of_stream"] for e in engine.rollout_log)
    observed = None if flushed else al_ms(d, reads, total, lx)  # type: ignore[arg-type]

    lags = [e["processing_lag_ms"] for e in engine.rollout_log]
    return {
        "scheduled_al_ms": round(scheduled, 1),
        "phase4_json_al_ms": phase4_al,
        "scheduled_matches_phase4": None if phase4_al is None else abs(scheduled - phase4_al) < 0.5,
        "observed_al_ms_audio_clock": None if observed is None else round(observed, 1),
        "stream_ended_before_source_fully_read": flushed,
        "first_emission_audio_ms": min(e["audio_ms"] for e in engine.rollout_log),
        "last_emission_audio_ms": max(e["audio_ms"] for e in engine.rollout_log),
        "mean_processing_lag_ms": round(float(np.mean(lags)), 1),
        "max_processing_lag_ms": round(float(np.max(lags)), 1),
        "frame_ms": FRAME_MS,
    }


# ---------------------------------------------------------------------------
# 6. Inputs and entry point
# ---------------------------------------------------------------------------
def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf  # only needed for --audio

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        frac = Fraction(SAMPLE_RATE, sr).limit_denominator(1000)
        mono = resample_poly(mono, frac.numerator, frac.denominator).astype(np.float32)
    return mono


def load_policy(json_path: Path, policy: str) -> Tuple[List[str], List[int], Optional[float], str]:
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if policy not in data:
            raise SystemExit(f"policy {policy!r} not in {json_path}; available: {sorted(data)}")
        r = data[policy]
        try:
            rel_src = str(json_path.resolve().relative_to(HERE.parent.parent))
        except ValueError:
            rel_src = str(json_path.name)
        return list(r["target"]), [int(g) for g in r["schedule"]], r.get("average_lagging_ms"), rel_src
    if policy != BUILTIN_POLICY:
        raise SystemExit(f"{json_path} not found and no built-in copy of policy {policy!r}")
    print(f"WARNING: {json_path} not found; using the built-in copy of {BUILTIN_POLICY}.")
    return list(BUILTIN_TARGET), list(BUILTIN_SCHEDULE), None, "builtin"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=None, help="wav file (default: synthetic noise of benchmark length)")
    ap.add_argument("--policy", default=BUILTIN_POLICY, help="key in the Phase 4 metrics JSON (e.g. Wait_3)")
    ap.add_argument("--schedule-json", type=Path, default=HERE.parent / "results" / "phase4_latency_metrics.json")
    ap.add_argument("--timestamps", type=Path, default=None, help="JSON list of per-word end times in ms")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier (1.0 = real time)")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "results")
    args = ap.parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")

    target, schedule, phase4_al, policy_src = load_policy(args.schedule_json, args.policy)
    word_end_ms = json.loads(args.timestamps.read_text()) if args.timestamps else DEFAULT_WORD_END_MS
    word_end_ms = [float(x) for x in word_end_ms]
    if max(schedule) > len(word_end_ms):
        raise SystemExit(f"schedule reads up to word {max(schedule)} but only {len(word_end_ms)} timestamps given")
    total_ms = word_end_ms[-1]

    warnings: List[str] = []
    if args.audio:
        samples, audio_label = load_audio(args.audio), str(args.audio)
        dur_ms = len(samples) / SAMPLE_RATE * 1000.0
        if abs(dur_ms - total_ms) > max(200.0, 0.15 * total_ms):
            warnings.append(f"audio is {dur_ms:.0f} ms but the benchmark timing ends at {total_ms:.0f} ms: "
                            "this replay assumes the benchmark sentence")
    else:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 0.05, int(math.ceil(total_ms / 1000.0 * SAMPLE_RATE))).astype(np.float32)
        audio_label = "synthetic_noise"
    for w in warnings:
        print(f"WARNING: {w}")

    chunks = build_chunks(target, schedule)
    print("=" * 96)
    print("STREAMING ROLLOUT REPLAY (schedule-driven; latency measured, not assumed)")
    print("=" * 96)
    print(f"policy: {args.policy} [{policy_src}] | audio: {audio_label} | hop: {FRAME_MS} ms | speed: {args.speed}x")
    print(f"target: {' '.join(target)}\nschedule g(t): {schedule}\n")

    engine = ParallelPipelinedEngine(samples, word_end_ms, chunks, speed=args.speed)
    engine.run()
    m = compute_metrics(engine, schedule, word_end_ms, phase4_al)

    print("\n" + "-" * 96)
    print(f"{'#':<3}{'CHUNK':<30}{'NEEDS':>6}{'AUDIO ms':>10}{'WALL ms':>10}{'PROC LAG ms':>13}{'CTX dBFS':>10}")
    for e in engine.rollout_log:
        print(f"{e['clause_id']:<3}{e['english_chunk']:<30}{e['required_source_words']:>6}{e['audio_ms']:>10}"
              f"{e['wall_ms']:>10.1f}{e['processing_lag_ms']:>13.1f}{e['context_rms_dbfs']:>10.1f}")
    print("-" * 96)
    print(f"scheduled AL (schedule x timestamps): {m['scheduled_al_ms']:.1f} ms")
    if phase4_al is not None:
        print(f"Phase 4 JSON AL                     : {phase4_al:.1f} ms  [{'match' if m['scheduled_matches_phase4'] else 'MISMATCH'}]")
    if m["observed_al_ms_audio_clock"] is None:
        print("observed AL (audio clock)           : n/a (stream ended before all required source words were read)")
    else:
        print(f"observed AL (audio clock)           : {m['observed_al_ms_audio_clock']:.1f} ms  (includes {FRAME_MS} ms hop quantisation)")
    print(f"first / last emission (audio clock) : {m['first_emission_audio_ms']} / {m['last_emission_audio_ms']} ms")
    print(f"processing lag (wall - audio), mean/max: {m['mean_processing_lag_ms']:.1f} / {m['max_processing_lag_ms']:.1f} ms")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "phase4_streaming_rollout.json"
    out_path.write_text(json.dumps({
        "policy": args.policy, "policy_source": policy_src, "audio": audio_label, "speed": args.speed,
        "replay_of_benchmark_sentence": True, "target": target, "schedule": schedule,
        "word_end_ms": word_end_ms, "warnings": warnings, "metrics": m, "rollout_log": engine.rollout_log,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ {out_path}")


if __name__ == "__main__":
    main()
