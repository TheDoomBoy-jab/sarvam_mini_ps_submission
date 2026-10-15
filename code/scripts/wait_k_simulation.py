"""
Phase 4: Streaming Latency, Wait-k Simulation & Latency/Quality Plot (Part E - ML Track)

Implements:
1. Average Lagging (AL), token-based and time-based [Ma et al., ACL 2019 / SimulEval]
2. Average Proportion (AP) [Cho & Esipova, 2016]
3. Fixed Wait-k schedules (k = 1..9)
4. StreamSpeech-style adaptive schedule (verb-gated), DERIVED from a source-target alignment
5. Meta-Seamless-style topicalized schedule, DERIVED from a source-target alignment
6. A computed Pareto frontier and a publication-style figure

What is computed vs. what is not
--------------------------------
COMPUTED from the schedules:  AL (tokens, ms), AP, premature-emission rate, verb-premature flag.
NOT computed (placeholders):  BLEU. These are illustrative values for plotting only, unless you
                              supply measured numbers with --quality measured_bleu.json
                              ({"Wait_3": 24.8, "StreamSpeech_Adaptive": 37.9, ...}).
The JSON output and the figure both state which source the BLEU values came from.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 1. Benchmark sentence, acoustic timestamps and alignments
# ---------------------------------------------------------------------------
SOURCE_TOKENS = ["मैंने", "कल", "अपना", "पुराना", "बैंक", "खाता", "बंद", "करवा", "दिया", "।"]
# End time (ms) of each source word, from the Phase 2 MMS CTC alignment (~410 ms/word).
SOURCE_TIMESTAMPS_MS = [410, 820, 1230, 1640, 2050, 2460, 2870, 3280, 3690, 4100]
assert len(SOURCE_TOKENS) == len(SOURCE_TIMESTAMPS_MS)

LX = len(SOURCE_TOKENS)

# Standard SVO English target.
SVO_TARGET = ["I", "closed", "my", "old", "bank", "account", "yesterday", "."]
# REQUIRED[t] = 1-based index of the last source word that must be read before target word t
# can be emitted faithfully. "closed" needs the whole verb complex "बंद करवा" (words 7-8).
SVO_REQUIRED = [1, 8, 3, 4, 5, 6, 2, 10]
SVO_VERB_IDX = 1  # 0-based position of "closed"

# Topicalized English target (clause fronting), reordered to be near-monotonic w.r.t. the source.
META_TARGET = ["My", "old", "bank", "account", "yesterday", "I", "closed", "."]
META_REQUIRED = [3, 4, 5, 6, 2, 1, 8, 10]
META_VERB_IDX = 6

assert len(SVO_TARGET) == len(SVO_REQUIRED) and len(META_TARGET) == len(META_REQUIRED)
LY = len(SVO_TARGET)  # both targets have the same length, so gamma is shared

WAIT_K_VALUES = list(range(1, 10))

# Illustrative BLEU placeholders (NOT measured). Overridable with --quality.
ILLUSTRATIVE_BLEU = {
    "Wait_1": 19.6, "Wait_2": 22.4, "Wait_3": 25.1, "Wait_4": 28.0, "Wait_5": 30.8,
    "Wait_6": 33.6, "Wait_7": 40.0, "Wait_8": 40.5, "Wait_9": 41.0,
    "StreamSpeech_Adaptive": 38.6, "Meta_Seamless": 37.2,
}


# ---------------------------------------------------------------------------
# 2. Metric definitions
# ---------------------------------------------------------------------------
def cutoff_step(g, lx):
    """tau(|x|): first 1-based target step at which the full source has been read
    (falls back to |y| if the policy never reads the whole source)."""
    for t, read in enumerate(g, 1):
        if read >= lx:
            return t
    return len(g)


def compute_al(g, lx, ly):
    """Token-based AL: (1/tau) * sum_{t<=tau} ( g(t) - (t-1)/gamma ),  gamma = |y|/|x|."""
    gamma = ly / lx
    tau = cutoff_step(g, lx)
    return sum(g[t - 1] - (t - 1) / gamma for t in range(1, tau + 1)) / tau


def compute_al_ms(g, times_ms, lx, ly):
    """Time-based AL using real source timestamps (SimulEval-style):
    (1/tau) * sum_{t<=tau} ( d(t) - (t-1) * T/|y| ),
    where d(t) = end time of the last source word read when emitting target word t."""
    total = times_ms[-1]
    tau = cutoff_step(g, lx)
    acc = 0.0
    for t in range(1, tau + 1):
        d_t = times_ms[min(g[t - 1], len(times_ms)) - 1]
        acc += d_t - (t - 1) * total / ly
    return acc / tau


def compute_ap(g, lx, ly):
    """Average Proportion: (1/(|x||y|)) * sum_t g(t)."""
    return sum(g) / (lx * ly)


def premature_rate_pct(g, required):
    """% of target words emitted before their aligned source word had been read."""
    return 100.0 * sum(1 for gt, r in zip(g, required) if gt < r) / len(g)


def pareto_frontier(points):
    """points: {name: (latency, quality)}. Minimise latency, maximise quality.
    Returns non-dominated names sorted by latency."""
    keep = []
    for n, (lat, q) in points.items():
        dominated = any(
            (l2 <= lat and q2 >= q) and (l2 < lat or q2 > q)
            for m, (l2, q2) in points.items() if m != n
        )
        if not dominated:
            keep.append(n)
    return sorted(keep, key=lambda n: points[n][0])


# ---------------------------------------------------------------------------
# 3. Schedule generation
# ---------------------------------------------------------------------------
def fixed_wait_k_schedule(k, lx, ly):
    """g(t) = min(|x|, k + t - 1)."""
    return [min(lx, k + t - 1) for t in range(1, ly + 1)]


def derive_adaptive_schedule(required, warmup, lx):
    """Alignment-gated policy: emit target word t as soon as every source word it depends on
    has been read (and never read less than before). `warmup` = words read before the first emission.
    g(t) = max(g(t-1), required(t))."""
    g, cur = [], warmup
    for r in required:
        cur = max(cur, r)
        g.append(min(cur, lx))
    return g


# ---------------------------------------------------------------------------
# 4. Simulation
# ---------------------------------------------------------------------------
def evaluate(label, schedule, target, required, verb_idx):
    al_tok = compute_al(schedule, LX, LY)
    return {
        "policy": label,
        "target": target,
        "schedule": schedule,
        "average_lagging_tokens": round(al_tok, 3),
        "average_lagging_ms": round(compute_al_ms(schedule, SOURCE_TIMESTAMPS_MS, LX, LY), 1),
        "average_proportion": round(compute_ap(schedule, LX, LY), 3),
        "premature_emission_rate_pct": round(premature_rate_pct(schedule, required), 1),
        "verb_premature": bool(schedule[verb_idx] < required[verb_idx]),
    }


def build_results(bleu_table, bleu_source):
    results = {}
    for k in WAIT_K_VALUES:
        sch = fixed_wait_k_schedule(k, LX, LY)
        results[f"Wait_{k}"] = evaluate(f"Fixed Wait-{k}", sch, SVO_TARGET, SVO_REQUIRED, SVO_VERB_IDX)

    ss = derive_adaptive_schedule(SVO_REQUIRED, warmup=2, lx=LX)
    results["StreamSpeech_Adaptive"] = evaluate(
        "StreamSpeech-style Adaptive", ss, SVO_TARGET, SVO_REQUIRED, SVO_VERB_IDX)

    ms = derive_adaptive_schedule(META_REQUIRED, warmup=3, lx=LX)
    results["Meta_Seamless"] = evaluate(
        "Meta-Seamless-style Topicalization", ms, META_TARGET, META_REQUIRED, META_VERB_IDX)

    # Sanity: time-based AL must equal token AL * per-word duration for uniform timestamps.
    per_word = SOURCE_TIMESTAMPS_MS[0]
    if all(SOURCE_TIMESTAMPS_MS[i] == per_word * (i + 1) for i in range(LX)):
        for key, r in results.items():
            assert abs(r["average_lagging_tokens"] * per_word - r["average_lagging_ms"]) < 1.0, key

    for key, r in results.items():
        r["bleu"] = bleu_table[key]
        r["bleu_source"] = bleu_source
    return results


def write_schedule_csv(results, path):
    """Long format: each policy keeps its own target order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "step", "target_word", "source_words_read", "time_of_last_read_ms"])
        for key, r in results.items():
            for t, (word, g) in enumerate(zip(r["target"], r["schedule"]), 1):
                w.writerow([key, t, word, g, SOURCE_TIMESTAMPS_MS[g - 1]])


def print_table(results):
    print("\n" + "=" * 98)
    print(f"{'POLICY':<36}{'AL(tok)':>8}{'AL(ms)':>9}{'AP':>7}{'PREMATURE%':>12}{'VERB OK':>9}{'BLEU*':>8}")
    print("=" * 98)
    for r in results.values():
        print(f"{r['policy']:<36}{r['average_lagging_tokens']:>8.3f}{r['average_lagging_ms']:>9.1f}"
              f"{r['average_proportion']:>7.3f}{r['premature_emission_rate_pct']:>12.1f}"
              f"{('no' if r['verb_premature'] else 'yes'):>9}{r['bleu']:>8.1f}")
    print("=" * 98)


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
def generate_plot(results, frontier, bleu_is_illustrative, plot_path):
    blue, orange, red, green = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"
    wait_keys = [f"Wait_{k}" for k in WAIT_K_VALUES]
    xs = [results[k]["average_lagging_ms"] for k in wait_keys]
    bleu = [results[k]["bleu"] for k in wait_keys]
    prem = [results[k]["premature_emission_rate_pct"] for k in wait_keys]

    fig, ax1 = plt.subplots(figsize=(10, 5.8), dpi=300)
    ax2 = ax1.twinx()

    # Fixed wait-k: filled = verb emitted after its evidence; hollow = verb emitted prematurely.
    ax1.plot(xs, bleu, color=blue, linestyle="--", linewidth=2, zorder=2)
    for k, x, y in zip(WAIT_K_VALUES, xs, bleu):
        safe = not results[f"Wait_{k}"]["verb_premature"]
        ax1.scatter(x, y, s=70, zorder=3, facecolors=blue if safe else "white",
                    edgecolors=blue, linewidths=2)
        ax1.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(8, -13),
                     fontsize=9, fontweight="bold", color=blue)

    ax2.plot(xs, prem, color=orange, marker="x", linestyle="-.", linewidth=1.8, markersize=7, zorder=1)

    ss, ms = results["StreamSpeech_Adaptive"], results["Meta_Seamless"]
    ax1.scatter(ss["average_lagging_ms"], ss["bleu"], marker="*", s=300, color=red, zorder=4)
    ax1.annotate(f"StreamSpeech-style\n(AL {ss['average_lagging_ms']:.0f} ms)",
                 (ss["average_lagging_ms"], ss["bleu"]), xytext=(-20, 28), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec=red, lw=1.2),
                 fontsize=9.5, fontweight="bold", color="#a01010",
                 arrowprops=dict(arrowstyle="->", color=red, lw=1.4))
    ax1.scatter(ms["average_lagging_ms"], ms["bleu"], marker="s", s=100, color=green, zorder=4)
    ax1.annotate(f"Meta-Seamless-style\n(AL {ms['average_lagging_ms']:.0f} ms)",
                 (ms["average_lagging_ms"], ms["bleu"]), xytext=(-12, 6), textcoords="offset points",
                 ha="right", fontsize=9, fontweight="bold", color="#1b6e1b")

    # Pareto frontier over all policies (minimise AL, maximise BLEU).
    fx = [results[n]["average_lagging_ms"] for n in frontier]
    fy = [results[n]["bleu"] for n in frontier]
    ax1.step(fx, fy, where="post", color="gray", linestyle=":", linewidth=1.6, zorder=1)

    ax1.set_xlabel("Average Lagging (ms, time-based AL)", fontsize=12, fontweight="bold", labelpad=10)
    bleu_label = "BLEU (illustrative placeholder)" if bleu_is_illustrative else "BLEU"
    ax1.set_ylabel(bleu_label, fontsize=12, fontweight="bold", color=blue, labelpad=10)
    ax2.set_ylabel("Premature-emission rate (%)", fontsize=12, fontweight="bold", color=orange, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=blue)
    ax2.tick_params(axis="y", labelcolor=orange)

    all_x = [r["average_lagging_ms"] for r in results.values()]
    all_y = [r["bleu"] for r in results.values()]
    pad = 0.06 * (max(all_x) - min(all_x))
    ax1.set_xlim(min(all_x) - pad, max(all_x) + 2.5 * pad)
    ax1.set_ylim(min(all_y) - 2, max(all_y) + 3)
    ax2.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.45)

    handles = [
        Line2D([], [], color=blue, marker="o", linestyle="--", lw=2, label="Fixed Wait-k (filled: verb after evidence)"),
        Line2D([], [], color=blue, marker="o", markerfacecolor="white", linestyle="None", label="Wait-k, verb emitted prematurely"),
        Line2D([], [], color=orange, marker="x", linestyle="-.", label="Premature-emission rate (computed)"),
        Line2D([], [], color=red, marker="*", markersize=14, linestyle="None", label="StreamSpeech-style adaptive"),
        Line2D([], [], color=green, marker="s", linestyle="None", label="Meta-Seamless-style topicalization"),
        Line2D([], [], color="gray", linestyle=":", label="Pareto frontier (AL vs. BLEU)"),
    ]
    ax1.legend(handles=handles, loc="lower right", framealpha=0.92, fontsize=8.5)

    plt.title("Simultaneous S2ST, Hindi (SOV) → English: Latency vs. Quality Trade-off",
              fontsize=13, fontweight="bold", pad=12)
    note = ("AL, AP and premature-emission rate are computed from the read/write schedules. "
            + ("BLEU values are illustrative placeholders, not measured results."
               if bleu_is_illustrative else "BLEU values were supplied from measured runs."))
    fig.text(0.01, 0.008, note, fontsize=8, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    # Prefer code/results if run from repo root or scripts
    default_out = script_dir.parent / "results" if (script_dir.parent / "results").exists() else script_dir / "results"
    
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--quality", type=Path, default=None,
                    help="JSON {policy_key: bleu} with measured BLEU; overrides the placeholders")
    args = ap.parse_args()

    bleu_table, bleu_source = dict(ILLUSTRATIVE_BLEU), "illustrative_placeholder"
    if args.quality:
        measured = json.loads(args.quality.read_text(encoding="utf-8"))
        missing = set(bleu_table) - set(measured)
        if missing:
            raise SystemExit(f"--quality is missing policies: {sorted(missing)}")
        bleu_table, bleu_source = {k: float(measured[k]) for k in bleu_table}, "measured"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "read_write_matrix.csv"
    json_path = args.out_dir / "phase4_latency_metrics.json"
    png_path = args.out_dir / "latency_vs_accuracy.png"

    print("=" * 98)
    print("PHASE 4: STREAMING LATENCY & WAIT-K SIMULATION (PART E - ML TRACK)")
    print("=" * 98)
    print(f"Source (SOV): {' '.join(SOURCE_TOKENS)}  (|x|={LX})")
    print(f"Target (SVO): {' '.join(SVO_TARGET)}  (|y|={LY}, gamma={LY / LX:.2f})")

    results = build_results(bleu_table, bleu_source)

    frontier = pareto_frontier({k: (r["average_lagging_ms"], r["bleu"]) for k, r in results.items()})
    for k, r in results.items():
        r["on_pareto_frontier"] = k in frontier

    write_schedule_csv(results, csv_path)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(results)
    print(f"Pareto frontier (AL vs BLEU): {', '.join(frontier)}")
    if bleu_source != "measured":
        print("* BLEU values are illustrative placeholders; pass --quality to use measured numbers.")

    generate_plot(results, frontier, bleu_source != "measured", png_path)
    print(f"\n✓ {csv_path}\n✓ {json_path}\n✓ {png_path}")


if __name__ == "__main__":
    main()
