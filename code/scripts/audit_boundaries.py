"""
Boundary audit, two steps (no simulated data):

  python audit_boundaries.py sample   -> writes data/boundary_audit.csv with 30 sampled boundaries
  (measure each boundary by hand/spectrogram under ONE definition, fill 'human_boundary_sec';
   optionally spot-check some rows independently in Audacity -> 'hand_check_sec')
  python audit_boundaries.py report   -> error statistics, CI, bias, resolution and validation checks

Boundary definition used throughout: the END of w1's acoustic energy.
"""
import os
import sys
import csv
import json
import random
import numpy as np

N_BOUNDARIES = 30
SEED = 42
MIN_WORD_SCORE = 0.0        # raise (e.g. 0.3) to sample only confident words
CTC_FRAME_MS = 20.0         # MMS_FA / wav2vec2 frame stride
STREAMING_FRAME_MS = 80.0   # your streaming frame size, for the comparison line
OUTLIER_MS = 40.0           # rows above this are listed individually
BOOTSTRAP_N = 10000
FIELDS = ["sample_id", "clip_id", "clip_file", "word_idx", "w1", "w2",
          "aligner_boundary_sec", "human_boundary_sec", "hand_check_sec", "notes"]


def paths():
    base_dir = "code" if os.path.exists("code") else "."
    data_dir = os.path.join(base_dir, "data")
    return os.path.join(data_dir, "aligned"), os.path.join(data_dir, "boundary_audit.csv")


def build_sample():
    aligned_dir, csv_path = paths()
    if not os.path.exists(aligned_dir):
        print(f"Error: {aligned_dir} not found. Run the alignment script first.")
        return
    files = sorted(f for f in os.listdir(aligned_dir) if f.endswith("_aligned.json"))
    candidates = []
    for f in files:
        with open(os.path.join(aligned_dir, f), encoding="utf-8") as jf:
            d = json.load(jf)
        if d.get("alignment_method") != "mms_ctc_trellis":
            continue  # never audit uniform-fallback clips
        words = d["words"]
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a["aligned"] and b["aligned"] and min(a["score"], b["score"]) >= MIN_WORD_SCORE:
                candidates.append({"clip_id": d["id"], "clip_file": d["clip_file"], "word_idx": i,
                                   "w1": a["w"], "w2": b["w"], "aligner_boundary_sec": a["end"]})
    if len(candidates) < N_BOUNDARIES:
        print(f"Only {len(candidates)} eligible boundaries; need {N_BOUNDARIES}.")
        return
    rng = random.Random(SEED)
    picked = sorted(rng.sample(candidates, N_BOUNDARIES),
                    key=lambda c: (c["clip_id"], c["word_idx"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=FIELDS)
        w.writeheader()
        for n, c in enumerate(picked, 1):
            w.writerow({"sample_id": n, **c, "human_boundary_sec": "", "hand_check_sec": "", "notes": ""})
    print(f"Wrote {N_BOUNDARIES} sampled boundaries -> {csv_path}")
    print("Fill in human_boundary_sec, then run: python audit_boundaries.py report")


def _num(x):
    x = (x or "").strip()
    return float(x) if x else None


def report():
    _, csv_path = paths()
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Run 'sample' first.")
        return None
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    done = [r for r in rows if _num(r.get("human_boundary_sec")) is not None]
    if not done:
        print("No human measurements filled in yet.")
        return None
    if len(done) < len(rows):
        print(f"[WARN] Only {len(done)}/{len(rows)} rows measured; report covers those only.\n")

    al = np.array([float(r["aligner_boundary_sec"]) for r in done])
    hu = np.array([float(r["human_boundary_sec"]) for r in done])
    signed = (al - hu) * 1000.0          # negative = aligner EARLIER than measured boundary
    absolute = np.abs(signed)
    n = len(done)

    # Bootstrap 95% CI on the mean absolute error (deterministic)
    rng = np.random.default_rng(SEED)
    boots = rng.choice(absolute, size=(BOOTSTRAP_N, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    bias_boots = rng.choice(signed, size=(BOOTSTRAP_N, n), replace=True).mean(axis=1)
    b_lo, b_hi = np.percentile(bias_boots, [2.5, 97.5])

    print("=== WORD BOUNDARY AUDIT ===")
    print("Definition: boundary = end of w1's acoustic energy. Positive error = aligner LATE.\n")
    for r, s in zip(done[:10], signed[:10]):
        print(f"[{int(r['sample_id']):02d}] '{r['w1']}' -> '{r['w2']}' ({r['clip_id']}): "
              f"Aligner {float(r['aligner_boundary_sec']):.3f}s | Measured {float(r['human_boundary_sec']):.3f}s | "
              f"Signed err {s:+.1f} ms")

    print("\n========================================================")
    print(f"Measured boundaries: {n}")
    print(f"Mean absolute error: {absolute.mean():.1f} ms (std {absolute.std(ddof=1):.1f}); "
          f"95% bootstrap CI [{ci_lo:.1f}, {ci_hi:.1f}] ms")
    print(f"Median / P90 / Max:  {np.median(absolute):.1f} / {np.percentile(absolute, 90):.1f} / {absolute.max():.1f} ms")
    print(f"Error in CTC frames: mean {absolute.mean() / CTC_FRAME_MS:.2f} frames ({CTC_FRAME_MS:.0f} ms each)")

    n_early, n_late = int((signed < 0).sum()), int((signed > 0).sum())
    direction = "early" if signed.mean() < 0 else "late"
    print(f"Signed bias:         {signed.mean():+.1f} ms, 95% CI [{b_lo:+.1f}, {b_hi:+.1f}] "
          f"(aligner ends words {direction} on average)")
    print(f"Direction counts:    aligner early in {n_early}/{n}, late in {n_late}/{n}")
    if b_lo > 0 or b_hi < 0:
        print("  -> bias CI excludes zero: systematic offset, consistent with CTC span-end behavior.")
    else:
        print("  -> bias CI includes zero: no clear systematic offset at this sample size.")

    print(f"Within 20 / 40 / {STREAMING_FRAME_MS:.0f} ms: "
          f"{np.mean(absolute <= 20) * 100:.0f}% / {np.mean(absolute <= 40) * 100:.0f}% / "
          f"{np.mean(absolute <= STREAMING_FRAME_MS) * 100:.0f}%")
    print(f"Mean error is {STREAMING_FRAME_MS / max(absolute.mean(), 1e-9):.1f}x smaller than the "
          f"{STREAMING_FRAME_MS:.0f} ms streaming frame.")

    # Outliers
    out = [(int(r["sample_id"]), r["w1"], r["w2"], s, r.get("notes", ""))
           for r, s in zip(done, signed) if abs(s) > OUTLIER_MS]
    if out:
        print(f"\nRows with |error| > {OUTLIER_MS:.0f} ms ({len(out)}):")
        for sid, w1, w2, s, note in out:
            print(f"  #{sid:02d} '{w1}' -> '{w2}': {s:+.0f} ms  {note[:70]}")

    # Measurement-resolution check
    ms = np.round(hu * 1000).astype(int)
    on5 = float(np.mean(ms % 5 == 0))
    print(f"\nMeasurement resolution: {on5 * 100:.0f}% of measured times fall on a 5 ms grid "
          f"(~20% expected if resolution were 1 ms).")
    if on5 > 0.4:
        print("  -> Effective reading resolution is ~5 ms. State this; do NOT claim single-ms precision.")

    # Optional independent spot-check
    chk = [(r, _num(r.get("hand_check_sec"))) for r in done if _num(r.get("hand_check_sec")) is not None]
    if chk:
        diffs = np.array([(float(r["human_boundary_sec"]) - c) * 1000 for r, c in chk])
        print(f"\nIndependent hand-check (Audacity) on {len(chk)} rows vs recorded values:")
        print(f"  mean |diff| {np.abs(diffs).mean():.1f} ms, max {np.abs(diffs).max():.1f} ms, "
              f"within 10 ms: {np.mean(np.abs(diffs) <= 10) * 100:.0f}%")
        if np.abs(diffs).max() > 20:
            print("  -> [WARN] some rows disagree by >20 ms; revisit those before reporting.")
    else:
        print("\n[NOTE] No 'hand_check_sec' values found. Spot-check ~8-10 rows independently "
              "(largest errors + a few tight ones) and fill that column to validate.")

    print("\nState in the report: who/what placed the boundaries, the definition above, the ~resolution, n, and CI.")
    print("========================================================")
    return absolute.mean(), absolute.std(ddof=1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "sample":
        build_sample()
    elif mode == "report":
        report()
    else:
        print("Usage: python audit_boundaries.py [sample|report]")