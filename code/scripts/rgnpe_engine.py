"""
RGNPE engine (Phase 3): entity locking, target assembly and verification.

DOES
  1. Entity locking: reads amount / phone / account / cheque entities from the manifest, parses them
     with rgnpe_extractors, and confirms each one is really present in the Hindi source (spoken digit
     runs are re-parsed and must equal the manifest digits; amount phrases must occur verbatim).
  2. Placeholder masking: writes a placeholder-constrained Hindi source (`rgnpe.hi_masked`) using HTML
     no-translate spans, plus a boundary-safe, exactly-once slot restorer for a future MT call.
  3. Target assembly: en_text / en_alt come from authored targets (sidecar JSON). If a clip has no
     authored en_text, it falls back to rule-based retraction of the manifest's en_fluent.
  4. Verification (the actual gate): committed entities appear in the English targets, retracted
     (abandoned) entities do not, and negation polarity is preserved. Exit code 1 on any failure.

DOES NOT
  - Call an NMT / speech model, derive chunk boundaries from CTC timestamps, or implement
    SeamlessStreaming or Moshi. Authored targets are labelled as such in `rgnpe.provenance`
    and in `translation_method`.

Usage:  python rgnpe_engine.py [--input M] [--output M2] [--targets T] [--selftest]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rgnpe_extractors import (  # noqa: E402
    HINDI_DIGITS_MAP,
    check_polarity_with_breaker_masking,
    format_dual_targets,
    parse_relative_scale_amount,
    parse_spoken_digits,
)

DATA_DIR = HERE.parent / "data"
CORPUS = "sarvam_dev_audited"
REQUIRED_KEYS = ("id", "clip_file", "speaker", "duration", "hi_verbatim", "en_fluent")
DIGIT_FIELDS = (("phone_digits", "PHONE"), ("account_digits", "ACCOUNT"), ("cheque_digits", "CHEQUE"))
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
MULTIPLIERS = {"डबल", "ट्रिपल"}
MIN_DIGIT_RUN = 6  # a spoken-digit run shorter than this is not treated as an identifier
FAR = 10**9


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def norm_en(s: str) -> str:
    """Lower-case and treat hyphens/dashes as spaces so 'thirty-five lakh' == 'thirty five lakh'."""
    return re.sub(r"\s+", " ", re.sub(r"[-\u2010-\u2015]", " ", s.lower())).strip()


def pair_variants(items: List[str]) -> List[Tuple[str, List[str]]]:
    """['पैंतीस लाख', 'thirty-five lakh'] -> [('पैंतीस लाख', ['thirty-five lakh'])]"""
    pairs: List[Tuple[str, List[str]]] = []
    for it in items:
        if DEVANAGARI.search(it):
            pairs.append((it, []))
        elif pairs:
            pairs[-1][1].append(it)
    return pairs


def format_digits(kind: str, d: str) -> str:
    if kind == "PHONE" and len(d) == 10:
        return f"{d[:5]} {d[5:]}"
    if kind == "ACCOUNT":
        return " ".join(d[i:i + 4] for i in range(0, len(d), 4))
    return d


def amount_variants(value: int | None, en_variants: List[str]) -> List[str]:
    """English surface forms to look for, including numeric ones ('35 lakh') so leaks are caught."""
    v = {norm_en(x) for x in en_variants}
    if value:
        if value >= 10_000_000:
            v.add(f"{value / 1e7:g} crore")
        elif value >= 100_000:
            v.add(f"{value / 1e5:g} lakh")
    return sorted(v)


def find_digit_runs(hi: str) -> List[Tuple[int, int]]:
    """Character spans in `hi` of >= MIN_DIGIT_RUN consecutive spoken-digit tokens."""
    runs: List[Tuple[int, int]] = []
    cur: List[Tuple[int, int, str]] = []

    def flush() -> None:
        if sum(1 for _, _, t in cur if t not in MULTIPLIERS) >= MIN_DIGIT_RUN:
            runs.append((cur[0][0], cur[-1][1]))
        cur.clear()

    for m in re.finditer(r"\S+", hi):
        tok = m.group().rstrip(".,।!?…")
        if tok in HINDI_DIGITS_MAP or tok in MULTIPLIERS:
            cur.append((m.start(), m.start() + len(tok), tok))
        else:
            flush()
    flush()
    return runs


# ---------------------------------------------------------------------------
# 1. Entity locking
# ---------------------------------------------------------------------------
def build_entities(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    hi = rec["hi_verbatim"]
    ents_in = rec.get("entities", {})
    found: List[Dict[str, Any]] = []

    runs = [(s, e, re.sub(r"\D", "", parse_spoken_digits(hi[s:e]))) for s, e in find_digit_runs(hi)]
    for field, kind in DIGIT_FIELDS:
        for raw in ents_in.get(field, []):
            if not re.fullmatch(r"\+?[\d ]+", raw.strip()):
                continue  # spoken-English variant, not the canonical digit string
            digits = re.sub(r"\D", "", raw)
            run = next(((s, e) for s, e, d in runs if d == digits), None)
            found.append({
                "kind": kind.lower(), "status": "committed", "digits": digits,
                "display": format_digits(kind, digits),
                "hi_text": hi[run[0]:run[1]] if run else None,
                "in_source": run is not None,
                "_start": run[0] if run else FAR,
                "_patterns": [r"(?<!\d)" + r"[ \-]?".join(digits) + r"(?!\d)"],
            })

    for key, status in (("amounts", "committed"), ("abandoned_amounts", "retracted")):
        for hi_phrase, en_vars in pair_variants(ents_in.get(key, [])):
            parsed = parse_relative_scale_amount(hi_phrase)  # per phrase: the extractor drops amounts if given a whole clip
            value = int(parsed[0][0]) if len(parsed) == 1 else None
            variants = amount_variants(value, en_vars)
            start = hi.find(hi_phrase)
            found.append({
                "kind": "amount", "status": status, "hi_text": hi_phrase, "value_inr": value,
                "display": en_vars[0] if en_vars else (variants[0] if variants else hi_phrase),
                "in_source": start >= 0,
                "_start": start if start >= 0 else FAR,
                "_patterns": [rf"(?<![\w.]){re.escape(p)}(?!\w)" for p in variants],
            })

    found.sort(key=lambda e: e["_start"])
    for i, e in enumerate(found):
        e["slot_id"] = f"SLOT_{e['kind'].upper()}_{i}"
    return found


def entity_hit(e: Dict[str, Any], text: str) -> bool:
    hay = norm_en(text) if e["kind"] == "amount" else text
    return any(re.search(p, hay) for p in e["_patterns"])


# ---------------------------------------------------------------------------
# 2. Placeholder masking / restoration
# ---------------------------------------------------------------------------
class SlotError(ValueError):
    """A protected slot was dropped, duplicated or left unresolved after translation."""


class SlotMasker:
    TAG = '<span translate="no" class="notranslate" id="{sid}">{sid}</span>'

    @classmethod
    def mask(cls, text: str, entities: List[Dict[str, Any]]) -> str:
        for e in sorted((x for x in entities if x["in_source"]), key=lambda x: x["_start"]):
            text = text.replace(e["hi_text"], cls.TAG.format(sid=e["slot_id"]), 1)
        return text


class SlotRestorer:
    @staticmethod
    def restore(translated: str, slots: Dict[str, str]) -> str:
        """Replace each slot exactly once (span or bare token); raise SlotError otherwise."""
        out = translated
        for sid, value in slots.items():
            span_re = re.compile(rf'<span\b[^>]*\bid=["\']{re.escape(sid)}["\'][^>]*>.*?</span>', re.I | re.S)
            plain_re = re.compile(rf"(?<!\w){re.escape(sid)}(?!\w)")
            n_span = len(span_re.findall(out))
            if n_span == 1:
                out = span_re.sub(lambda _m, v=value: v, out)
            elif n_span == 0 and len(plain_re.findall(out)) == 1:
                out = plain_re.sub(lambda _m, v=value: v, out)
            else:
                raise SlotError(f"{sid}: expected exactly one occurrence after translation")
        out = re.sub(r"\s+", " ", re.sub(r"</?span\b[^>]*>", "", out)).strip()
        if re.search(r"SLOT_[A-Z]+_\d+", out):
            raise SlotError("unresolved slot token left in translated text")
        return out


def selftest() -> None:
    slots = {"SLOT_AMOUNT_1": "one lakh", "SLOT_AMOUNT_10": "ten lakh"}  # _1 must not clobber _10
    mt = 'Pay <span translate="no" id="SLOT_AMOUNT_10">SLOT_AMOUNT_10</span> and SLOT_AMOUNT_1 today.'
    got = SlotRestorer.restore(mt, slots)
    assert got == "Pay ten lakh and one lakh today.", got
    for bad in ('Pay <span id="SLOT_AMOUNT_10">SLOT_AMOUNT_10</span> today.',      # SLOT_AMOUNT_1 dropped
                "SLOT_AMOUNT_1 and SLOT_AMOUNT_1 and SLOT_AMOUNT_10"):             # SLOT_AMOUNT_1 duplicated
        try:
            SlotRestorer.restore(bad, slots)
        except SlotError:
            continue
        raise AssertionError(f"expected SlotError for: {bad!r}")
    print("selftest passed")


# ---------------------------------------------------------------------------
# 3. Target assembly
# ---------------------------------------------------------------------------
def assemble_targets(rec: Dict[str, Any], authored: Dict[str, Any] | None) -> Tuple[str, str, str, str]:
    a = authored or {}
    if a.get("en_text"):
        en_text, p_text = a["en_text"], "authored"
    else:
        en_text, _ = format_dual_targets(rec["en_fluent"], rec.get("breakers", []))
        p_text = "rule_derived"
    chunks = a.get("en_alt_chunks")
    en_alt, p_alt = (" ".join(chunks), "authored") if chunks else ("", "missing")
    return en_text, en_alt, p_text, p_alt


# ---------------------------------------------------------------------------
# 4. Verification
# ---------------------------------------------------------------------------
def breaker_masks(rec: Dict[str, Any]) -> List[str]:
    """Repair markers, plus the restated-negation fragment ('पच्चीस नहीं') that follows some repairs."""
    masks: List[str] = []
    for b in rec.get("breakers", []):
        if b.get("marker"):
            masks.append(b["marker"])
        if b.get("type") == "self_repair" and b.get("abandoned"):
            masks.append(f'{b["abandoned"].split()[0]} नहीं')
    return masks


def verify(rec, entities, en_text, en_alt, strict_alt) -> Dict[str, Any]:
    hi, masks, problems = rec["hi_verbatim"], breaker_masks(rec), []
    for e in entities:
        if not e["in_source"]:
            problems.append(f"{e['slot_id']}: {e['hi_text'] or e['display']!r} not verified in Hindi source")
        if e["kind"] == "amount" and e["value_inr"] is None:
            problems.append(f"{e['slot_id']}: amount {e['hi_text']!r} did not parse to a single value")

    polarity: Dict[str, bool] = {}
    targets = [("en_text", en_text, True)] + ([("en_alt", en_alt, strict_alt)] if en_alt else [])
    for name, text, enforce_retraction in targets:
        for e in entities:
            hit = entity_hit(e, text)
            if e["status"] == "committed" and not hit:
                problems.append(f"{name}: committed {e['display']!r} not found")
            elif e["status"] == "retracted" and hit and enforce_retraction:
                problems.append(f"{name}: retracted {e['display']!r} still present")
        hi_neg, en_neg, status = check_polarity_with_breaker_masking(hi, text, masks)
        polarity["source_negative"] = hi_neg
        polarity[f"{name}_negative"] = en_neg
        if status != "PRESERVED":
            problems.append(f"{name}: {status}")
    return {"ok": not problems, "problems": problems, "polarity": polarity}


# ---------------------------------------------------------------------------
# 5. Record assembly
# ---------------------------------------------------------------------------
def process_record(rec: Dict[str, Any], authored: Dict[str, Any] | None, strict_alt: bool) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise ValueError(f"{rec.get('id', '<no id>')}: manifest record missing keys {missing}")
    cid, hi = rec["id"], rec["hi_verbatim"]

    entities = build_entities(rec)
    en_text, en_alt, p_text, p_alt = assemble_targets(rec, authored)
    verification = verify(rec, entities, en_text, en_alt, strict_alt)
    public = [{k: v for k, v in e.items() if not k.startswith("_")} for e in entities]

    return {
        "id": cid,
        "audio": f"audio/{cid}{Path(rec['clip_file']).suffix or '.wav'}",
        "duration": rec["duration"],
        "speaker_id": rec["speaker"],
        "source": {
            "corpus": CORPUS,
            "source_file": rec["clip_file"],
            "start": 0.0,
            "end": rec["duration"],
            "synthetic": rec.get("synthetic", "eleven" in rec["speaker"]),
        },
        "sentences": [{
            "hi_text": hi, "hi_start": 0.0, "hi_end": rec["duration"],
            "en_text": en_text,
            "en_alt": [en_alt] if en_alt else [],
        }],
        "entities_locked": {e["slot_id"]: e["display"] for e in public if e["status"] == "committed" and e["in_source"]},
        "entities_retracted": {e["slot_id"]: e["display"] for e in public if e["status"] == "retracted"},
        "translation_method": f"en_text:{p_text}|en_alt:{p_alt}",
        "rgnpe": {
            "entities": public,
            "hi_masked": SlotMasker.mask(hi, entities),
            "provenance": {"en_text": p_text, "en_alt": p_alt},
            "verification": verification,
        },
    }


def clip(s: str, n: int = 90) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DATA_DIR / "manifest.json")
    ap.add_argument("--output", type=Path, default=DATA_DIR / "manifest_rgnpe.json")
    ap.add_argument("--targets", type=Path, default=DATA_DIR / "rgnpe_authored_targets.json")
    ap.add_argument("--allow-retraction-in-alt", action="store_true",
                    help="do not fail when a retracted value appears in en_alt (en_text is always strict)")
    ap.add_argument("--selftest", action="store_true", help="run the slot masker/restorer self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output must differ from --input (the original manifest is never overwritten)")
    if not args.input.exists():
        raise SystemExit(f"Error: {args.input} not found.")

    records = json.loads(args.input.read_text(encoding="utf-8"))
    authored_all: Dict[str, Any] = {}
    if args.targets.exists():
        raw = json.loads(args.targets.read_text(encoding="utf-8"))
        authored_all = {k: v for k, v in raw.items() if not k.startswith("_")}
        unknown = sorted(set(authored_all) - {r.get("id") for r in records})
        if unknown:
            print(f"WARNING: authored targets for ids not in the manifest: {unknown}")
    else:
        print(f"WARNING: {args.targets} not found; every clip falls back to rule-derived en_text and has no en_alt.")

    print("=" * 88)
    print("PHASE 3: RGNPE ENGINE (entity locking + authored targets + verification)")
    print("=" * 88)

    out_records, failed, no_alt = [], [], []
    for rec in records:
        out = process_record(rec, authored_all.get(rec.get("id")), not args.allow_retraction_in_alt)
        out_records.append(out)
        v, s = out["rgnpe"]["verification"], out["sentences"][0]
        prov = out["rgnpe"]["provenance"]
        if not v["ok"]:
            failed.append(out["id"])
        if prov["en_alt"] == "missing":
            no_alt.append(out["id"])
        print(f"\n[{out['id']}] ({out['source']['source_file']})  {'✓' if v['ok'] else '✗'}")
        print(f"   • locked {list(out['entities_locked'].values())}  retracted {list(out['entities_retracted'].values())}")
        print(f"   • en_text [{prov['en_text']}]: \"{clip(s['en_text'])}\"")
        print(f"   • en_alt  [{prov['en_alt']}]: \"{clip(s['en_alt'][0]) if s['en_alt'] else ''}\"")
        print(f"   • polarity: {v['polarity']}")
        for p in v["problems"]:
            print(f"   ✗ {p}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_records, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(out_records)
    passed = total - len(failed)
    print("\n" + "=" * 88)
    print(f"Wrote {args.output}")
    print(f"Verification: {passed}/{total} clips passed ({100.0 * passed / max(total, 1):.1f}%)")
    if failed:
        print(f"FAILED: {failed}")
    if no_alt:
        print(f"WARNING: no authored en_alt for {no_alt}")
    print("=" * 88)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
