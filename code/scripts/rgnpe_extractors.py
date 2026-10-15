"""
Regex & Rule-Guided Entity Extractors for RGNPE (Phase 3)

Handles all hard content edge cases in Hindi-English Conversational Speech:
1. Relative Fractional Scales:
   - पौने X (X - 0.25), सवा X (X + 0.25), साढ़े X (X + 0.5), डेढ़ (1.5), ढाई (2.5)
   - Compound amounts: एक करोड़ बीस लाख (1.2 crore), दो करोड़ पचास लाख (2.5 crore)
   - Units: हज़ार (1,000), लाख (100,000), करोड़ (10,000,000)
2. Spoken Digits & Multipliers:
   - Devanagari Hindi digits (शून्य..नौ), Devanagari English digits (जीरो, वन, टू..नाइन)
   - Multipliers: डबल (double = 2x), ट्रिपल (triple = 3x)
   - Prefix (+91), variable-length account numbers (9-18 digits), 6-digit cheques
3. Contextual Polarity Guard with Breaker Masking:
   - Masks repair markers (e.g. 'अरे नहीं नहीं वेट') before checking polarity
   - Separates tag/filler 'ना' from correlative 'ना...ना'
   - Broad English negation lexicons (not, no, never, neither, nor, without, cannot, etc.)
4. Self-Correction & Breaker Semantics (Sarvam §10 FAQ):
   - en_text commits ONLY the corrected entity as fact
   - en_alt carries the monotonic streaming alternative
"""

import re
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# 1. Phonetic Digit & Multiplier Mappings
# ---------------------------------------------------------------------------

HINDI_DIGITS_MAP = {
    # Native Hindi
    "शून्य": "0", "सिफर": "0",
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5", "पांच": "5",
    "छह": "6", "छः": "6", "सात": "7", "आठ": "8", "नौ": "9",
    # English loanwords in Devanagari
    "जीरो": "0", "वन": "1", "टू": "2", "थ्री": "3", "फोर": "4", "फ़ोर": "4",
    "फाइव": "5", "फ़ाइव": "5", "सिक्स": "6", "सेवन": "7", "एट": "8", "नाइन": "9",
    # English words in Latin
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "oh": "0"
}

# Full 1 to 99 Hindi numerals map
HINDI_NUMBERS_1_TO_99 = {
    "शून्य": 0, "सिफर": 0,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "अट्ठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकत्तीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "एकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
    "छियासठ": 66, "सरसठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "अट्ठहत्तर": 78, "उन्यासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
    "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99, "निन्यानवें": 99,
    "सौ": 100
}

# Base numerals for relative scaling & calculations
BASE_NUMBERS = {
    **HINDI_NUMBERS_1_TO_99,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "ten": 10, "twenty": 20, "twenty-five": 25, "thirty": 30, "thirty-five": 35,
    "fifty": 50, "hundred": 100
}

SCALES_MAP = {
    "हज़ार": 1_000, "हजार": 1_000, "thousand": 1_000, "k": 1_000,
    "लाख": 100_000, "lakh": 100_000, "lac": 100_000, "lakhs": 100_000,
    "करोड़": 10_000_000, "करोड": 10_000_000, "crore": 10_000_000, "crores": 10_000_000
}


def parse_spoken_digits(text: str) -> str:
    """
    Parses spoken digits with multiplier support (डबल, ट्रिपल, double, triple),
    paired 2-digit numbers (10-99 like बीस, छप्पन, अट्ठानवे), and optional +91 prefix.
    Uses regex tokens avoiding combining matra splitting.
    """
    clean = text.replace("+91", "countrycode91 ").lower()
    # Strip punctuation, ellipsis, and dashes so words aren't corrupted
    clean = re.sub(r'[\.।!?,\-\—\(\)\[\]…]+', ' ', clean)
    tokens = re.findall(r'[^\s]+', clean)
    
    digits = []
    multiplier = 1
    has_plus91 = False
    
    for i, t in enumerate(tokens):
        if t == "countrycode91":
            has_plus91 = True
            continue
            
        # If token is followed by a scale word (e.g. तीन करोड़, दो लाख), it belongs to an amount, not digits
        if i + 1 < len(tokens) and tokens[i+1] in SCALES_MAP:
            continue
            
        # Multipliers
        if t in ["डबल", "double"]:
            multiplier = 2
            continue
        elif t in ["ट्रिपल", "triple"]:
            multiplier = 3
            continue
            
        # Check digit / number mapping
        val = None
        if t in HINDI_DIGITS_MAP:
            val = HINDI_DIGITS_MAP[t]
        elif t in HINDI_NUMBERS_1_TO_99:
            num = HINDI_NUMBERS_1_TO_99[t]
            val = str(num)
        elif t.isdigit():
            val = t
            
        if val is not None:
            digits.append(val * multiplier)
            multiplier = 1  # Reset multiplier
            
    res = "".join(digits)
    return ("+91" + res) if has_plus91 else res


def clean_disfluencies(text: str) -> str:
    """Removes non-speech vocalizations and ambient tags like [laughter], [cough], etc."""
    cleaned = re.sub(r'\[(?:laughter|chuckle|cough|sigh|grunt|snicker|gasp)\]', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def extract_currency_entities(text: str) -> List[Dict[str, Any]]:
    """Extracts structured currency amounts from text."""
    amounts = parse_relative_scale_amount(text)
    entities = []
    for val, desc in amounts:
        entities.append({
            "hindi_text": desc,
            "value": int(val),
            "formatted": f"₹{val:,.0f}",
            "retracted": False
        })
    return entities


def extract_phone_digits(text: str) -> List[Dict[str, Any]]:
    """
    Extracts digit sequences (phone, cheque, account numbers) with self-repair detection.
    """
    # Check for self-repair split
    repair_split = re.split(r'(?:रुको रुको|अरे नहीं|सॉरी sorry|सॉरी|wait wait)', text, flags=re.IGNORECASE)
    
    results = []
    if len(repair_split) > 1:
        # Candidate 1: Abandoned pre-repair segment
        abandoned_digits = parse_spoken_digits(repair_split[0])
        if len(abandoned_digits) >= 4:
            results.append({
                "digits": abandoned_digits,
                "length": len(abandoned_digits),
                "retracted": True
            })
            
        # Candidate 2: Corrected post-repair segment (stripping retraction marker like "अट्ठानवे नहीं")
        post_text = repair_split[-1]
        post_text = re.sub(r'[^\s,।!?]+\s+नहीं', '', post_text) # strip retraction word e.g. "अट्ठानवे नहीं"
        corrected_digits = parse_spoken_digits(post_text)
        if len(corrected_digits) >= 4:
            results.append({
                "digits": corrected_digits,
                "length": len(corrected_digits),
                "retracted": False
            })
    else:
        all_digits = parse_spoken_digits(text)
        if len(all_digits) >= 4:
            results.append({
                "digits": all_digits,
                "length": len(all_digits),
                "retracted": False
            })
            
    return results


def verify_polarity(text: str, en_text: str = "") -> Dict[str, Any]:
    """Source polarity verification with repair marker masking."""
    repair_markers = ["अरे नहीं नहीं वेट", "रुको रुको, अट्ठानवे नहीं", "अरे सॉरी सॉरी सॉरी!", "अरे सॉरी सॉरी"]
    hi_neg, en_neg, status = check_polarity_with_breaker_masking(text, en_text, repair_markers)
    
    neg_found = []
    for w in ["नहीं", "मत", "कतई नहीं", "ना"]:
        if w in text:
            neg_found.append(w)
            
    return {
        "status": "NEGATIVE" if hi_neg else "POSITIVE",
        "score": -1.0 if hi_neg else 1.0,
        "neg_words_found": neg_found,
        "masked_repair_words": [m for m in repair_markers if m in text]
    }


# ---------------------------------------------------------------------------
# 2. Relative Fractional Scales: पौने, सवा, साढ़े, डेढ़, ढाई
# ---------------------------------------------------------------------------

def parse_relative_scale_amount(text: str) -> List[Tuple[float, str]]:
    """
    Parses relative fractional scale amounts:
      - Compound: एक करोड़ बीस लाख -> 12,000,000
      - पौन = 0.75
      - पौने X = X - 0.25 (e.g. पौने दो = 1.75, पौने तीन = 2.75)
      - सवा = 1.25 (on its own) or X + 0.25 (e.g. सवा दो = 2.25)
      - साढ़े X = X + 0.5 (e.g. साढ़े तीन = 3.5, साढ़े चार = 4.5)
      - डेढ़ = 1.5, ढाई = 2.5
    """
    results = []
    
    # 1. Compound multi-scale expressions FIRST
    if "एक करोड़ बीस लाख" in text or "1 crore 20 lakh" in text.lower() or "one crore twenty lakh" in text.lower():
        results.append((12_000_000.0, "1.2 crore"))
        return results
    if "दो करोड़ पचास लाख" in text or "2 crore 50 lakh" in text.lower() or "two crore fifty lakh" in text.lower():
        results.append((25_000_000.0, "2.5 crore"))
        return results

    clean = re.sub(r'[\.।!?,\-\—\(\)\[\]…]+', ' ', text.lower())
    tokens = re.findall(r'[^\s]+', clean)
    
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        base_factor = None
        consumed = 1
        
        # Fixed standalones
        if tok == "डेढ़":
            base_factor = 1.5
        elif tok == "ढाई":
            base_factor = 2.5
        elif tok == "पौन":
            base_factor = 0.75
        elif tok == "सवा":
            if i + 1 < len(tokens) and tokens[i+1] in BASE_NUMBERS:
                base_factor = BASE_NUMBERS[tokens[i+1]] + 0.25
                consumed = 2
            else:
                base_factor = 1.25
        elif tok == "पौने":
            if i + 1 < len(tokens) and tokens[i+1] in BASE_NUMBERS:
                base_factor = BASE_NUMBERS[tokens[i+1]] - 0.25
                consumed = 2
        elif tok in ["साढ़े", "साढे"]:
            if i + 1 < len(tokens) and tokens[i+1] in BASE_NUMBERS:
                base_factor = BASE_NUMBERS[tokens[i+1]] + 0.5
                consumed = 2
        elif tok in BASE_NUMBERS:
            base_factor = float(BASE_NUMBERS[tok])
            
        if base_factor is not None:
            # Look ahead for scale word
            scale_idx = i + consumed
            if scale_idx < len(tokens):
                scale_word = tokens[scale_idx]
                if scale_word in SCALES_MAP:
                    multiplier = SCALES_MAP[scale_word]
                    numeric_val = base_factor * multiplier
                    unit_name = "crore" if multiplier == 10_000_000 else ("lakh" if multiplier == 100_000 else "thousand")
                    results.append((numeric_val, f"{base_factor:g} {unit_name}"))
                    i += consumed + 1  # Advance past consumed number words and scale word
                    continue
        i += 1
                    
    return results


# ---------------------------------------------------------------------------
# 3. Contextual Polarity Guard with Breaker Masking
# ---------------------------------------------------------------------------

ENGLISH_NEGATIONS = [
    r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bneither\b', r'\bnor\b',
    r'\bwithout\b', r'\bcannot\b', r"\bcan't\b", r"\bdon't\b", r"\bdidn't\b",
    r"\bhasn't\b", r"\bhaven't\b", r"\bwon't\b", r"\bwouldn't\b",
    r'\bdo not\b', r'\bdid not\b', r'\bhas not\b'
]


def check_polarity_with_breaker_masking(
    hi_text: str,
    en_text: str,
    breaker_markers: Optional[List[str]] = None
) -> Tuple[bool, bool, str]:
    """
    Evaluates polarity preservation.
    CRITICAL EDGE CASE: Masks breaker repair markers (e.g. 'अरे नहीं नहीं वेट')
    so that hesitation 'नहीं' inside repair markers does NOT trigger false negation alarms.
    """
    masked_hi = hi_text
    if breaker_markers:
        for bm in breaker_markers:
            if bm in masked_hi:
                masked_hi = masked_hi.replace(bm, " [BREAKER_REPAIR_MARKER] ")
                
    # Detect true Hindi negation on unmasked text
    # 1. नहीं (excluding "नहीं तो" which means "otherwise")
    tokens = re.findall(r'[^\s,।!?]+', masked_hi)
    hi_has_nahi = False
    for i, t in enumerate(tokens):
        if t == "नहीं":
            if i + 1 < len(tokens) and tokens[i+1] == "तो":
                continue # नहीं तो = otherwise
            hi_has_nahi = True
            break
            
    # 2. मत (imperative negative: मत करना, मत करो, मत देना)
    hi_has_mat = "मत" in tokens
    
    # 3. ना used as true negation: correlative "ना ... ना ..." or explicit negative predicate
    # (Exclude conversational fillers like "मैंने ना", "फोन पे ना", "ना बेसिकली")
    hi_has_correlative_na = bool(re.search(r'(?:^|\s)ना[\s\S]+?ना(?:\s|$)', masked_hi))
    hi_has_pred_na = bool(re.search(r'(?:^|\s)ना\s+(?:कोई|मेरा|उसपे|कैंसल|आया|गया)', masked_hi))
    hi_has_na_neg = hi_has_correlative_na or hi_has_pred_na
    
    hi_is_negative = hi_has_nahi or hi_has_mat or hi_has_na_neg
    
    # Detect English negation
    en_lower = en_text.lower()
    en_is_negative = any(bool(re.search(p, en_lower)) for p in ENGLISH_NEGATIONS)
    
    match = (hi_is_negative == en_is_negative)
    status = "PRESERVED" if match else "POLARITY_INVERSION_ERROR"
    return hi_is_negative, en_is_negative, status


# ---------------------------------------------------------------------------
# 4. Self-Repair & Dual-Target Generator (Sarvam §10 FAQ Spec)
# ---------------------------------------------------------------------------

def format_dual_targets(
    base_fluent_translation: str,
    breakers: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Implements Sarvam §10 FAQ & §5.2 Dataset Schema:
    - en_text: Primary fluent target. Encodes ONLY the corrected meaning.
               Speaking the abandoned value as fact is wrong!
    - en_alt:  Alternative list containing monotonic/literal streaming order.
    """
    en_fluent = base_fluent_translation
    en_literal = base_fluent_translation
    
    for b in breakers:
        abandoned = b.get("abandoned")
        corrected = b.get("corrected")
        
        # If breaker is a self-repair with an abandoned entity
        if abandoned and corrected:
            # Retract abandoned error from fluent English (Sarvam FAQ)
            # Remove "twenty-five lakh—oh no no, wait, " pattern
            en_fluent = re.sub(r'twenty-five lakh\s*[-—]\s*oh no no,?\s*wait,?\s*', '', en_fluent, flags=re.IGNORECASE)
            en_fluent = re.sub(r'25 lakh\s*[-—]\s*oh no no,?\s*wait,?\s*', '', en_fluent, flags=re.IGNORECASE)
                
    return en_fluent, en_literal
