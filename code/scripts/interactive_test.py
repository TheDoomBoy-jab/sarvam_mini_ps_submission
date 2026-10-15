"""
Interactive Playground for RGNPE Extractors & Pipeline

Test any custom Hindi / Hinglish sentence or run default challenging sentences:
Usage:
    uv run python code/scripts/interactive_test.py "आपका हिंदी वाक्य यहाँ लिखें"
    uv run python code/scripts/interactive_test.py
"""

import sys
import os

# Add code/scripts to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from rgnpe_extractors import (
    parse_spoken_digits,
    parse_relative_scale_amount,
    check_polarity_with_breaker_masking,
    format_dual_targets
)


def inspect_sentence(text: str, breaker_marker: str = None):
    print("=" * 80)
    print(f"INPUT HINDI TEXT: \"{text}\"")
    if breaker_marker:
        print(f"BREAKER MARKER:   \"{breaker_marker}\"")
    print("-" * 80)

    # 1. Spoken Digits
    digits = parse_spoken_digits(text)
    if digits:
        print(f"  [1] Extracted Digits:       {digits} (Length: {len(digits)})")
    else:
        print(f"  [1] Extracted Digits:       None detected")

    # 2. Relative Fractional Scales & Amounts
    amounts = parse_relative_scale_amount(text)
    if amounts:
        for val, desc in amounts:
            print(f"  [2] Extracted Amount:       ₹{val:,.0f} ({desc})")
    else:
        print(f"  [2] Extracted Amount:       None detected")

    # 3. Breaker-Masked Polarity Check
    hi_neg, _, _ = check_polarity_with_breaker_masking(
        text, "", breaker_markers=[breaker_marker] if breaker_marker else None
    )
    print(f"  [3] Source Negation Status: {'NEGATIVE (Requires negative English operator)' if hi_neg else 'AFFIRMATIVE (Non-negative)'}")

    # 4. AI4Bharat Slot Masking Demonstration
    masked = text
    slots = {}
    s_idx = 0
    if digits and len(digits) >= 6:
        slot_id = f"SLOT_DIGITS_{s_idx}"
        slots[slot_id] = digits
        s_idx += 1
    for val, desc in amounts:
        slot_id = f"SLOT_AMOUNT_{s_idx}"
        slots[slot_id] = desc
        s_idx += 1

    print(f"  [4] AI4Bharat Protected Slots: {slots}")
    print("=" * 80 + "\n")


def main():
    if len(sys.argv) > 1:
        custom_sentence = " ".join(sys.argv[1:])
        inspect_sentence(custom_sentence)
    else:
        print("No sentence provided. Running 4 diverse conversational edge-case examples:\n")
        
        # Example 1: Multiplier phone dictation
        inspect_sentence("मेरा नया मोबाइल नंबर नोट करो +91 डबल नाइन डबल आठ शून्य चार पाँच छह सात आठ")
        
        # Example 2: Relative fractional scale in crores
        inspect_sentence("उन्होंने प्रॉपर्टी के लिए पौने तीन करोड़ रुपये मांगे हैं पर बजट सिर्फ सवा दो करोड़ है")
        
        # Example 3: Self-repair with breaker masking
        inspect_sentence("अकाउंट में पच्चीस लाख अरे नहीं नहीं वेट पैंतीस लाख ट्रांसफर करने हैं", breaker_marker="अरे नहीं नहीं वेट")
        
        # Example 4: Conversational filler vs imperative negation
        inspect_sentence("मैंने ना बेसिकली वो चेक क्लियर करवा दिया था पर गलती से भी उसको कैंसल मत करना")


if __name__ == "__main__":
    main()
