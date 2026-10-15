"""
Model Artifact Verification and Fetch Utility
Meta MMS 300M CTC Trellis Forced Alignment Model
"""

import sys
from pathlib import Path
import torch

LOCAL_DVC_PATH = Path(__file__).resolve().parent.parent / "models" / "mms_fa" / "mms_300m.pt"
DVC_POINTER = Path(__file__).resolve().parent.parent / "models" / "mms_fa" / "mms_300m.pt.dvc"

def check_and_fetch_model():
    print("=== Meta MMS 300M Model Artifact Audit ===")
    print(f"DVC Pointer: {DVC_POINTER}")
    if DVC_POINTER.exists():
        print(f"✓ DVC pointer verified:\n{DVC_POINTER.read_text().strip()}")
    else:
        print("✗ Warning: DVC pointer file missing!")

    if LOCAL_DVC_PATH.exists():
        size_mb = LOCAL_DVC_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Local DVC model artifact found: {LOCAL_DVC_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"• Local DVC artifact not materialized at {LOCAL_DVC_PATH}")
        print("  Run 'dvc pull' or fetch from torchaudio.pipelines.MMS_FA...")

    # Verify model loadability via torchaudio bundle
    try:
        from torchaudio.pipelines import MMS_FA as bundle
        print("• Verifying torchaudio MMS_FA bundle initialization...")
        model = bundle.get_model()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Successfully loaded Meta MMS 300M model ({num_params / 1e6:.1f}M parameters)")
        labels = bundle.get_labels()
        print(f"✓ CTC Vocab Dimension: {len(labels)} labels (star token included: {'*' in labels})")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = check_and_fetch_model()
    sys.exit(0 if success else 1)
