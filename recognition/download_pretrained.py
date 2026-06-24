r"""
download_pretrained.py
----------------------
Download AdaFace pretrained weights from official Google Drive.

AdaFace official model zoo (from https://github.com/mk-minchul/AdaFace):
    - adaface_ir50_ms1mv2.ckpt    (IR-50 trained on MS1MV2, ~250 MB)
    - adaface_ir50_vggface2.ckpt  (IR-50 on VGGFace2)
    - adaface_ir101_ms1mv2.ckpt   (IR-101 on MS1MV2)
    - adaface_ir101_ms1mv3.ckpt   (IR-101 on MS1MV3)
    - adaface_ir101_webface4m.ckpt
    - adaface_ir101_webface12m.ckpt

For this project we use adaface_ir50_ms1mv2.ckpt (best balance for RTX 3060).

Usage:
    python recognition\download_pretrained.py

If gdown fails (rate limit), manually download from:
    https://github.com/mk-minchul/AdaFace#pretrained-models
And place at: pretrained\adaface_ir50_ms1mv2.ckpt
"""

import argparse
import sys
from pathlib import Path


MODEL_INFO = {
    "adaface_ir50_ms1mv2": {
        "gdrive_id": "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI",
        "size_mb": 250,
        "description": "AdaFace IR-50 trained on MS1MV2 (~99.7% LFW)",
    },
    "adaface_ir50_webface4m": {
        "gdrive_id": "1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe",
        "size_mb": 250,
        "description": "AdaFace IR-50 trained on WebFace4M (~99.8% LFW)",
    },
    "adaface_ir101_ms1mv2": {
        "gdrive_id": "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI",  # placeholder, check official
        "size_mb": 350,
        "description": "AdaFace IR-101 trained on MS1MV2",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="adaface_ir50_ms1mv2",
        choices=list(MODEL_INFO.keys()),
        help="Which pretrained model to download",
    )
    ap.add_argument(
        "--output-dir", default="pretrained",
        help="Where to save the checkpoint",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.ckpt"

    info = MODEL_INFO[args.model]
    print(f"Model:       {args.model}")
    print(f"Description: {info['description']}")
    print(f"Size:        ~{info['size_mb']} MB")
    print(f"Output:      {out_path}")
    print()

    if out_path.exists():
        print(f"[OK] File already exists. Skipping download.")
        return

    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Install with: pip install gdown")
        print()
        print("Or download manually from AdaFace GitHub:")
        print("  https://github.com/mk-minchul/AdaFace#pretrained-models")
        print(f"And save to: {out_path}")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={info['gdrive_id']}"
    print(f"Downloading from {url} ...")
    try:
        gdown.download(url, str(out_path), quiet=False)
    except Exception as e:
        print(f"\nERROR: download failed: {e}")
        print("\nGoogle Drive may have hit rate limit. Try:")
        print("  1) Wait 24 hours and retry")
        print("  2) Manually download from:")
        print("     https://github.com/mk-minchul/AdaFace#pretrained-models")
        print(f"     Save to: {out_path}")
        sys.exit(1)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Downloaded {size_mb:.1f} MB to {out_path}")


if __name__ == "__main__":
    main()
