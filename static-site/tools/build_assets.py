"""
Copy the static assets the in-browser app needs into static-site/:
  - style.css  (from the original web interface)
  - one reference video per letter into assets/alphabet/<letter>/
  - assets/videos.json  (Romanian-ordered [{letter, url}] list)

Run:  python static-site/tools/build_assets.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_STATIC = ROOT / "web interface" / "app" / "static"
ALPHABET_SRC = ROOT / "alphabet"
SITE = ROOT / "static-site"
ASSETS = SITE / "assets"
ALPHABET_DST = ASSETS / "alphabet"

_RO_ORDER = "AĂÂBCDEFGHIÎJKLMNOPQRSȘTȚUVWXYZ"
_RO_RANK = {c: i for i, c in enumerate(_RO_ORDER)}
VIDEO_EXTS = (".mp4", ".avi", ".webm", ".mov")


def main():
    # 1) style.css
    shutil.copy2(SRC_STATIC / "style.css", SITE / "style.css")
    print("Copied style.css")

    # 2) one video per letter folder + videos.json
    ALPHABET_DST.mkdir(parents=True, exist_ok=True)
    dirs = sorted(
        (d for d in ALPHABET_SRC.iterdir() if d.is_dir()),
        key=lambda p: _RO_RANK.get(p.name.upper(), len(_RO_ORDER)),
    )
    videos = []
    for letter_dir in dirs:
        for vf in sorted(letter_dir.iterdir()):
            if vf.suffix.lower() in VIDEO_EXTS:
                dst_dir = ALPHABET_DST / letter_dir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vf, dst_dir / vf.name)
                videos.append({
                    "letter": letter_dir.name.upper(),
                    "url": f"./assets/alphabet/{letter_dir.name}/{vf.name}",
                })
                break  # one video per letter

    (ASSETS / "videos.json").write_text(
        json.dumps({"videos": videos}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote videos.json ({len(videos)} letters)")


if __name__ == "__main__":
    main()
