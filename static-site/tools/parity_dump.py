"""
Generate a random landmark sequence, run the REAL Python FrameFeatureExtractor,
and dump both the raw landmark inputs and the resulting 216-dim features to JSON.

parity_check.mjs then feeds the identical inputs through feature_extractor.js and
asserts the features match — proving the JS port is numerically faithful.

Run:  python static-site/tools/parity_dump.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FE_PATH = ROOT / "web interface" / "app" / "feature_extractor.py"
OUT = Path(__file__).resolve().parent / "parity_fixture.json"

# Import feature_extractor.py (path has spaces) without importing the package.
spec = importlib.util.spec_from_file_location("fe_py", FE_PATH)
fe = importlib.util.module_from_spec(spec)
sys.modules["fe_py"] = fe  # dataclass needs the module registered before exec
spec.loader.exec_module(fe)


# --- minimal MediaPipe-shaped mock objects ---------------------------------
class _LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z


class _Landmarks:
    def __init__(self, pts): self.landmark = [_LM(*p) for p in pts]


class _Classification:
    def __init__(self, label): self.label = label


class _Handedness:
    def __init__(self, label): self.classification = [_Classification(label)]


def rand_pts(n, rng):
    return rng.random((n, 3)).astype(np.float32).tolist()


def main():
    rng = np.random.default_rng(1234)
    ext = fe.FrameFeatureExtractor(bridge_frames=3, use_face_mesh=True)

    NUM_FRAMES = 60
    frames_json = []
    features = []

    for t in range(NUM_FRAMES):
        # Vary presence to exercise bridging, velocity, slot assignment.
        has_left = (t % 7) != 0          # left hand drops out periodically
        has_right = (t % 11) != 2
        has_face = (t % 5) != 4
        has_pose = (t % 3) != 1          # forces face-anchor fallback sometimes

        hands_pts, labels = [], []
        if has_right:
            hands_pts.append(rand_pts(21, rng)); labels.append("Right")
        if has_left:
            hands_pts.append(rand_pts(21, rng)); labels.append("Left")

        face_pts = rand_pts(478, rng) if has_face else None
        pose_pts = rand_pts(33, rng) if has_pose else None

        # Build mock objects for the Python extractor.
        hand_objs = [_Landmarks(p) for p in hands_pts] if hands_pts else None
        hd_objs = [_Handedness(l) for l in labels] if labels else None
        face_obj = _Landmarks(face_pts) if face_pts else None
        pose_obj = _Landmarks(pose_pts) if pose_pts else None

        feat = ext.process_frame(
            hand_landmarks_list=hand_objs,
            handedness_list=hd_objs,
            face_landmarks=face_obj,
            pose_landmarks=pose_obj,
        )
        features.append([float(v) for v in feat])

        frames_json.append({
            "hands": hands_pts,        # list of [21][3]
            "handedness": labels,      # parallel labels
            "face": face_pts,          # [478][3] or null
            "pose": pose_pts,          # [33][3] or null
        })

    OUT.write_text(json.dumps({
        "total_features": int(fe.TOTAL_FEATURES_SIZE),
        "frames": frames_json,
        "expected_features": features,
    }), encoding="utf-8")
    print(f"Wrote {OUT} ({NUM_FRAMES} frames)")


if __name__ == "__main__":
    main()
