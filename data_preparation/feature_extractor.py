"""
Shared per-frame feature extractor.

Used by BOTH the offline pipeline (extract_augmented_fast.py /
extract_gesture_data.py) and the live recognizer (realtime_recognition.py)
so the feature space is identical in training and inference.

Per-frame layout (TOTAL_FEATURES_SIZE features):
    [  0:126]  hand local          - 2 hands x 21 lm x 3 (wrist-centered, scale-norm)
    [126:186]  face anchors        - 20 x 3 (nose-centered, face-height-norm)
    [186:192]  wrist global        - 2 hands x 3, body-anchored (NOT re-centered per frame)
    [192:198]  fingertip global    - 2 hands x 3, body-anchored, index-fingertip
    [198:204]  wrist velocity      - first time-derivative of wrist global
    [204:210]  wrist acceleration  - second time-derivative of wrist global
    [210:212]  hand-present mask   - 1 = hand was actually detected in this frame
    [212:216]  reserved            - kept zero (room for future features without retrain)

The "global" trajectory features are what distinguishes J/Z/Q/Ș/Ț from static
letters - never re-center them per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Layout constants (mirror config.py)
# ---------------------------------------------------------------------------

NUM_HAND_LANDMARKS = 21
HAND_LOCAL_SIZE = 126           # 2 * 21 * 3
NUM_FACE_ANCHORS = 20
FACE_FEATURES_SIZE = NUM_FACE_ANCHORS * 3  # 60
WRIST_GLOBAL_SIZE = 6
FINGERTIP_GLOBAL_SIZE = 6
WRIST_VELOCITY_SIZE = 6
WRIST_ACCEL_SIZE = 6
MASK_RESERVED_SIZE = 6  # 2 mask + 4 reserved

OFF_HAND_LOCAL = 0
OFF_FACE = OFF_HAND_LOCAL + HAND_LOCAL_SIZE
OFF_WRIST_GLOBAL = OFF_FACE + FACE_FEATURES_SIZE
OFF_FINGERTIP_GLOBAL = OFF_WRIST_GLOBAL + WRIST_GLOBAL_SIZE
OFF_WRIST_VELOCITY = OFF_FINGERTIP_GLOBAL + FINGERTIP_GLOBAL_SIZE
OFF_WRIST_ACCEL = OFF_WRIST_VELOCITY + WRIST_VELOCITY_SIZE
OFF_MASK = OFF_WRIST_ACCEL + WRIST_ACCEL_SIZE

TOTAL_FEATURES_SIZE = OFF_MASK + MASK_RESERVED_SIZE  # 216

# Default face anchor index map (kept in sync with config.FACE_MESH_ANCHORS).
DEFAULT_FACE_ANCHORS = {
    'nose_tip': 1, 'nose_bridge': 6,
    'left_eye_inner': 133, 'left_eye_outer': 33, 'left_eye_center': 468,
    'right_eye_inner': 362, 'right_eye_outer': 263, 'right_eye_center': 473,
    'left_eyebrow_inner': 107, 'left_eyebrow_outer': 70,
    'right_eyebrow_inner': 336, 'right_eyebrow_outer': 300,
    'mouth_left': 61, 'mouth_right': 291, 'mouth_top': 13, 'mouth_bottom': 14,
    'chin': 152, 'forehead': 10, 'left_cheek': 123, 'right_cheek': 352,
}

# MediaPipe Pose landmark indices we care about.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

INDEX_FINGERTIP_LM = 8


# ---------------------------------------------------------------------------
# Per-hand state used to bridge MediaPipe dropouts and compute derivatives.
# ---------------------------------------------------------------------------

@dataclass
class _HandSlotState:
    """Tracks per-slot history needed for bridging + velocity."""
    last_local: Optional[np.ndarray] = None       # (21, 3) wrist-centered/scaled
    last_wrist_global: Optional[np.ndarray] = None  # (3,) body-anchored
    last_fingertip_global: Optional[np.ndarray] = None  # (3,) body-anchored
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    frames_since_seen: int = 10**6


# ---------------------------------------------------------------------------
# Stateful frame extractor.
# ---------------------------------------------------------------------------

class FrameFeatureExtractor:
    """
    Stateful per-frame feature extractor.

    Hold one instance per *video stream* (offline video, or the live camera).
    Call `process_frame` once per frame in temporal order. The returned vector
    has shape (TOTAL_FEATURES_SIZE,) and is ready to stack into a sequence.

    State is required because:
      - velocity / acceleration need previous frames
      - we bridge MediaPipe dropouts of <= bridge_frames by holding last-known
      - hand slot assignment is anchored to MediaPipe handedness (or x-sort
        as fallback) and needs to be consistent across frames
    """

    def __init__(
        self,
        face_anchors: Optional[dict] = None,
        bridge_frames: int = 3,
        use_face_mesh: bool = True,
    ):
        self.face_anchors = face_anchors or DEFAULT_FACE_ANCHORS
        self.face_anchor_keys = list(self.face_anchors.keys())
        self._chin_idx = self.face_anchor_keys.index('chin')
        self._forehead_idx = self.face_anchor_keys.index('forehead')

        self.bridge_frames = bridge_frames
        self.use_face_mesh = use_face_mesh

        # Slot 0 = LEFT hand, slot 1 = RIGHT hand (consistent across frames).
        self._slots: List[_HandSlotState] = [_HandSlotState(), _HandSlotState()]

    def reset(self):
        self._slots = [_HandSlotState(), _HandSlotState()]

    # ------------------------------------------------------------------
    # Body-anchor frame
    # ------------------------------------------------------------------

    def _body_anchor(self, pose_landmarks) -> Optional[Tuple[np.ndarray, float]]:
        """
        Returns (anchor_origin, body_scale) where positions are in
        normalized image coordinates. Origin = shoulder midpoint, scale =
        shoulder-to-hip distance. Returns None if pose is not detected.
        """
        if pose_landmarks is None:
            return None
        lm = pose_landmarks.landmark
        ls = np.array([lm[POSE_LEFT_SHOULDER].x, lm[POSE_LEFT_SHOULDER].y, lm[POSE_LEFT_SHOULDER].z], dtype=np.float32)
        rs = np.array([lm[POSE_RIGHT_SHOULDER].x, lm[POSE_RIGHT_SHOULDER].y, lm[POSE_RIGHT_SHOULDER].z], dtype=np.float32)
        lh = np.array([lm[POSE_LEFT_HIP].x, lm[POSE_LEFT_HIP].y, lm[POSE_LEFT_HIP].z], dtype=np.float32)
        rh = np.array([lm[POSE_RIGHT_HIP].x, lm[POSE_RIGHT_HIP].y, lm[POSE_RIGHT_HIP].z], dtype=np.float32)
        shoulder_mid = (ls + rs) * 0.5
        hip_mid = (lh + rh) * 0.5
        torso_height = float(np.linalg.norm(shoulder_mid - hip_mid)) + 1e-6
        return shoulder_mid, torso_height

    def _face_anchor(self, face_landmarks) -> Optional[Tuple[np.ndarray, float]]:
        """Fallback body-anchor when pose is unavailable: nose + face-height."""
        if face_landmarks is None:
            return None
        lm = face_landmarks.landmark
        nose = np.array([lm[1].x, lm[1].y, lm[1].z], dtype=np.float32)
        chin = np.array([lm[152].x, lm[152].y, lm[152].z], dtype=np.float32)
        forehead = np.array([lm[10].x, lm[10].y, lm[10].z], dtype=np.float32)
        face_height = float(np.linalg.norm(forehead - chin)) + 1e-6
        # Use ~3x face height as a torso-equivalent scale.
        return nose + np.array([0.0, face_height * 1.5, 0.0], dtype=np.float32), face_height * 3.0

    # ------------------------------------------------------------------
    # Hand slot assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_hand_slots(hand_landmarks_list, handedness_list) -> List[Optional[object]]:
        """
        Returns [left_hand_lm_or_None, right_hand_lm_or_None].

        Uses MediaPipe handedness when available; falls back to x-sort
        (left = smaller x in mirrored image) only if labels are missing.

        Note: MediaPipe operates on a non-mirrored frame internally, so
        'Left' label = signer's right hand from the camera's POV. For our
        purposes we only need *consistent* slotting frame-to-frame, so we
        just use the labels as MediaPipe gives them.
        """
        if not hand_landmarks_list:
            return [None, None]

        slots: List[Optional[object]] = [None, None]

        if handedness_list and len(handedness_list) == len(hand_landmarks_list):
            for hand_lm, handedness in zip(hand_landmarks_list, handedness_list):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                idx = 0 if label == 'Left' else 1
                if slots[idx] is None:
                    slots[idx] = hand_lm
                else:
                    # Two hands with same label -> assign by lower confidence to other slot
                    other = 1 - idx
                    if slots[other] is None:
                        slots[other] = hand_lm
        else:
            # Fallback: sort by wrist x.
            sorted_hands = sorted(
                hand_landmarks_list[:2],
                key=lambda h: h.landmark[0].x,
            )
            for i, hand_lm in enumerate(sorted_hands[:2]):
                slots[i] = hand_lm

        return slots

    # ------------------------------------------------------------------
    # Per-hand local features (wrist-centered, scale-normalized)
    # ------------------------------------------------------------------

    @staticmethod
    def _local_handshape(hand_lm) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            local_landmarks (21, 3)  - wrist-centered, max-norm scaled
            wrist_xyz (3,)           - in original image coordinates
            fingertip_xyz (3,)       - index-fingertip in original image coordinates
        """
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
            dtype=np.float32,
        )
        wrist = raw[0].copy()
        fingertip = raw[INDEX_FINGERTIP_LM].copy()
        centered = raw - wrist
        scale = float(np.max(np.abs(centered))) + 1e-7
        local = centered / scale
        return local, wrist, fingertip

    # ------------------------------------------------------------------
    # Face features (nose-centered, face-height-normalized)
    # ------------------------------------------------------------------

    def _face_features(self, face_landmarks) -> np.ndarray:
        out = np.zeros(FACE_FEATURES_SIZE, dtype=np.float32)
        if face_landmarks is None:
            return out

        lm = face_landmarks.landmark
        n = len(lm)
        anchors = []
        for name, idx in self.face_anchors.items():
            if idx >= n:
                if 'left_eye_center' in name:
                    idx = 159
                elif 'right_eye_center' in name:
                    idx = 386
                else:
                    idx = 0
            anchors.append([lm[idx].x, lm[idx].y, lm[idx].z])
        anchors = np.asarray(anchors, dtype=np.float32)

        nose = anchors[0].copy()
        face_height = float(np.linalg.norm(anchors[self._forehead_idx] - anchors[self._chin_idx])) + 1e-6
        normalized = (anchors - nose) / face_height

        flat = normalized.flatten()
        out[: min(len(flat), FACE_FEATURES_SIZE)] = flat[:FACE_FEATURES_SIZE]
        return out

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        hand_landmarks_list,
        handedness_list,
        face_landmarks,
        pose_landmarks,
    ) -> np.ndarray:
        """
        Build the per-frame feature vector.

        Args:
            hand_landmarks_list: results.multi_hand_landmarks (list or None)
            handedness_list:     results.multi_handedness (list or None)
            face_landmarks:      single face landmarks object or None
            pose_landmarks:      single pose landmarks object or None
        """
        feat = np.zeros(TOTAL_FEATURES_SIZE, dtype=np.float32)

        # --- Block B: face ---
        if self.use_face_mesh and face_landmarks is not None:
            feat[OFF_FACE:OFF_FACE + FACE_FEATURES_SIZE] = self._face_features(face_landmarks)

        # --- Body anchor (pose preferred, face fallback) ---
        anchor = self._body_anchor(pose_landmarks)
        if anchor is None:
            anchor = self._face_anchor(face_landmarks)
        anchor_origin, anchor_scale = (anchor if anchor is not None else (None, None))

        # --- Hand slot assignment ---
        slots = self._assign_hand_slots(
            hand_landmarks_list if hand_landmarks_list else [],
            handedness_list if handedness_list else [],
        )

        # --- Per-slot processing ---
        for slot_idx, hand_lm in enumerate(slots):
            state = self._slots[slot_idx]
            local_off = OFF_HAND_LOCAL + slot_idx * 63
            wrist_off = OFF_WRIST_GLOBAL + slot_idx * 3
            tip_off = OFF_FINGERTIP_GLOBAL + slot_idx * 3
            vel_off = OFF_WRIST_VELOCITY + slot_idx * 3
            acc_off = OFF_WRIST_ACCEL + slot_idx * 3
            mask_off = OFF_MASK + slot_idx

            if hand_lm is not None:
                local, wrist_xyz, tip_xyz = self._local_handshape(hand_lm)
                if anchor_origin is not None:
                    wrist_global = (wrist_xyz - anchor_origin) / anchor_scale
                    tip_global = (tip_xyz - anchor_origin) / anchor_scale
                else:
                    # No body anchor: fall back to raw image coords (still
                    # better than zeros — the model can learn the offset).
                    wrist_global = wrist_xyz - 0.5
                    tip_global = tip_xyz - 0.5

                # Velocity / acceleration vs previous frame.
                if state.last_wrist_global is not None and state.frames_since_seen <= self.bridge_frames + 1:
                    velocity = wrist_global - state.last_wrist_global
                    accel = velocity - state.last_velocity
                else:
                    velocity = np.zeros(3, dtype=np.float32)
                    accel = np.zeros(3, dtype=np.float32)

                feat[local_off:local_off + 63] = local.flatten()
                feat[wrist_off:wrist_off + 3] = wrist_global
                feat[tip_off:tip_off + 3] = tip_global
                feat[vel_off:vel_off + 3] = velocity
                feat[acc_off:acc_off + 3] = accel
                feat[mask_off] = 1.0

                state.last_local = local
                state.last_wrist_global = wrist_global.astype(np.float32)
                state.last_fingertip_global = tip_global.astype(np.float32)
                state.last_velocity = velocity.astype(np.float32)
                state.frames_since_seen = 0
            else:
                # Missing hand. Bridge if recent.
                state.frames_since_seen += 1
                if (
                    state.last_local is not None
                    and state.last_wrist_global is not None
                    and state.frames_since_seen <= self.bridge_frames
                ):
                    feat[local_off:local_off + 63] = state.last_local.flatten()
                    feat[wrist_off:wrist_off + 3] = state.last_wrist_global
                    feat[tip_off:tip_off + 3] = (
                        state.last_fingertip_global
                        if state.last_fingertip_global is not None
                        else state.last_wrist_global
                    )
                    # Velocity decays to 0 during bridging — the hand isn't
                    # actually moving, MediaPipe just lost it.
                    feat[vel_off:vel_off + 3] = 0.0
                    feat[acc_off:acc_off + 3] = 0.0
                    feat[mask_off] = 0.5  # bridged
                else:
                    # Genuinely absent: leave zeros, mask=0.
                    feat[mask_off] = 0.0

        return feat


# ---------------------------------------------------------------------------
# Sequence helpers used by the offline pipeline.
# ---------------------------------------------------------------------------

def hand_presence_ratio(sequence: np.ndarray) -> float:
    """Fraction of frames in the sequence with at least one hand detected."""
    if sequence.size == 0:
        return 0.0
    masks = sequence[:, OFF_MASK:OFF_MASK + 2]
    per_frame_present = (masks > 0.0).any(axis=1)
    return float(per_frame_present.mean())


def resample_frame_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    """
    Indices to read from a clip of `num_frames` at `source_fps` so the
    resulting stream is at `target_fps`. Always returns a strictly
    increasing index array; falls through to identity if FPS info is bad.
    """
    if source_fps is None or source_fps <= 0 or target_fps <= 0:
        return np.arange(num_frames)
    if abs(source_fps - target_fps) < 0.5:
        return np.arange(num_frames)
    duration_sec = num_frames / source_fps
    target_count = max(1, int(round(duration_sec * target_fps)))
    if target_count >= num_frames:
        return np.arange(num_frames)
    # Even-spaced indices.
    return np.linspace(0, num_frames - 1, target_count).astype(int)


__all__ = [
    'FrameFeatureExtractor',
    'TOTAL_FEATURES_SIZE',
    'HAND_LOCAL_SIZE',
    'FACE_FEATURES_SIZE',
    'WRIST_GLOBAL_SIZE',
    'FINGERTIP_GLOBAL_SIZE',
    'WRIST_VELOCITY_SIZE',
    'WRIST_ACCEL_SIZE',
    'MASK_RESERVED_SIZE',
    'OFF_HAND_LOCAL', 'OFF_FACE', 'OFF_WRIST_GLOBAL',
    'OFF_FINGERTIP_GLOBAL', 'OFF_WRIST_VELOCITY', 'OFF_WRIST_ACCEL', 'OFF_MASK',
    'DEFAULT_FACE_ANCHORS',
    'hand_presence_ratio',
    'resample_frame_indices',
]
