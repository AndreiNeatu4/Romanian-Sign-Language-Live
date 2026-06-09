"""
Model loader and real-time inference engine.

Mirrors training/train_model.py (CNNBiLSTMAttnModel) and
data_preparation/feature_extractor.py so the feature space + architecture
are identical to the latest offline training run.
"""
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from .feature_extractor import (
    DEFAULT_FACE_ANCHORS,
    FrameFeatureExtractor,
    TOTAL_FEATURES_SIZE,
)

logger = logging.getLogger(__name__)

# Paths – override with env vars if needed
_DEFAULT_BASE = Path(__file__).resolve().parent.parent.parent  # handle keypoints/
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_DEFAULT_BASE / "models")))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_DEFAULT_BASE / "data")))

# MediaPipe hand skeleton connections used for the frontend drawing.
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
]


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class CNNBiLSTMAttnModel(nn.Module):
    """CNN + bidirectional LSTM + attention pooling. Matches training script."""

    def __init__(self, input_size: int, num_classes: int = 30, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.attn_proj = nn.Linear(256, 128)
        self.attn_score = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        attn_h = torch.tanh(self.attn_proj(x))
        attn_logits = self.attn_score(attn_h).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# Gesture recognizer
# ---------------------------------------------------------------------------


class GestureRecognizer:
    SEQUENCE_LENGTH = 45  # matches config.SEQUENCE_LENGTH

    def __init__(self):
        self.model_loaded = False
        self.model: Optional[nn.Module] = None
        # Use the RTX 4080 when it is exposed to the container; otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            # cuDNN autotuner picks the fastest kernels for our fixed input shape.
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA available — running BiLSTM on %s", torch.cuda.get_device_name(0))
        else:
            logger.info("CUDA not available — running BiLSTM on CPU")
        self.class_names: list[str] = []
        self.input_size: int = TOTAL_FEATURES_SIZE
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=self.SEQUENCE_LENGTH)
        self.last_prediction: Optional[dict] = None
        self._was_hand_visible = False

        self._load_class_names()
        self._load_model()
        self._init_mediapipe()

        # Stateful per-stream feature extractor.
        self._ffe = FrameFeatureExtractor(
            face_anchors=DEFAULT_FACE_ANCHORS,
            bridge_frames=3,
            use_face_mesh=True,
        )

        # Face mesh and pose run every other frame — they change slowly and
        # are the heaviest MediaPipe models.  Running them on every frame is
        # the main cause of rising inference time over a long session.
        self._slow_frame_idx = 0
        self._last_face_result = None
        self._last_pose_result = None

    # ------------------------------------------------------------------
    # Class label loading
    # ------------------------------------------------------------------

    def _load_class_names(self) -> None:
        candidates = [
            MODEL_DIR / "alphabet" / "class_labels.json",
            DATA_DIR / "alphabet_processed" / "class_labels.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.class_names = data
            elif isinstance(data, dict):
                if "classes" in data and isinstance(data["classes"], list):
                    self.class_names = data["classes"]
                elif "label_to_name" in data:
                    d = data["label_to_name"]
                    max_idx = max(int(k) for k in d)
                    self.class_names = [d[str(i)] for i in range(max_idx + 1)]
                elif "0" in data or 0 in data:
                    max_idx = max(int(k) for k in data)
                    self.class_names = [
                        data.get(str(i), data.get(i, str(i)))
                        for i in range(max_idx + 1)
                    ]
                else:
                    inv = {v: k for k, v in data.items() if not isinstance(v, (list, dict))}
                    self.class_names = [inv[i] for i in range(len(inv))]
            logger.info("Loaded %d class names from %s", len(self.class_names), path)
            return

        self.class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Â", "Ă", "Ș", "Ț"]
        logger.warning("class_labels.json not found – using default Romanian alphabet")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_path = MODEL_DIR / "alphabet" / "best_model.pth"
        if not model_path.exists():
            logger.error("best_model.pth not found at %s", model_path)
            return

        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            self.model = CNNBiLSTMAttnModel(
                input_size=TOTAL_FEATURES_SIZE,
                num_classes=len(self.class_names),
                dropout=0.3,
            )
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info(
                "Loaded cnn_bilstm_attn (%d classes, input_size=%d) from %s",
                len(self.class_names), TOTAL_FEATURES_SIZE, model_path,
            )
        except Exception as exc:
            logger.error("Failed to load model weights: %s", exc)

    # ------------------------------------------------------------------
    # MediaPipe init
    # ------------------------------------------------------------------

    def _init_mediapipe(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("MediaPipe initialised (Hands + FaceMesh + Pose)")

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_hands(lm_list, hd_list, threshold: float = 0.12):
        """Drop one detection when two wrists fall on the same physical hand.
        MediaPipe sometimes labels the same hand as both Left and Right for
        1-2 frames; the wrist (landmark 0) positions end up nearly identical."""
        if not lm_list or len(lm_list) < 2:
            return lm_list, hd_list
        w0, w1 = lm_list[0].landmark[0], lm_list[1].landmark[0]
        dx, dy = w0.x - w1.x, w0.y - w1.y
        if (dx * dx + dy * dy) ** 0.5 < threshold:
            s0 = hd_list[0].classification[0].score if hd_list[0].classification else 0.0
            s1 = hd_list[1].classification[0].score if hd_list[1].classification else 0.0
            keep = 0 if s0 >= s1 else 1
            return [lm_list[keep]], [hd_list[keep]]
        return lm_list, hd_list

    @staticmethod
    def _raw_hands_for_frontend(lm_list) -> list[list[list[float]]]:
        if not lm_list:
            return []
        return [[[lm.x, lm.y] for lm in hand_lm.landmark] for hand_lm in lm_list]

    def _raw_face_for_frontend(self, face_results) -> list[list[float]]:
        if not (face_results and face_results.multi_face_landmarks):
            return []
        face_lm = face_results.multi_face_landmarks[0]
        out: list[list[float]] = []
        for idx in DEFAULT_FACE_ANCHORS.values():
            if idx < len(face_lm.landmark):
                lm = face_lm.landmark[idx]
                out.append([lm.x, lm.y])
            else:
                out.append([0.0, 0.0])
        return out

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process one BGR frame and return a JSON-serialisable result dict."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self._hands.process(rgb)

        # Run face mesh and pose on alternate frames only.
        self._slow_frame_idx += 1
        if self._slow_frame_idx % 3 == 0:
            self._last_face_result = self._face_mesh.process(rgb)
            self._last_pose_result = self._pose.process(rgb)
        face_results = self._last_face_result
        pose_results = self._last_pose_result

        face_lm = (
            face_results.multi_face_landmarks[0]
            if face_results and face_results.multi_face_landmarks
            else None
        )
        pose_lm = pose_results.pose_landmarks if pose_results and pose_results.pose_landmarks else None

        # Remove duplicate detections of the same physical hand.
        hand_lm_list = hand_results.multi_hand_landmarks or None
        hand_hd_list = hand_results.multi_handedness or None
        if hand_lm_list and hand_hd_list:
            hand_lm_list, hand_hd_list = self._deduplicate_hands(hand_lm_list, hand_hd_list)

        features = self._ffe.process_frame(
            hand_landmarks_list=hand_lm_list,
            handedness_list=hand_hd_list,
            face_landmarks=face_lm,
            pose_landmarks=pose_lm,
        )

        raw_hands = self._raw_hands_for_frontend(hand_lm_list)
        raw_face = self._raw_face_for_frontend(face_results)
        has_hand = len(raw_hands) > 0

        # Pre-fill the buffer on hand re-entry so we don't carry zero frames.
        if has_hand and not self._was_hand_visible:
            self.frame_buffer.clear()
            for _ in range(self.SEQUENCE_LENGTH):
                self.frame_buffer.append(features)
        else:
            self.frame_buffer.append(features)
        self._was_hand_visible = has_hand

        result: dict = {
            "buffer_fill": len(self.frame_buffer),
            "buffer_max": self.SEQUENCE_LENGTH,
            "hands": raw_hands,
            "face": raw_face,
            "hand_connections": HAND_CONNECTIONS,
            "prediction": None,
            "confidence": 0.0,
            "top_predictions": [],
            "frozen": False,
        }

        if has_hand and len(self.frame_buffer) == self.SEQUENCE_LENGTH and self.model_loaded:
            sequence = np.asarray(self.frame_buffer, dtype=np.float32)
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)  # (1, T, F)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

            top5 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
            result["prediction"] = self.class_names[top5[0]]
            result["confidence"] = float(probs[top5[0]])
            result["top_predictions"] = [
                {"label": self.class_names[i], "confidence": float(probs[i])}
                for i in top5
            ]
            self.last_prediction = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "top_predictions": result["top_predictions"],
            }
        elif not has_hand and self.last_prediction is not None:
            # Freeze on last good prediction while no hand is visible.
            result["prediction"] = self.last_prediction["prediction"]
            result["confidence"] = self.last_prediction["confidence"]
            result["top_predictions"] = self.last_prediction["top_predictions"]
            result["frozen"] = True

        return result

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.last_prediction = None
        self._was_hand_visible = False
        self._ffe.reset()
