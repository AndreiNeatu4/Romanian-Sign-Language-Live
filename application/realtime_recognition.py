"""
Real-Time Gesture Recognition (60fps capture, 30fps inference)
==============================================================

Mirrors the offline training feature pipeline 1:1 by reusing
`data_preparation.feature_extractor.FrameFeatureExtractor`. The webcam runs
at 60fps for smooth display; MediaPipe + the model run on every other
frame so the inference stream is at the same TARGET_FPS (30fps) the model
was trained on.

Controls:
- 'q' quit
- 'r' reset frame buffer
- '+' / '-' adjust motion sensitivity
- 's' / 'a' adjust static-confidence override
"""

import cv2
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# Add parent directory for imports.
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.feature_extractor import (  # noqa: E402
    DEFAULT_FACE_ANCHORS,
    FrameFeatureExtractor,
    OFF_FACE,
    OFF_FINGERTIP_GLOBAL,
    OFF_HAND_LOCAL,
    OFF_MASK,
    OFF_WRIST_ACCEL,
    OFF_WRIST_GLOBAL,
    OFF_WRIST_VELOCITY,
    TOTAL_FEATURES_SIZE,
)


# ============================================================================
# MODEL - identical layout to training/train_model.py CNNBiLSTMAttnModel
# ============================================================================


class CNNBiLSTMAttnModel(nn.Module):
    def __init__(self, input_size, num_classes=10, dropout=0.3):
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

    def forward(self, x):
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


class CNNLSTMModel(nn.Module):
    """Kept for backwards-compat with old (186-feature) checkpoints."""

    def __init__(self, input_size, num_classes=10, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


# ============================================================================
# THREADED VIDEO CAPTURE
# ============================================================================


class VideoCapture:
    def __init__(self, src=0, width=1280, height=720, fps=60):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()


# ============================================================================
# RECOGNIZER
# ============================================================================


class RealtimeGestureRecognizer:
    def __init__(
        self,
        model_path: str,
        class_labels_path: str,
        sequence_length: int = 45,
        confidence_threshold: float = 0.7,
        use_face_mesh: bool = True,
        use_pose: bool = True,
        target_fps: float = 30.0,
        face_anchors=None,
    ):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_face_mesh = use_face_mesh
        self.use_pose = use_pose
        self.target_fps = target_fps

        self.face_anchors = face_anchors or DEFAULT_FACE_ANCHORS

        # Load class labels.
        with open(class_labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        self.class_names = labels_data['classes']
        self.num_classes = len(self.class_names)
        print(f"Loaded {self.num_classes} gesture classes")

        # Build model and load weights. Auto-detect input size from the
        # checkpoint so old (186) and new (216) checkpoints both work.
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        input_size = self._infer_input_size_from_state_dict(state_dict)
        self.input_size = input_size
        print(f"Inferred input feature size from checkpoint: {input_size}")

        if self._looks_like_bilstm_attn(state_dict):
            self.model = CNNBiLSTMAttnModel(
                input_size=input_size, num_classes=self.num_classes, dropout=0.3,
            )
            self.model_kind = 'cnn_bilstm_attn'
        else:
            self.model = CNNLSTMModel(
                input_size=input_size, num_classes=self.num_classes, dropout=0.3,
            )
            self.model_kind = 'cnn_lstm'
        print(f"Model architecture: {self.model_kind}")

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        # MediaPipe Face Mesh.
        self.face_mesh = None
        if self.use_face_mesh:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        # MediaPipe Pose.
        self.pose = None
        if self.use_pose:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        # Shared feature extractor (stateful - one instance per stream).
        self.ffe = FrameFeatureExtractor(
            face_anchors=self.face_anchors,
            bridge_frames=3,
            use_face_mesh=self.use_face_mesh,
        )

        # Hand-count hysteresis: MediaPipe occasionally flickers a forearm /
        # mirror as a "second hand" for 1-2 frames. The FFE bridges DROPOUTS
        # but accepts NEW hands instantly, so a single flicker is enough to
        # turn V into W. We require a hand to be detected for HAND_WARMUP
        # consecutive frames (per left/right slot) before forwarding it.
        self.HAND_WARMUP_FRAMES = 3
        self._hand_warmup = {'Left': 0, 'Right': 0}

        # Pose and face mesh run every other inference frame to reduce CPU load.
        # Shoulders/face change slowly so stale results are imperceptible.
        self._slow_model_idx = 0
        self._last_face_result = None
        self._last_pose_result = None

        # Sequence buffer.
        self.frame_buffer = deque(maxlen=sequence_length)

        # Latest landmarks for drawing (thread-safe).
        self.latest_hand_landmarks = None
        self.latest_face_landmarks = None
        self.latest_pose_landmarks = None
        self.landmarks_lock = threading.Lock()

        # Prediction state.
        self.current_prediction = None
        self.current_confidence = 0.0
        self.prediction_lock = threading.Lock()

        # Motion detection thresholds.
        self.motion_threshold = 0.015
        self.is_moving = False

        # Static-gesture override.
        self.static_confidence_threshold = 0.90
        self.is_static_gesture = False

        # Display timer.
        self.display_duration = 5.0
        self.last_prediction_time = 0
        self.displayed_prediction = None
        self.displayed_confidence = 0.0

        # Processing thread.
        self.processing_frame = None
        self.processing_lock = threading.Lock()
        self.processing_running = True
        self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)

        # FPS counters.
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Inference rate-limiting: only feed every Nth captured frame to
        # MediaPipe so the per-frame TIME between buffer entries matches
        # 1 / target_fps. With 60fps capture and target_fps=30, we keep
        # every 2nd frame.
        self.capture_fps_estimate = 60.0
        self._capture_frame_idx = 0
        self._refresh_capture_skip()

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_input_size_from_state_dict(sd: dict) -> int:
        # New model: input_norm.weight has shape (input_size,).
        if 'input_norm.weight' in sd:
            return int(sd['input_norm.weight'].shape[0])
        # Old model: conv1.weight has shape (out, input_size, k).
        if 'conv1.weight' in sd:
            return int(sd['conv1.weight'].shape[1])
        raise ValueError("Could not infer input size from checkpoint state_dict")

    @staticmethod
    def _looks_like_bilstm_attn(sd: dict) -> bool:
        return 'attn_proj.weight' in sd and 'input_norm.weight' in sd

    # ------------------------------------------------------------------
    # Capture-rate <-> target-FPS bookkeeping
    # ------------------------------------------------------------------

    def _refresh_capture_skip(self):
        if self.capture_fps_estimate <= 0:
            self._capture_skip = 1
            return
        skip = max(1, int(round(self.capture_fps_estimate / self.target_fps)))
        self._capture_skip = skip

    # ------------------------------------------------------------------
    # Background processing thread
    # ------------------------------------------------------------------

    def _processing_loop(self):
        while self.processing_running:
            with self.processing_lock:
                frame = self.processing_frame
                self.processing_frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            # Resize to half-resolution for MediaPipe — landmarks are returned
            # as normalized [0,1] coordinates so display drawing is unaffected.
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            hand_results = self.hands.process(frame_rgb)

            # Run pose and face mesh every other inference frame; reuse the
            # cached result on skipped frames.
            self._slow_model_idx += 1
            run_slow = (self._slow_model_idx % 2 == 0)

            if run_slow:
                face_results = self.face_mesh.process(frame_rgb) if self.face_mesh is not None else None
                pose_results = self.pose.process(frame_rgb) if self.pose is not None else None
                self._last_face_result = face_results
                self._last_pose_result = pose_results
            else:
                face_results = self._last_face_result
                pose_results = self._last_pose_result

            face_lm = face_results.multi_face_landmarks[0] if (face_results and face_results.multi_face_landmarks) else None
            pose_lm = pose_results.pose_landmarks if (pose_results and pose_results.pose_landmarks) else None

            raw_landmarks = hand_results.multi_hand_landmarks if hand_results else None
            raw_handedness = hand_results.multi_handedness if hand_results else None
            filtered_landmarks, filtered_handedness = self._apply_hand_hysteresis(
                raw_landmarks, raw_handedness
            )

            with self.landmarks_lock:
                self.latest_hand_landmarks = filtered_landmarks
                self.latest_face_landmarks = face_results.multi_face_landmarks if (face_results and face_results.multi_face_landmarks) else None
                self.latest_pose_landmarks = pose_lm

            features = self.ffe.process_frame(
                hand_landmarks_list=filtered_landmarks,
                handedness_list=filtered_handedness,
                face_landmarks=face_lm,
                pose_landmarks=pose_lm,
            )

            # Pad/trim to the model's expected input size (handles old 186-dim
            # checkpoints too).
            if features.shape[0] != self.input_size:
                if features.shape[0] > self.input_size:
                    features = features[: self.input_size]
                else:
                    pad = np.zeros(self.input_size - features.shape[0], dtype=np.float32)
                    features = np.concatenate([features, pad])

            self.frame_buffer.append(features)

            if len(self.frame_buffer) == self.sequence_length:
                sequence = np.asarray(self.frame_buffer, dtype=np.float32)
                motion_score = self._detect_motion(sequence)
                self.is_moving = motion_score > self.motion_threshold

                class_name, confidence = self._predict(sequence)

                with self.prediction_lock:
                    self.is_static_gesture = (
                        not self.is_moving and confidence >= self.static_confidence_threshold
                    )
                    if self.is_moving and confidence >= self.confidence_threshold:
                        self.current_prediction = class_name
                        self.current_confidence = confidence
                        self.displayed_prediction = class_name
                        self.displayed_confidence = confidence
                        self.last_prediction_time = time.time()
                    elif self.is_static_gesture:
                        self.current_prediction = class_name
                        self.current_confidence = confidence
                        self.displayed_prediction = class_name
                        self.displayed_confidence = confidence
                        self.last_prediction_time = time.time()
                    else:
                        self.current_prediction = None
                        self.current_confidence = confidence

    # ------------------------------------------------------------------
    # Hand-count hysteresis (anti-flicker)
    # ------------------------------------------------------------------

    def _apply_hand_hysteresis(self, landmarks_list, handedness_list):
        """Drop hands that haven't been continuously detected for
        HAND_WARMUP_FRAMES frames. Counters are keyed by handedness label
        ('Left' / 'Right') so each slot warms up independently. A hand that
        was already accepted last frame stays accepted (so legit two-hand
        signs aren't penalised after the first stable frame)."""

        # See which labels are detected this frame.
        seen = {'Left': None, 'Right': None}
        if landmarks_list and handedness_list:
            for lm, h in zip(landmarks_list, handedness_list):
                if not h.classification:
                    continue
                label = h.classification[0].label
                if label in seen and seen[label] is None:
                    seen[label] = (lm, h)

        kept_lm = []
        kept_h = []
        for label in ('Left', 'Right'):
            if seen[label] is None:
                self._hand_warmup[label] = 0
                continue
            self._hand_warmup[label] += 1
            if self._hand_warmup[label] >= self.HAND_WARMUP_FRAMES:
                lm, h = seen[label]
                kept_lm.append(lm)
                kept_h.append(h)

        return (kept_lm or None), (kept_h or None)

    # ------------------------------------------------------------------
    # Motion detection on the trajectory channels (NOT the local handshape)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_motion(sequence: np.ndarray) -> float:
        if len(sequence) < 2 or sequence.shape[1] <= OFF_WRIST_VELOCITY:
            # Old layout: fall back to total mean-abs diff.
            diffs = np.diff(sequence, axis=0)
            return float(np.mean(np.abs(diffs)))
        # New layout: use the wrist-velocity channels directly. They already
        # live in body-anchor units and ignore handshape jitter.
        vel = sequence[:, OFF_WRIST_VELOCITY:OFF_WRIST_VELOCITY + 6]
        return float(np.mean(np.abs(vel)))

    def _predict(self, sequence: np.ndarray):
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            return self.class_names[predicted_idx.item()], float(confidence.item())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        with self.landmarks_lock:
            hand_lms = self.latest_hand_landmarks
            face_lms = self.latest_face_landmarks
            pose_lms = self.latest_pose_landmarks

        if hand_lms:
            for hand_landmarks in hand_lms:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                )

        if self.use_face_mesh and face_lms:
            for face_landmarks in face_lms:
                for anchor_name, idx in self.face_anchors.items():
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        if 'eye' in anchor_name:
                            color = (0, 255, 255)
                        elif 'mouth' in anchor_name:
                            color = (0, 165, 255)
                        elif 'nose' in anchor_name:
                            color = (255, 0, 255)
                        elif 'eyebrow' in anchor_name:
                            color = (255, 255, 0)
                        else:
                            color = (0, 255, 0)
                        cv2.circle(frame, (x, y), 3, color, -1)
                        cv2.circle(frame, (x, y), 5, color, 1)

        if self.use_pose and pose_lms:
            # Draw shoulder points so you can see the body anchor.
            for idx in (11, 12, 23, 24):
                lm = pose_lms.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 6, (0, 200, 255), -1)
                cv2.circle(frame, (x, y), 9, (0, 200, 255), 2)

        # HUD background.
        cv2.rectangle(frame, (10, 10), (w - 10, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 160), (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        buffer_progress = len(self.frame_buffer) / self.sequence_length
        bar_width = int((w - 140) * buffer_progress)
        cv2.rectangle(frame, (20, 20), (w - 120, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 20), (20 + bar_width, 40), (0, 255, 0), -1)
        cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}",
                    (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if len(self.frame_buffer) == self.sequence_length:
            if self.is_moving:
                motion_color = (0, 255, 0)
                motion_text = "MOTION DETECTED"
            elif self.is_static_gesture:
                motion_color = (255, 165, 0)
                motion_text = "STATIC GESTURE"
            else:
                motion_color = (100, 100, 100)
                motion_text = "WAITING..."
            cv2.putText(frame, motion_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)

        with self.prediction_lock:
            prediction = self.current_prediction
            confidence = self.current_confidence
            displayed = self.displayed_prediction
            displayed_conf = self.displayed_confidence
            pred_time = self.last_prediction_time

        time_since = time.time() - pred_time
        show_displayed = displayed and time_since < self.display_duration
        remaining = max(0, self.display_duration - time_since)

        if prediction:
            cv2.putText(frame, f"Gesture: {prediction}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif show_displayed:
            cv2.putText(frame, f"Gesture: {displayed}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {displayed_conf:.1%} ({remaining:.1f}s)",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            if len(self.frame_buffer) == self.sequence_length:
                if not self.is_moving:
                    cv2.putText(frame, "Hold still = no prediction", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                else:
                    cv2.putText(frame, f"Low confidence: {confidence:.1%}", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "Collecting frames...", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "'q' quit | 'r' reset | '+/-' motion | 's/a' static threshold",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.ffe.reset()
        with self.prediction_lock:
            self.current_prediction = None
            self.current_confidence = 0.0
            self.displayed_prediction = None
            self.displayed_confidence = 0.0
            self.last_prediction_time = 0
        print("Buffer reset")

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
            # Update inference subsampling based on observed capture rate.
            self.capture_fps_estimate = self.fps if self.fps > 1 else self.capture_fps_estimate
            self._refresh_capture_skip()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, camera_id: int = 0):
        print(f"\nStarting webcam at 60fps (camera {camera_id})...")
        cap = VideoCapture(camera_id, width=1280, height=720, fps=60)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        self.process_thread.start()

        print("Webcam started!")
        print("Show your hand gestures to the camera")
        print(f"Inference at {self.target_fps:.0f} effective fps (every {self._capture_skip}th captured frame)\n")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            self._capture_frame_idx += 1
            if self._capture_frame_idx % self._capture_skip == 0:
                with self.processing_lock:
                    self.processing_frame = frame.copy()

            self.update_fps()
            frame = self.draw_ui(frame)
            cv2.imshow('Gesture Recognition - 60fps', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_buffer()
            elif key == ord('+') or key == ord('='):
                self.motion_threshold += 0.005
                print(f"Motion threshold: {self.motion_threshold:.3f}")
            elif key == ord('-'):
                self.motion_threshold = max(0.005, self.motion_threshold - 0.005)
                print(f"Motion threshold: {self.motion_threshold:.3f}")
            elif key == ord('s'):
                self.static_confidence_threshold = min(0.99, self.static_confidence_threshold + 0.05)
                print(f"Static confidence threshold: {self.static_confidence_threshold:.0%}")
            elif key == ord('a'):
                self.static_confidence_threshold = max(0.5, self.static_confidence_threshold - 0.05)
                print(f"Static confidence threshold: {self.static_confidence_threshold:.0%}")

        # Cleanup.
        self.processing_running = False
        self.process_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.pose is not None:
            self.pose.close()
        print("\nGesture recognition stopped")


def main():
    # Pull defaults from config.py if available.
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        import config as cfg
        sequence_length = cfg.SEQUENCE_LENGTH
        target_fps = getattr(cfg, 'TARGET_FPS', 30)
        confidence_threshold = cfg.PREDICTION_CONFIDENCE_THRESHOLD
        face_anchors = cfg.FACE_MESH_ANCHORS
        use_face_mesh = cfg.EXTRACT_FACE_MESH
        use_pose = cfg.EXTRACT_POSE
    except Exception:
        sequence_length = 45
        target_fps = 30
        confidence_threshold = 0.7
        face_anchors = DEFAULT_FACE_ANCHORS
        use_face_mesh = True
        use_pose = True

    MODEL_PATH = Path(__file__).parent.parent / "models" / "alphabet" / "best_model.pth"
    LABELS_PATH = Path(__file__).parent.parent / "data" / "alphabet_processed" / "class_labels.json"

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Train the model first: python training/train_model.py")
        return

    if not LABELS_PATH.exists():
        print(f"Error: Class labels not found at {LABELS_PATH}")
        print("Run prepare_augmented_dataset.py first")
        return

    print("=" * 60)
    print("REAL-TIME GESTURE RECOGNITION (v2)")
    print("Body-anchored trajectory + BiLSTM + attention")
    print("=" * 60)

    recognizer = RealtimeGestureRecognizer(
        model_path=str(MODEL_PATH),
        class_labels_path=str(LABELS_PATH),
        sequence_length=sequence_length,
        target_fps=target_fps,
        confidence_threshold=confidence_threshold,
        use_face_mesh=use_face_mesh,
        use_pose=use_pose,
        face_anchors=face_anchors,
    )

    recognizer.run(camera_id=0)


if __name__ == "__main__":
    main()
