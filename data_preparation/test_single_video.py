"""
Quick test script to process a single video and visualize the extraction.
Use this to verify your setup before processing the entire dataset.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


def test_video_processing(video_path: str, show_visualization: bool = True):
    """
    Test MediaPipe on a single video file.

    Args:
        video_path: Path to video file
        show_visualization: Show real-time visualization
    """
    # Try to load config values
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        import config
        min_detection = config.MIN_DETECTION_CONFIDENCE
        min_tracking = config.MIN_TRACKING_CONFIDENCE
    except:
        min_detection = 0.5
        min_tracking = 0.5

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking
    )

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print(f"Video: {Path(video_path).name}")
    print(f"{'='*60}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")

    # Statistics
    frames_processed = 0
    frames_with_hands = 0
    total_hands_detected = 0

    # Process video
    print(f"\nProcessing... (Press 'Q' to quit)")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frames_processed += 1

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(frame_rgb)

        # Extract landmarks
        if results.multi_hand_landmarks:
            frames_with_hands += 1
            num_hands = len(results.multi_hand_landmarks)
            total_hands_detected += num_hands

            if show_visualization:
                # Draw landmarks
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # Get handedness (left/right)
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    score = results.multi_handedness[hand_idx].classification[0].score

                    # Display handedness
                    h, w, _ = frame.shape
                    wrist = hand_landmarks.landmark[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)

                    cv2.putText(
                        frame,
                        f"{handedness} ({score*100:.0f}%)",
                        (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        if show_visualization:
            # Display info on frame
            cv2.putText(
                frame,
                f"Frame: {frames_processed}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if results.multi_hand_landmarks:
                cv2.putText(
                    frame,
                    f"Hands detected: {num_hands}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No hands detected",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Show frame
            cv2.imshow('Hand Detection Test', frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break

    # Cleanup
    cap.release()
    if show_visualization:
        cv2.destroyAllWindows()

    hands.close()

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Frames processed: {frames_processed}/{total_frames}")
    print(f"Frames with hands detected: {frames_with_hands} ({frames_with_hands/frames_processed*100:.1f}%)")
    print(f"Average hands per frame: {total_hands_detected/frames_processed:.2f}")

    if frames_with_hands / frames_processed < 0.5:
        print(f"\n[WARNING] Hands detected in less than 50% of frames")
        print(f"   Consider:")
        print(f"   - Improving lighting")
        print(f"   - Ensuring hands are clearly visible")
        print(f"   - Lowering detection confidence threshold")
    else:
        print(f"\n[OK] Good detection rate!")

    print(f"\nEstimated sequences from this video:")

    sequence_length = 30  # Default from extract_gesture_data.py
    overlap = sequence_length // 2
    num_sequences = max(0, (frames_with_hands - sequence_length) // overlap + 1)
    print(f"  With sequence_length={sequence_length}: ~{num_sequences} sequences")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_single_video.py <path_to_video> [--no-viz]")
        print("\nExample:")
        print("  python test_single_video.py training_videos/thumbs_up/video1.mp4")
        print("  python test_single_video.py training_videos/wave/video1.mp4 --no-viz")
        return

    video_path = sys.argv[1]
    show_viz = '--no-viz' not in sys.argv

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return

    test_video_processing(video_path, show_visualization=show_viz)


if __name__ == "__main__":
    main()
