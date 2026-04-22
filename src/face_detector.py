import time
import cv2
import face_recognition

try:
    from picamera2 import Picamera2
except ImportError:
    raise SystemExit(
        "picamera2 is not installed. On Raspberry Pi OS, run:\n"
        "  sudo apt update\n"
        "  sudo apt install -y python3-picamera2\n"
    )


def detect_faces(rgb_frame):
    """Detect faces in an RGB frame.
    
    Args:
        rgb_frame: Frame in RGB format (not BGR)
    
    Returns:
        List of face locations (top, right, bottom, left)
    """
    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations


if __name__ == "__main__":
    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # let auto-exposure settle

    print("Raspberry Pi Camera opened successfully. Starting face detection...")

    frame_count = 0
    fail_count = 0

    try:
        while True:
            frame_count += 1

            try:
                frame = picam2.capture_array()  # RGB array
            except Exception as e:
                fail_count += 1
                print(f"Frame {frame_count} FAILED: {e}")
                continue

            if frame is None or frame.size == 0:
                fail_count += 1
                print(f"Frame {frame_count} FAILED: empty frame")
                continue

            # Frame is already in RGB from picamera2
            rgb_frame = frame

            # Detect faces
            face_locations = detect_faces(rgb_frame)

            # Draw bounding boxes
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Convert RGB to BGR for display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Face Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"Program exited. Total frames: {frame_count}, Failed frames: {fail_count}.")
