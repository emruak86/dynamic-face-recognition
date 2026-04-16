import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")

frame_count = 0
fail_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"[SUMMARY] Frames: {frame_count}, Failures: {fail_count}")

    if not ret:
        fail_count += 1
        print(f"Frame {frame_count} FAILED.")
        print("Warning: Could not read frame. Retrying...")
        continue
    else:
        print(f"Frame {frame_count} read successfully.")
        frame_count += 1

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Program exited successfully. Total frames: {frame_count}, Failed frames: {fail_count}.")
