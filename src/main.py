import cv2
import face_recognition
import os
import time
import numpy as np

# Load known faces
KNOWN_FACES_DIR = r'C:\Users\emran\desktop\dynamic-face-recognition\data\known_faces'
known_encodings = []
known_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            encoding = encodings[0]
            name = os.path.splitext(filename)[0]
            known_encodings.append(encoding)
            known_names.append(name)
            print(f"Loaded: {name}")

print(f"Total known faces: {len(known_names)}")

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting face recognition with enrollment. Unknown faces will be prompted after 4 seconds.")

unknown_start_time = None
unknown_encoding = None
last_enrolled_time = None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Scale back face locations
    face_locations = [(top*5, right*5, bottom*5, left*5) for (top, right, bottom, left) in face_locations]

    handled_unknown = False

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        greeting = ""
        color = (0, 0, 255)  # Default red

        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
                color = (0, 255, 0)  # Green
                unknown_start_time = None
                unknown_encoding = None
                greeting = f"Hello {name}"
            else:
                # Unknown face
                if not handled_unknown:
                    handled_unknown = True
                    if last_enrolled_time and time.time() - last_enrolled_time < 5:
                        continue
                    if unknown_encoding is None:
                        unknown_encoding = encoding
                        unknown_start_time = time.time()
                    else:
                        distance = face_recognition.face_distance([unknown_encoding], encoding)[0]
                        if distance < 0.6:
                            # Same face → continue timer
                            if time.time() - unknown_start_time >= 4:
                                # Prompt for name
                                name_input = input("Enter name for this unknown face: ")
                                if name_input:
                                    # Safe crop
                                    h, w, _ = frame.shape
                                    top_crop = max(0, top)
                                    left_crop = max(0, left)
                                    bottom_crop = min(h, bottom)
                                    right_crop = min(w, right)

                                    face_image = frame[top_crop:bottom_crop, left_crop:right_crop]

                                    image_path = os.path.join(KNOWN_FACES_DIR, f"{name_input}.jpg")
                                    cv2.imwrite(image_path, face_image)

                                    # Reload and encode properly
                                    saved_image = face_recognition.load_image_file(image_path)
                                    saved_encodings = face_recognition.face_encodings(saved_image)

                                    if len(saved_encodings) > 0:
                                        known_encodings.append(saved_encodings[0])
                                        known_names.append(name_input)
                                        print(f"Enrolled: {name_input}")
                                        last_enrolled_time = time.time()
                                    else:
                                        print("Failed to encode saved face.")

                                unknown_start_time = None
                                unknown_encoding = None
                        else:
                            # Different face → reset timer
                            unknown_encoding = encoding
                            unknown_start_time = time.time()

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # Draw name
        cv2.putText(frame, name, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # Draw greeting if known
        if greeting:
            cv2.putText(frame, greeting, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Program exited.")
