import cv2
import face_recognition
import os
import time

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

print("Starting enrollment. Unknown faces will be prompted after 4 seconds.")

unknown_start_time = None
unknown_encoding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_unknown = False

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            # Known face
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            color = (0, 255, 0)  # Green
            unknown_start_time = None  # Reset timer
        else:
            # Unknown face
            color = (0, 0, 255)  # Red
            current_unknown = True
            if unknown_start_time is None:
                unknown_start_time = time.time()
                unknown_encoding = encoding
            elif time.time() - unknown_start_time >= 4:
                # Prompt for name
                name_input = input("Enter name for this unknown face: ")
                if name_input:
                    # Save image
                    image_path = os.path.join(KNOWN_FACES_DIR, f"{name_input}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Enrolled: {name_input}")
                    # Reload known faces? For simplicity, just append
                    known_encodings.append(unknown_encoding)
                    known_names.append(name_input)
                unknown_start_time = None
                unknown_encoding = None

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if not current_unknown:
        unknown_start_time = None

    cv2.imshow("Enrollment", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Enrollment exited.")
