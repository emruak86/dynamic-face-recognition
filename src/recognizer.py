import cv2
import face_recognition
import os
import numpy as np

# Path to known faces
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'known_faces')

def load_known_faces():
    """Load known faces from the data/known_faces directory."""
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Known faces directory not found: {KNOWN_FACES_DIR}")
        return known_encodings, known_names

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

    print(f"Total faces loaded: {len(known_names)}")
    return known_encodings, known_names

def recognize_faces():
    """Main face recognition function following the pipeline:
    detect face -> encode face -> compare with known encodings -> return name or Unknown
    """
    # Load known faces
    known_encodings, known_names = load_known_faces()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Starting face recognition...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)

        # Encode faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Compare with known encodings
            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                # Find the best match
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.6:  # Adjust tolerance as needed
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display result above head
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")

if __name__ == "__main__":
    recognize_faces()
