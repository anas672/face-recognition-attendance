import cv2
import face_recognition
import numpy as np
import os
import time
import csv
from datetime import datetime

from utils import generate_variations, is_live_face

# ================= PATHS =================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "backend", "attendance.csv")

# ================= CONFIG =================

RECOGNITION_THRESHOLD = 0.48
FRAME_SKIP = 2
SCALE = 0.25

# ================= CSV INIT =================

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status"])

# ================= LOAD FACES =================

known_encodings = []
known_names = []

print("[INFO] Loading known faces...")

for file in os.listdir(IMAGES_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(file)[0]
        path = os.path.join(IMAGES_DIR, file)

        image = face_recognition.load_image_file(path)
        variations = generate_variations(image)

        count = 0
        for img in variations:
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(name)
                count += 1

        print(f"[INFO] {name}: {count} encodings")

known_encodings = np.array(known_encodings)
print(f"[INFO] Total encodings: {len(known_encodings)}")

# ================= ATTENDANCE MEMORY =================

marked_today = set()
prev_faces = {}

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{name}_{today}"

    if key in marked_today:
        return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            today,
            datetime.now().strftime("%H:%M:%S"),
            "Present"
        ])

    marked_today.add(key)
    print(f"[ATTENDANCE] {name} marked PRESENT")

# ================= CAMERA =================

cap = cv2.VideoCapture(0)
print("[INFO] Camera started. Press Q to quit.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    scale = int(1 / SCALE)
    locations = [(t*scale, r*scale, b*scale, l*scale) for t, r, b, l in locations]

    for (top, right, bottom, left), encoding in zip(locations, encodings):

        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            continue

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_id = f"{top}_{right}_{bottom}_{left}"

        live, prev_faces[face_id] = is_live_face(gray, prev_faces.get(face_id))

        if not live:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "FAKE", (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        distances = face_recognition.face_distance(known_encodings, encoding)
        best = np.argmin(distances)

        name = "Unknown"
        if distances[best] < RECOGNITION_THRESHOLD:
            name = known_names[best]
            mark_attendance(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} (LIVE)", (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    fps = frame_count / max(1, (time.time() - start_time))
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
