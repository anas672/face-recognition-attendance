# Face Recognition Attendance System 🎓

A Python-based face recognition attendance system using OpenCV and `face_recognition`.

## Features
- Real-time face recognition
- Liveness detection (anti-photo spoofing)
- Automatic attendance logging (CSV)
- Data augmentation for higher accuracy

## Tech Stack
- Python
- OpenCV
- face_recognition (dlib)
- NumPy

## Project Structure

backend/
 ├── video_detect.py
 ├── utils.py
 └── attendance.csv
images/
 └── person images
How to Run
bash
Copy code
python backend/video_detect.py
Notes
Close attendance.csv before running

Press Q to quit camera
