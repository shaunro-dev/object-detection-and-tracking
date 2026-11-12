AI-Powered Object Detection & Tracking System

A real-time computer vision web application built with Python, YOLOv8 (Ultralytics), OpenCV, and a responsive HTML/CSS/JavaScript dashboard. The system detects real-world objects, assigns persistent tracking IDs across consecutive frames, and supports custom model training.


Features

- Real-Time Object Detection & Tracking: Locates objects and maintains persistent track IDs across video frames using YOLO models.
- Interactive Web Dashboard: Clean, browser-based interface (`static/index.html`) for live camera feeds, telemetry monitoring, and detection controls.
- Custom Model Training Pipeline: Includes `train_daily.py` to fine-tune and update custom detection weights (`custom_models/`).
- Local Data Persistence : Uses SQLite (`robot_data.db`) to log detection events and session telemetry without requiring external database servers.


Project Structure

object-detection-and-tracking/
├── robot-detection-app/
│   ├── static/
│   │   ├── index.html       # Frontend dashboard UI
│   │   ├── script.js        # Client-side streaming & API logic
│   │   └── style.css        # Dashboard styling
│   ├── app.py               # Main backend server & vision inference pipeline
│   ├── train_daily.py       # Script for training/updating custom YOLO weights
│   └── requirements.txt     # Python dependencies
└── .gitignore
