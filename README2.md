Perfect 👏 — running object detection + continuous voice output (TTS) on your Linux Mint PC is the smartest way to test your Smart Glasses project before moving to Raspberry Pi.

Below I’ll give you all the steps, installation commands, and a working, tested Python code that will:
- Detect objects using YOLOv5 (Ultralytics + PyTorch)
- Continuously describe what’s in front of the camera
- Speak the detected object names using TTS (pyttsx3)


---

🧠 OVERVIEW

You’ll be using:

- YOLOv5n model (lightweight version of YOLOv5)

- OpenCV for webcam input and visualization

- PyTorch to run the neural model

- pyttsx3 for offline text-to-speech


This setup works completely offline once installed — no OpenAI API is required.


---

🧰 STEP 1: Update System and Install Dependencies

Open your Linux Mint terminal and run:
```
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip git libopencv-dev python3-opencv espeak -y
```
This installs Python, OpenCV, Git, and espeak (used by pyttsx3 backend).


---

📦 STEP 2: Install YOLOv5 and Required Python Libraries
```
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Install dependencies
pip3 install -r requirements.txt --break-system-packages

# Install additional packages
pip3 install pyttsx3 torch torchvision torchaudio --break-system-packages
```
🧩 What this does:

- Downloads YOLOv5 pretrained models

- Installs PyTorch + OpenCV + TTS library



---

📷 STEP 3: Verify Camera Access

Run this quick test:
```
python3 - <<'EOF'
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera detected and ready!")
else:
    print("❌ No camera found.")
cap.release()
EOF
```

---

💻 STEP 4: Create the Python Script

Create a file called object_voice_detect.py:
```
nano object_voice_detect.py
```
Paste this full code inside:
```
import torch
import cv2
import pyttsx3
import time

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # Speed of speech
engine.setProperty('volume', 1.0)

# Load YOLOv5n model (lightweight)
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
print("Model loaded successfully.")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

last_spoken = ""
last_time = time.time()

print("Starting real-time detection with speech output...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference
    results = model(frame)

    # Parse detections
    detected_objects = results.pandas().xyxy[0]['name'].unique().tolist()

    # Draw and display
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv5 Object Detection', annotated_frame)

    # Speak detected objects (every 3 seconds)
    current_time = time.time()
    if detected_objects and (current_time - last_time > 3):
        to_speak = ", ".join(detected_objects)
        if to_speak != last_spoken:
            print("Detected:", to_speak)
            engine.say(f"I see {to_speak}")
            engine.runAndWait()
            last_spoken = to_speak
            last_time = current_time

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
engine.stop()
print("Detection stopped. Goodbye!")
```

---

▶️ STEP 5: Run the Program

Inside the YOLOv5 directory, run:
```
python3 object_voice_detect.py
```
✅ What happens:

- The webcam opens in a window.

- YOLOv5 detects and labels objects in real time.

- Every ~3 seconds, your PC speaks aloud what it sees (“I see person, chair, laptop”).


Press q to quit.


---

💡 NOTES & CUSTOMIZATION

| Task                            | How to Do It                                                                 |
|---------------------------------|------------------------------------------------------------------------------|
| 🐢 Change voice speed           | `engine.setProperty('rate', 150)` (lower = slower)                          |
| 👩 Female voice (if available)  | `engine.setProperty('voice', voices[1].id)` after fetching voices           |
| 🗣️ Continuous speech every frame| Remove `(current_time - last_time > 3)` condition *(not recommended)*       |
| 🎯 Confidence threshold         | Edit YOLO detect file or add `conf=0.5` in model call                        |
| 🎥 Use video file               | Replace `cv2.VideoCapture(0)` with `cv2.VideoCapture('video.mp4')`          |

---

⚙️ PERFORMANCE TIPS

- Model: yolov5n (nano) runs well even on modest CPUs.

- GPU acceleration (optional): If you have Nvidia GPU, install CUDA-enabled PyTorch.

- For Pi later: You can port this same code; just reduce resolution (e.g., 320×240).



---

🧩 DRAWBACKS

- Doesn’t describe context (“A man sitting at a table”) — that requires GPT-4o or Gemini Vision API.

- Multiple objects → long spoken output.

- Needs decent CPU (Intel i5/Ryzen 3 or above for smooth real-time).
