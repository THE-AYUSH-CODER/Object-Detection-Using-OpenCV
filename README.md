# Object-Detection-Using-OpenCV

🧠 Project: Smart Glasses Object Detection + Voice Description

Goal:
Detect objects continuously through a camera, display them in a window with bounding boxes, and speak out what’s detected.


---

⚙️ 1️⃣ Prerequisites

- Before starting:

- You’re booted into Linux Mint Cinnamon.

- You have an internet connection.

- You have a USB webcam connected and working.


You can verify your webcam:

```
ls /dev/video*
```
If you see /dev/video0, it’s connected correctly. ✅


---

⚙️ 2️⃣ Update system

```
sudo apt update && sudo apt upgrade -y
```

---

⚙️ 3️⃣ Install required packages

```
sudo apt install python3-pip git python3-opencv espeak -y
```

---

⚙️ 4️⃣ Install Python libraries

```
pip3 install torch torchvision torchaudio ultralytics pyttsx3 --break-system-packages
```

📦 Explanation:

- torch, torchvision, torchaudio → Machine learning backend for YOLO

- ultralytics → YOLOv8 library (latest version)

- pyttsx3 → Text-to-Speech

- opencv-python → Already installed from step 3



---

⚙️ 5️⃣ Verify your camera

Run this simple command to ensure your camera is recognized:

```
python3 - <<'EOF'
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found!")
else:
    print("✅ Camera working fine!")
cap.release()
EOF
```

If you get ✅ Camera working fine!, you’re ready.


---

⚙️ 6️⃣ Download the YOLO model (automatically done on first run)

We’ll use the YOLOv8 nano model (yolov8n.pt) — it’s lightweight but powerful.


---

⚙️ 7️⃣ Create the main Python file

Create a file named detect_and_speak.py:

```
nano detect_and_speak.py
```

Paste this complete code 👇


---

🐍 Full Python Code: detect_and_speak.py
```
import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # downloads automatically first run

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

print("✅ Camera started. Press 'Q' to quit.")

last_spoken = ""  # to avoid repeating same object names

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame!")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Get class names of detected objects
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detected.append(results[0].names[cls_id])

    # Draw boxes and labels on frame
    annotated_frame = results[0].plot()

    # Speak detected objects if any
    if detected:
        # Remove duplicates
        detected = list(set(detected))
        sentence = "I can see " + ", ".join(detected)

        if sentence != last_spoken:
            print(sentence)
            engine.say(sentence)
            engine.runAndWait()
            last_spoken = sentence

    # Show the annotated frame
    cv2.imshow("Smart Glasses Camera View", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
Save and exit (press Ctrl + O, then Enter, then Ctrl + X).


---

⚙️ 8️⃣ Run your program

In terminal:

```
python3 detect_and_speak.py
```

✅ You should now see:

- A window showing your camera feed

- Bounding boxes and labels over detected objects

- The system speaking aloud the object names (e.g. “I can see person, chair”)


Press Q to exit.


---

⚙️ 9️⃣ (Optional) Auto-start on boot (for future)

Later, when you move this to a Raspberry Pi, you can set it to run automatically using:

```
sudo nano /etc/rc.local
```

and add this line before exit 0:

```
python3 /home/pi/detect_and_speak.py &
```

---

⚙️ 🔟 Troubleshooting

Problem	Fix

No module named torch	Re-run pip3 install torch torchvision torchaudio --break-system-packages
cv2.imshow crashes	Reduce window resolution or run sudo apt install python3-opencv again
No audio	Check espeak works: run espeak "hello world"
Laggy video	Use smaller model (yolov8n.pt) and close other apps



---

🧠 1️⃣1️⃣ Notes & Drawbacks

Limitation	Explanation

🧮 CPU-only inference	YOLO runs slower (~3–5 FPS) without GPU
🔊 Repetitive speech	Code limits repeats with last_spoken variable
🧠 Limited accuracy	YOLOv8n detects 80 COCO objects (person, car, chair, etc.)
🔋 High CPU use	Constant video + TTS can use ~60–80% CPU on PC
⚡ Slow USB	Live USB storage can cause small lag



---

✅ In Summary

|Step|	Command|
----------------
||Update system|	sudo apt update && sudo apt upgrade -y
|Install base tools|	sudo apt install python3-pip git python3-opencv |espeak -y
|Install libraries|	pip3 install torch torchvision torchaudio |ultralytics pyttsx3 --break-system-packages
|Run code|	python3 detect_and_speak.py



---

🧩 Example Output

A camera window titled “Smart Glasses Camera View”

Green boxes with labels like:

```
person
bottle
chair
```

Voice output saying:
“I can see person and bottle.”

---

