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
