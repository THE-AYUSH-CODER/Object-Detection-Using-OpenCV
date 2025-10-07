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
