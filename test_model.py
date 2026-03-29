from ultralytics import YOLO
import cv2

# Load the plate detection model
model = YOLO("models/plate_detector.pt")

# Read the test image
img = cv2.imread("test.jpg")

# Check if image loaded properly
if img is None:
    print("Error: test.jpg not found or could not be opened")
    exit()

# Run detection
results = model(img, conf=0.25)

# Draw boxes on detected plates
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save result image
cv2.imwrite("output.jpg", img)

print("Done. Check output.jpg")