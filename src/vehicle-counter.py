from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your trained model
model = YOLO('PH_Vehicle_Finetuning/yolov8m_ph_finetune_run12/weights/best.pt')

# Load image
image = cv2.imread("demo5.png")
height, width = image.shape[:2]

# Calculate font scale relative to image size
reference_dimension = 1000  
scale_factor = max(height, width) / reference_dimension
font_scale = 0.7 * scale_factor  
thickness = max(1, int(2 * scale_factor)) 

# Run detection
results = model.predict(image, conf=0.5, classes=[0, 1, 2, 3, 4, 5, 6, 7])

# Process detections
total_vehicles = 0
for box in results[0].boxes:
    total_vehicles += 1
    # Draw bounding box without label
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Draw total count
text = f"Total Vehicles: {total_vehicles}"
(text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
cv2.putText(image, text, (10, 30 + text_height), 
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

# Save and display
cv2.imwrite("output_image.jpg", image)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()