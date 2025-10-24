from ultralytics import YOLO

# Load your trained model
model = YOLO('PH_Vehicle_Finetuning/yolov8m_ph_finetune_run14/weights/best.pt')  # Path to your best.pt

# Evaluate on test data
metrics = model.val(
    data='dataset.yaml',  # Path to your dataset YAML
    split='test',         # Or 'val' if no test split
    imgsz=640,            # Same as training size
    conf=0.5,            # Confidence threshold
    iou=0.5,             # IoU threshold for mAP@0.5
    device='cpu',        # 'cpu' or 'cuda'
)
print(metrics.box.map)    # mAP@0.5
print(metrics.box.map50)  # mAP@0.5:0.95