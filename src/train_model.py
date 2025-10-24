# train_roboflow_model.py
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Path to the data.yaml file from the downloaded Roboflow dataset
    # This path might look like: 'Vehicles-Detection-3/data.yaml' or similar
    # Adjust it based on where Roboflow downloaded the dataset.
    roboflow_dataset_yaml = os.path.join('miniproject4_1-11', 'data.yaml')

    # Check if the YAML file exists
    if not os.path.exists(roboflow_dataset_yaml):
        print(f"Error: Dataset YAML file not found at {roboflow_dataset_yaml}")
        print("Please ensure you've downloaded the Roboflow dataset and the path is correct.")
        exit()

    # Load a YOLOv8 model (e.g., yolov8n.pt for faster training, yolov8m.pt for better accuracy)
    # Using '.pt' loads pre-trained weights from COCO.
    model = YOLO('yolov8n.pt')

    # Training parameters
    project_name = 'Roboflow_Vehicle_Training'
    experiment_name = 'yolov8m_roboflow_vehicles_run1'
    epochs = 30  # Start with a moderate number (e.g., 25-50)
    batch_size = 4 # Adjust based on your GPU VRAM
    image_size = 640

    print(f"Starting training on Roboflow dataset: {roboflow_dataset_yaml}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {image_size}")

    # Train the model
    results = model.train(
        data=roboflow_dataset_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=project_name,
        name=experiment_name,
        patience=10 # Early stopping patience
    )

    print(f"Training complete. Results saved in: {results.save_dir}")
    print(f"Best model weights: {results.save_dir}/weights/best.pt")
    # This 'best.pt' is your "Roboflow-trained model"