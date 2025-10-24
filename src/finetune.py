# finetune_ph_model.py
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Path to your Roboflow-trained model's weights
    roboflow_trained_weights = 'Roboflow_Vehicle_Training/yolov8m_roboflow_vehicles_run13/weights/best.pt' # Path to the .pt file from Part 1

    # Path to your custom Philippine dataset's YAML file
    custom_dataset_yaml = 'Vehicle-Counting-2/data.yaml' # Adjust this path

    # Check if files exist
    if not os.path.exists(roboflow_trained_weights):
        print(f"Error: Roboflow-trained weights not found at {roboflow_trained_weights}")
        exit()
    if not os.path.exists(custom_dataset_yaml):
        print(f"Error: Custom dataset YAML file not found at {custom_dataset_yaml}")
        exit()

    # Load the YOLOv8 model, initializing with your Roboflow-trained weights
    model = YOLO(roboflow_trained_weights)

    # Fine-tuning parameters
    project_name = 'PH_Vehicle_Finetuning'
    experiment_name = 'yolov8m_ph_finetune_run1'
    epochs = 100  # You might need more or fewer epochs for fine-tuning
    batch_size = 4 # Adjust based on VRAM and new dataset characteristics
    image_size = 640
    # Consider a smaller learning rate for fine-tuning initially
    learning_rate = 0.0005 # Example: lr0=0.0005

    print(f"Starting fine-tuning with weights: {roboflow_trained_weights}")
    print(f"Using custom dataset: {custom_dataset_yaml}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {image_size}, LR: {learning_rate}")

    # Fine-tune the model
    results = model.train(
        data=custom_dataset_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=project_name,
        name=experiment_name,
        patience=15, # Early stopping patience
        lr0=learning_rate # Set the initial learning rate
        # You can also freeze some initial layers if you want, but often fine-tuning all layers works well.
        # freeze=10 # Example: freeze first 10 layers (requires knowing model structure)
    )

    print(f"Fine-tuning complete. Results saved in: {results.save_dir}")
    print(f"Best fine-tuned model weights: {results.save_dir}/weights/best.pt")