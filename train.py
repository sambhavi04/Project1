from ultralytics import YOLO

# Path to your data.yaml file
data_path = r'/home2/santosh/projects/mtech_research/paddy_classification/yolov8_dataset/data.yaml'

# Initialize a new YOLO model
model = YOLO('yolov8n.pt')  # Starting with a pre-trained YOLOv8 model

# Train the model on your dataset
model.train(data=data_path, epochs=20, imgsz=640, device=0)
