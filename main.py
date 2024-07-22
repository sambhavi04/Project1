import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO

def classify_and_display_image(model, image_path, class_names):
    """
    Classify and display the image using YOLOv8 model.

    Parameters:
    model (YOLO): Pre-trained YOLOv8 model.
    image_path (str): Path to the image file.
    class_names (list): List of class names for the YOLO model.
    """
    # Load the image using PIL
    img = Image.open(image_path).convert('RGB')
    
    # Convert PIL image to tensor
    img_tensor = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
    
    # Predict using the model
    results = model(img_tensor)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')

    if results[0].probs is not None:
        # Debug: Print the probabilities and top classes
        print(f"Class probabilities: {results[0].probs}")
        print(f"Top class indices: {results[0].probs.top5}")
        print(f"Top class confidences: {results[0].probs.top5conf}")
        
        top_class_index = results[0].probs.top1
        top_class_confidence = results[0].probs.top1conf.item()
        
        print(f"Predicted top class index: {top_class_index}")
        
        if 0 <= top_class_index < len(class_names):
            top_class_name = class_names[top_class_index]
            plt.title(f"{top_class_name} ({top_class_confidence:.2f})", fontsize=16, color='red')
        else:
            print(f"Warning: Predicted class index {top_class_index} is out of range.")
            plt.title(f"Unknown class ({top_class_confidence:.2f})", fontsize=16, color='red')
    else:
        print("No class probabilities found in the results.")
        plt.title(f"Unknown class", fontsize=16, color='red')

    plt.show()

def plot_random_image_from_each_class(folder_path, model, class_names):
    """
    Plot a random image from each class folder and classify it using YOLOv8 model.

    Parameters:
    folder_path (str): Path to the folder containing images.
    model (YOLO): Pre-trained YOLOv8 model.
    class_names (list): List of class names for the YOLO model.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided folder path does not exist: {folder_path}")

    class_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    if not class_dirs:
        raise ValueError("No class directories found in the dataset.")

    class_dir = random.choice(class_dirs)
    class_path = os.path.join(folder_path, class_dir)
    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

    if image_files:
        image_file = random.choice(image_files)
        image_path = os.path.join(class_path, image_file)
        classify_and_display_image(model, image_path, class_names)
    else:
        print(f"No images found in the class directory: {class_dir}")

if __name__ == "__main__":
    dataset_folder = r"/home2/santosh/projects/mtech_research/paddy_classification/yolov8_dataset/train_images"
    model_path = r"/home2/santosh/projects/mtech_research/paddy_classification/yolov8_dataset/yolov8n-cls.pt"
    class_names = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

    model = YOLO(model_path)
    plot_random_image_from_each_class(dataset_folder, model, class_names)
