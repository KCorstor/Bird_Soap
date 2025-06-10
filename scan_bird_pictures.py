import os
from ultralytics import YOLO
import cv2
from pathlib import Path

def detect_birds_in_folder(folder_path):
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Using the nano model
    
    # Get all image files in the folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.heic', '.HEIC')
    folder_path = "/Users/kevincorstorphine/Desktop/Bird_Pictures"  # Exact path
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images. Starting bird detection...")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"\nProcessing: {image_file}")
        
        try:
            # Run inference
            results = model(image_path)
            
            # Check for birds
            bird_detected = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Class 14 is 'bird' in COCO dataset
                    if box.cls == 14:
                        confidence = box.conf.item()
                        if confidence > 0.5:  # Only consider detections with >50% confidence
                            bird_detected = True
                            print(f"Bird detected! Confidence: {confidence:.2f}")
            
            if not bird_detected:
                print("No bird detected in this image")
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    print("\nBird detection complete!")

if __name__ == "__main__":
    # Path to your Bird_Pictures folder
    pictures_folder = os.path.expanduser("~/Desktop/Bird_Pictures")
    
    # Check if folder exists
    if not os.path.exists(pictures_folder):
        print(f"Folder not found: {pictures_folder}")
        print("Please create the folder or update the path in the script.")
    else:
        detect_birds_in_folder(pictures_folder) 