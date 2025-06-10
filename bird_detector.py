import boto3
import os
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Initialize AWS S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-west-1')
)

bucket = "birdbucket111"

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using the nano model, you can use larger models for better accuracy

def download_latest_image():
    """Download the most recent image from S3 bucket"""
    try:
        # List objects in bucket
        response = s3.list_objects_v2(Bucket=bucket)
        if 'Contents' not in response:
            print("No images found in bucket")
            return None
            
        # Get the most recent image
        latest_image = max(response['Contents'], key=lambda x: x['LastModified'])
        image_key = latest_image['Key']
        
        # Download the image
        local_path = f"temp_{image_key}"
        s3.download_file(bucket, image_key, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return None

def detect_bird(image_path):
    """Detect if there's a bird in the image"""
    try:
        # Run inference
        results = model(image_path)
        
        # Check if any birds are detected
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class 14 is 'bird' in COCO dataset
                if box.cls == 14:
                    return True, box.conf.item()
        
        return False, 0.0
    except Exception as e:
        print(f"Error detecting bird: {str(e)}")
        return False, 0.0

def main():
    print("Starting bird detection...")
    while True:
        try:
            # Download latest image
            image_path = download_latest_image()
            if image_path is None:
                continue
                
            # Detect bird
            is_bird, confidence = detect_bird(image_path)
            
            if is_bird:
                print(f"Bird detected! Confidence: {confidence:.2f}")
            else:
                print("No bird detected")
                
            # Clean up
            os.remove(image_path)
            
        except KeyboardInterrupt:
            print("\nStopping bird detection...")
            break
        except Exception as e:
            print(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main() 