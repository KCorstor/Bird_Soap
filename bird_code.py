import cv2
import boto3
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials from environment variables
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-west-1')  # Default to us-west-1 if not specified

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize S3 client with credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

bucket = "birdbucket111"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Create filename with timestamp
        filename = f"frame-{datetime.utcnow().isoformat()}.jpg"
        
        # Save frame locally
        cv2.imwrite(filename, frame)
        
        try:
            # Upload to S3
            s3.upload_file(filename, bucket, filename)
            print(f"Successfully uploaded {filename}")
            
            # Clean up local file
            os.remove(filename)
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
        
        # Wait for 1 second before next capture
        time.sleep(1.0)
            
except KeyboardInterrupt:
    print("\nStopping video capture...")
finally:
    cap.release()
    cv2.destroyAllWindows() 