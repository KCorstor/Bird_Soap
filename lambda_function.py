import json
import boto3
import os
import base64
from ultralytics import YOLO

def lambda_handler(event, context):
    # Get the S3 bucket and key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process image files
    if not key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        return {
            'statusCode': 200,
            'body': json.dumps('Not an image file, skipping processing')
        }
    
    try:
        # Initialize clients
        s3 = boto3.client('s3')
        sns = boto3.client('sns')
        
        # Get image directly from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        # Load YOLO model and run inference
        model = YOLO('yolov8n.pt')
        results = model(image_data)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 14:  # Class 14 is 'bird' in COCO dataset
                    confidence = float(box.conf.item())
                    if confidence > 0.5:  # Only consider detections with >50% confidence
                        detections.append({
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
        
        # Create result object
        result_data = {
            'image_key': key,
            'bird_detected': len(detections) > 0,
            'detections': detections,
            'timestamp': event['Records'][0]['eventTime']
        }
        
        # Save results back to S3
        result_key = f"results/{os.path.splitext(key)[0]}_results.json"
        s3.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=json.dumps(result_data),
            ContentType='application/json'
        )
        
        # If birds were detected
        if len(detections) > 0:
            # Copy the image to the bird-invocations-1 bucket
            s3.copy_object(
                Bucket='bird-invocations-1',
                CopySource={'Bucket': bucket, 'Key': key},
                Key=key
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing complete',
                'result_key': result_key,
                'bird_detected': len(detections) > 0,
                'num_detections': len(detections)
            })
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        } 