# Bird Soap 🐦

A real-time bird detection system that monitors a bird feeder using computer vision and AWS cloud services. This project demonstrates a complete pipeline from image capture to automated bird detection and notification.

## Overview

Bird Soap uses YOLOv8 (You Only Look Once) object detection to identify when birds are feeding at a bird feeder. The system captures images from a webcam, uploads them to AWS S3, and processes them using AWS Lambda for automated bird detection.

## Architecture

The system consists of three main components:

1. **Image Capture** (`bird_code.py`) - Captures frames from webcam and uploads to S3
2. **Bird Detection** (`bird_detector.py`) - Downloads images from S3 and detects birds
3. **AWS Lambda Processing** (`lambda_function.py`) - Serverless bird detection triggered by S3 uploads

## Features

- **Real-time Monitoring**: Continuous webcam feed with 1-second intervals
- **Cloud Storage**: Automatic upload to AWS S3 bucket
- **AI Detection**: YOLOv8 model for accurate bird identification
- **Serverless Processing**: AWS Lambda for scalable image processing
- **Batch Processing**: Offline analysis of existing bird photos
- **Confidence Scoring**: Only reports detections above 50% confidence

## Project Structure

```
Bird_Soap/
├── bird_code.py              # Webcam capture and S3 upload
├── bird_detector.py          # Real-time bird detection from S3
├── scan_bird_pictures.py     # Batch processing of local images
├── lambda_function.py        # AWS Lambda bird detection handler
├── requirements.txt          # Python dependencies
├── bird_dataset/            # Training data for custom model
├── bird_model/              # Trained YOLO model files
└── finetuned_bird_model.pt  # Custom fine-tuned model (WIP)
```

## Setup

### Prerequisites

- Python 3.9+
- AWS Account with S3 and Lambda access
- Webcam for real-time capture

### AWS Setup

1. Create S3 buckets:
   - `birdbucket111` - For incoming images
   - `bird-invocations-1` - For images with detected birds

2. Deploy Lambda function with the provided `lambda_function.py`

## Usage

### Real-time Monitoring
```bash
python bird_code.py          # Start webcam capture
python bird_detector.py      # Monitor for birds in uploaded images
```

### Batch Processing
```bash
python scan_bird_pictures.py  # Process existing photos in Bird_Pictures folder
```

## Technical Details

- **Model**: YOLOv8n (nano) for fast inference
- **Detection Class**: COCO dataset class 14 (bird)
- **Confidence Threshold**: 50% minimum for valid detections
- **Image Formats**: Supports JPG, PNG, BMP, HEIC
- **Cloud Storage**: AWS S3 with automatic cleanup

## Custom Model Training

The project includes a (work in progress) custom fine-tuned model trained on bird feeder images:
- Training data in `bird_dataset/`
- Model weights in `bird_model/weights/`
- Training notebook: `yolo_finetuning.ipynb`

## Future Enhancements

This project serves as a foundation for enterprise-scale deployment with potential for:
- Real-time streaming with AWS Kinesis
- Database logging of detection events
- Mobile app notifications
- Multi-camera support
- Species identification beyond general bird detection

