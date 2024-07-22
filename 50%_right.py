import onnxruntime
import numpy as np
import cv2
import yaml
import logging
import os
from pathlib import Path
import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import FaceBoxesONNXDetector
from FaceBoxesV2.transforms import ResizeImage, LetterBox, ConvertColor, ExpandBatchDim, toDevice, RemovePadOffset

# Set up logging
logging.basicConfig(filename='face_detection_confidence_A.log', level=logging.INFO)

def preprocess(faceBoxesCfg_yaml):
    return [
        ResizeImage(faceBoxesCfg_yaml['imageSize']),
        LetterBox(faceBoxesCfg_yaml['imageSize']),
        ConvertColor('GRAY1ch'),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485, std=0.229),
        ExpandBatchDim(),
        toDevice('cpu')
    ]

def img_size(img):
    img_height, img_width = img.shape[:2]
    return img_height, img_width

def faceBoxWrite(img_info, img, detections, confidences, plotColor=(0, 255, 0), lineThickness=2):
    height, width = img_info
    for detection in detections:
        bbox = detection[2:] * [width, height, width, height]
        bbox = bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), plotColor, lineThickness)
        cv2.putText(img, f"face: {detection[1]:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, plotColor, 2)
        confidences.append(detection[1])
        logging.info(f"Detection confidence: {detection[1]}")
    return img

def load_config():
    with open('./FaceBoxesV2/faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
        faceBoxesCfg_yaml = yaml.safe_load(file)
    with open('./FaceBoxesV2/priorCfg.yaml', 'r', encoding='utf-8') as file:
        priorCfg_yaml = yaml.safe_load(file)
    return faceBoxesCfg_yaml, priorCfg_yaml

def process_video(video_file, faceDetector, preprocs, faceBoxesCfg_yaml):
    cap = cv2.VideoCapture(video_file)
    confidences = []
    video_filename = os.path.basename(video_file)
    log_filename = f'face_detection_confidence_{video_filename}.txt'
    
    total_frames = 0
    detected_frames = 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    logging.info(f"Total frames: {frame_count}, FPS: {fps}, Duration: {duration}s")  # Log total frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame {total_frames + 1} could not be read.")  # Log frame read failure
            total_frames += 1  # Count failed read as a frame
            continue  # Skip this frame if read fails

        total_frames += 1
        img_info = img_size(frame)
        removePadOffset = RemovePadOffset(img_info, faceBoxesCfg_yaml['imageSize'])

        in_frame = frame.copy()
        for proc in preprocs:
            in_frame = proc(in_frame)
        
        faceDetections = faceDetector.detect(in_frame)
        if faceDetections.size > 0:
            detected_frames += 1
            faceDetections = removePadOffset(faceDetections)
            faceBoxWrite(img_info, frame, faceDetections, confidences)
            logging.info(f"Frame {total_frames}: Detected {len(faceDetections)} faces.")  # Log detection info
        else:
            logging.info(f"Frame {total_frames}: No faces detected.")  # Log no detection info

        # Display original frame with detections
        cv2.imshow('Detected Faces', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate and log the average confidence and detection rate
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        detection_rate = detected_frames / total_frames if total_frames > 0 else 0
        with open(log_filename, 'a') as f:
            f.write(f"\nAverage detection confidence: {avg_confidence:.2f}\n")
            f.write(f"Detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)\n")
        logging.info(f"Average detection confidence: {avg_confidence:.2f}")
        logging.info(f"Detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)")

    return log_filename

folder_path = "C:\\Users\\movon\\Downloads\\FD_Grid_Video"
video_files = list(Path(folder_path).rglob("*.mp4"))

# Load configurations
faceBoxesCfg_yaml, priorCfg_yaml = load_config()
preprocs = preprocess(faceBoxesCfg_yaml)

# Model init & Inference (ONNX)
model_path = 'mdfd.onnx'
onnxruntime.get_device()
sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
faceDetector = FaceBoxesONNXDetector(model_path, faceBoxesCfg_yaml, priorCfg_yaml, 'cpu')

log_files = []

# Process each video file
for video_file in video_files:
    log_file = process_video(str(video_file), faceDetector, preprocs, faceBoxesCfg_yaml)
    log_files.append(log_file)

# Open all log files at once
for log_file in log_files:
    os.system(f'notepad.exe {log_file}')
