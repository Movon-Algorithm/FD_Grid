import onnxruntime
import numpy as np
import cv2
import yaml
import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import *
from FaceBoxesV2.transforms import *
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(filename='face_detection_confidence_A.log', level=logging.INFO)

def preprocess():
    preprocs_list = [ResizeImage(faceBoxesCfg_yaml['imageSize']),
                     LetterBox(faceBoxesCfg_yaml['imageSize'])]
    preprocs_list.append(ConvertColor('GRAY1ch'))
    preprocs_list.append(transforms.ToTensor())
    preprocs_list.append(transforms.Normalize(mean=0.485, std=0.229))
    preprocs_list += [ExpandBatchDim(), toDevice('cpu')]
    return preprocs_list

def img_size(img):
    img_height, img_width = img.shape[:2]
    return img_height, img_width

def faceBoxWrite(img_info, img, detections, confidences, plotColor=(0, 255, 0), lineThickness=2):
    height, width = img_info
    for detection in detections:
        bbox = detection[2:]
        if bbox.ndim == 1:
            bbox[0] = int(bbox[0] * width)
            bbox[1] = int(bbox[1] * height)
            bbox[2] = int(bbox[2] * width)
            bbox[3] = int(bbox[3] * height)

            bbox = bbox.astype(int)
            cv2.rectangle(img, 
                          (bbox[0], bbox[1]), 
                          (bbox[2], bbox[3]), 
                          plotColor, lineThickness)
            cv2.putText(img, 
                        "face" + " : {0:.2f}".format(detection[1]), 
                        (bbox[0], bbox[1]), 
                        cv2.FONT_ITALIC, 
                        0.5, 
                        plotColor)
            confidences.append(detection[1])
            logging.info(f"Detection confidence: {detection[1]}")
    return img

def load_config():
    with open('.\\FaceBoxesV2\\faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
        faceBoxesCfg_yaml = yaml.safe_load(file)
    with open('.\\FaceBoxesV2\\priorCfg.yaml', 'r', encoding='utf-8') as file:
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
    
    if cap.isOpened():
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

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

preprocs = preprocess()

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