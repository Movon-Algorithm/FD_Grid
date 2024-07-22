import cv2
import onnxruntime
import numpy as np
import yaml
import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import *
from FaceBoxesV2.transforms import *
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(filename='face_detection_confidence_A.log', level=logging.INFO)

def preprocess(faceBoxesCfg_yaml):
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

def adjust_brightness_contrast(image):
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 50   # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

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
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        plotColor, 
                        2)
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
    
    scaling_factor = 0.75
    num_levels = 5
    min_size = (30, 30)
    display_size = (640, 480)  # 일정한 크기로 영상 표시

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected = False
            original_frame = frame.copy()  # 원본 크기 유지

            for level in range(num_levels):
                width = int(frame.shape[1] * (scaling_factor ** level))
                height = int(frame.shape[0] * (scaling_factor ** level))
                
                if width < min_size[0] or height < min_size[1]:
                    break
                
                resized_frame = cv2.resize(frame, (width, height))
                img_info = img_size(resized_frame)
                removePadOffset = RemovePadOffset(img_info, faceBoxesCfg_yaml['imageSize'])

                # Adjust brightness and contrast
                resized_frame = adjust_brightness_contrast(resized_frame)

                in_frame = resized_frame.copy()
                for proc in preprocs:
                    in_frame = proc(in_frame)
                
                faceDetections = faceDetector.detect(in_frame)
                if faceDetections.size > 0:
                    faceDetections = removePadOffset(faceDetections)
                    
                    original_frame = faceBoxWrite(img_size(frame), frame, faceDetections, confidences)
                    detected = True

                if detected:
                    break

            # 원본 크기의 영상을 일정한 크기로 조정하여 표시
            display_frame = cv2.resize(original_frame, display_size)
            cv2.imshow('Detected Faces', display_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate and log the average confidence
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        with open(log_filename, 'a') as f:
            f.write(f"\nAverage detection confidence: {avg_confidence:.2f}\n")
        logging.info(f"Average detection confidence: {avg_confidence:.2f}")

    return log_filename

folder_path = "C:\\Users\\movon\\Downloads\\FD_Grid_Video"
video_files = list(Path(folder_path).rglob("*.mp4"))

# Load configurations
faceBoxesCfg_yaml, priorCfg_yaml = load_config()
preprocs = preprocess(faceBoxesCfg_yaml)

# FaceBoxesV2 모델 초기화 (ONNX)
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
