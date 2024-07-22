import cv2
import onnxruntime
import numpy as np
import yaml
import logging
import os
from pathlib import Path
from FaceBoxesV2.faceBoxesV2_detector_onnx import FaceBoxesONNXDetector
from FaceBoxesV2.transforms import ResizeImage, LetterBox, ConvertColor, ExpandBatchDim, toDevice, RemovePadOffset
import torchvision.transforms as transforms

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

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def faceBoxWrite(img, detections, confidences, plotColor=(0, 255, 0), lineThickness=2):
    height, width = img.shape[:2]
    for detection in detections:
        bbox = detection[2:] * [width, height, width, height]
        bbox = bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), plotColor, lineThickness)
        cv2.putText(img, f"face: {detection[1]:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, plotColor, 2)
        confidences.append(detection[1])
        logging.info(f"Detection confidence: {detection[1]}")
    return img

def load_config():
    with open('FaceBoxesV2/faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
        faceBoxesCfg_yaml = yaml.safe_load(file)
    with open('FaceBoxesV2/priorCfg.yaml', 'r', encoding='utf-8') as file:
        priorCfg_yaml = yaml.safe_load(file)
    return faceBoxesCfg_yaml, priorCfg_yaml

def process_video(video_file, faceDetector, preprocs, faceBoxesCfg_yaml):
    cap = cv2.VideoCapture(video_file)
    confidences = []
    total_frames = 0
    detected_frames = 0
    log_filename = f'face_detection_confidence_{os.path.basename(video_file)}.txt'
    display_size = (640, 480)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            original_frame = frame.copy()
            detected = False
            for level in range(5):
                scale = 0.75 ** level
                resized_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
                if resized_frame.shape[0] < 30 or resized_frame.shape[1] < 30:
                    break

                for proc in preprocs:
                    resized_frame = proc(resized_frame)

                faceDetections = faceDetector.detect(resized_frame)
                if faceDetections.size > 0:
                    removePadOffset = RemovePadOffset(resized_frame.shape[:2], faceBoxesCfg_yaml['imageSize'])
                    faceDetections = removePadOffset(faceDetections)
                    original_frame = faceBoxWrite(original_frame, faceDetections, confidences)
                    detected = True
                    break

            if detected:
                detected_frames += 1

            display_frame = cv2.resize(original_frame, display_size)
            cv2.imshow('Detected Faces', display_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        detection_rate = detected_frames / total_frames if total_frames > 0 else 0
        with open(log_filename, 'a') as f:
            f.write(f"\nAverage detection confidence: {avg_confidence:.2f}\n")
            f.write(f"Detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)\n")
        logging.info(f"Average detection confidence: {avg_confidence:.2f}")
        logging.info(f"Detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)")

    return log_filename

# Main Execution
folder_path = "C:\\Users\\movon\\Downloads\\FD_Grid_Video"
video_files = list(Path(folder_path).rglob("*.mp4"))

faceBoxesCfg_yaml, priorCfg_yaml = load_config()
preprocs = preprocess(faceBoxesCfg_yaml)
faceDetector = FaceBoxesONNXDetector('mdfd.onnx', faceBoxesCfg_yaml, priorCfg_yaml, 'cpu')

log_files = [process_video(str(video_file), faceDetector, preprocs, faceBoxesCfg_yaml) for video_file in video_files]

for log_file in log_files:
    os.system(f'notepad.exe {log_file}')