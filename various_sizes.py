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
    preprocs_list = [
        ResizeImage(faceBoxesCfg_yaml['imageSize']),
        LetterBox(faceBoxesCfg_yaml['imageSize']),
        ConvertColor('GRAY1ch'),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485, std=0.229),
        ExpandBatchDim(),
        toDevice('cpu')
    ]
    return preprocs_list

def img_size(img):
    img_height, img_width = img.shape[:2]
    return img_height, img_width

def faceBoxWrite(img_info, img, detections, confidences, plotColor=(0, 255, 0), lineThickness=2):
    height, width = img_info
    for detection in detections:
        bbox = detection[2:]
        bbox[0] = int(bbox[0] * width)
        bbox[1] = int(bbox[1] * height)
        bbox[2] = int(bbox[2] * width)
        bbox[3] = int(bbox[3] * height)

        bbox = bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), plotColor, lineThickness)
        cv2.putText(img, "face" + " : {0:.2f}".format(detection[1]), (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, plotColor, 2)
        confidences.append(detection[1])
        logging.info(f"Detection confidence: {detection[1]}")
    return img

def load_config():
    try:
        with open('./FaceBoxesV2/faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
            faceBoxesCfg_yaml = yaml.safe_load(file)
        with open('./FaceBoxesV2/priorCfg.yaml', 'r', encoding='utf-8') as file:
            priorCfg_yaml = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading configuration files: {e}")
        raise
    return faceBoxesCfg_yaml, priorCfg_yaml

def process_video(video_file, faceDetector, preprocs, faceBoxesCfg_yaml):
    cap = cv2.VideoCapture(video_file)
    confidences = []
    video_filename = os.path.basename(video_file)
    log_filename = f'face_detection_confidence_{video_filename}.txt'
    
    total_frames = 0
    detected_frames = 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing video: {video_filename}, Total frames: {frame_count}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame {total_frames + 1} could not be read.")
            continue  # Skip this frame if read fails

        total_frames += 1
        img_info = img_size(frame)
        removePadOffset = RemovePadOffset(img_info, faceBoxesCfg_yaml['imageSize'])

        # Improved pre-processing
        in_frame = frame.copy()
        for proc in preprocs:
            in_frame = proc(in_frame)
        
        # Adjust detection parameters (e.g., confidence threshold, NMS)
        faceDetections = faceDetector.detect(in_frame)

        if faceDetections.size > 0:
            detected_frames += 1
            faceDetections = removePadOffset(faceDetections)

            # Filter out low-confidence detections
            faceDetections = [det for det in faceDetections if det[1] > 0.5]  # Confidence threshold

            if len(faceDetections) > 0:
                faceBoxWrite(img_info, frame, faceDetections, confidences)
                logging.info(f"Frame {total_frames}: Detected {len(faceDetections)} faces.")
            else:
                logging.info(f"Frame {total_frames}: No faces detected after filtering.")
        else:
            logging.info(f"Frame {total_frames}: No faces detected.")

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
        try:
            with open(log_filename, 'a') as f:
                f.write(f"\nAverage detection confidence: {avg_confidence:.2f}\n")
                f.write(f"Detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)\n")
            logging.info(f"Average detection confidence: {avg_confidence:.2f}")
            logging.info(f"Detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)")
        except Exception as e:
            logging.error(f"Error writing to log file: {e}")

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
