import onnxruntime
import numpy as np
import cv2
import yaml

import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import *
from FaceBoxesV2.transforms import *

from pathlib import Path

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

def faceBoxWrite(img_info, img, detections, plotColor=(0, 255, 0), lineThickness=2):
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
    return img

def load_config():
    with open('.\\FaceBoxesV2\\faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
        faceBoxesCfg_yaml = yaml.safe_load(file)
    with open('.\\FaceBoxesV2\\priorCfg.yaml', 'r', encoding='utf-8') as file:
        priorCfg_yaml = yaml.safe_load(file)
    return faceBoxesCfg_yaml, priorCfg_yaml

def process_video(video_file, faceDetector, preprocs, faceBoxesCfg_yaml):
    cap = cv2.VideoCapture(video_file)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                img_info = img_size(frame)
                removePadOffset = RemovePadOffset(img_info, faceBoxesCfg_yaml['imageSize'])

                # 영상의 오른쪽 절반 선택
                right_half_frame = frame[:, frame.shape[1] // 2:]

                in_frame = right_half_frame.copy()
                for proc in preprocs:
                    in_frame = proc(in_frame)
                
                faceDetections = faceDetector.detect(in_frame)
                if faceDetections.size > 0:
                    faceDetections = removePadOffset(faceDetections)
                    
                    # 오른쪽 절반에 해당하는 감지 결과를 원래 프레임에 네모 박스로 그리기
                    for detection in faceDetections:
                        bbox = detection[2:]
                        if bbox.ndim == 1:
                            bbox[0] = int(bbox[0] * right_half_frame.shape[1]) + frame.shape[1] // 2
                            bbox[1] = int(bbox[1] * right_half_frame.shape[0])
                            bbox[2] = int(bbox[2] * right_half_frame.shape[1]) + frame.shape[1] // 2
                            bbox[3] = int(bbox[3] * right_half_frame.shape[0])
                            cv2.rectangle(frame, 
                                          (int(bbox[0]), int(bbox[1])), 
                                          (int(bbox[2]), int(bbox[3])), 
                                          (0, 255, 0), 2)
                            cv2.putText(frame, 
                                        "face" + " : {0:.2f}".format(detection[1]), 
                                        (int(bbox[0]), int(bbox[1])), 
                                        cv2.FONT_ITALIC, 
                                        0.5, 
                                        (0, 255, 0))
                cv2.imshow('Detected Faces', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

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

# Process each video file
for video_file in video_files:
    process_video(str(video_file), faceDetector, preprocs, faceBoxesCfg_yaml)

