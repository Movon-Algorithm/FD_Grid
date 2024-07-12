import onnxruntime
import numpy as np
import cv2
import yaml

import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import *
from FaceBoxesV2.transforms import *


def preprocess():
    # Preprocess for image & video
    preprocs_list = [ResizeImage(faceBoxesCfg_yaml['imageSize']),
                     LetterBox(faceBoxesCfg_yaml['imageSize'])]
    preprocs_list.append(ConvertColor('GRAY1ch'))
    preprocs_list.append(transforms.ToTensor())
    preprocs_list.append(transforms.Normalize(mean=0.485, std=0.229))
    preprocs_list += [ExpandBatchDim(), toDevice('cpu')]
    return preprocs_list

def crop_img(img, x_start, x_end, y_start, y_end):
    # Crop the image based on provided coordinates
    cropped_img = img[y_start:y_end, x_start:x_end]
    return cropped_img, x_start, y_start

def faceBoxWrite(img_info, img, detections, plotColor=(0, 255, 0), lineThickness=2):
    # Bbox rectangle Write for image & video
    height, width, width_offset, height_offset = img_info  # Get the height, width, and offset

    for detection in detections:
        bbox = detection[2:]
        
        bbox[0] = int(bbox[0] * width) + width_offset
        bbox[1] = int(bbox[1] * height) + height_offset
        bbox[2] = int(bbox[2] * width) + width_offset
        bbox[3] = int(bbox[3] * height) + height_offset

        bbox = bbox.astype(int)
        cv2.rectangle(img,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      plotColor, lineThickness)
        cv2.putText(img,
                    "face" + " : {0:.2f}".format(detection[1]),
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_ITALIC,
                    color=plotColor,
                    fontScale=0.5)
        
        print("face" + " : {0:.2f}".format(detection[1]))
    return img

# Setting Process (yaml file load)
with open('.\\FaceBoxesV2\\faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
    faceBoxesCfg_yaml = yaml.safe_load(file)
with open('.\\FaceBoxesV2\\priorCfg.yaml', 'r', encoding='utf-8') as file:
    priorCfg_yaml = yaml.safe_load(file)

# Work for preprocess
preprocs_list = preprocess()
preprocs = Compose(preprocs_list)

# Model init & Inference (ONNX)
model_path = 'mdfd.onnx'
onnxruntime.get_device()
sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
faceDetector = FaceBoxesONNXDetector(model_path, faceBoxesCfg_yaml, priorCfg_yaml, 'cpu')

# Model Video Capture
video_path = 'input_video01.mp4'
cap = cv2.VideoCapture(video_path)
output_size = (1280, 720)

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output01_Cropping.avi', fourcc, 30.0, output_size)

# Crop coordinates
x_start, x_end = 1000, 1700
y_start, y_end = 320, 1080

count = 0
failed_frames = []  # List to store frame numbers where detection failed

frame_number = 0  # Initialize frame number
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_number += 1  # Increment frame number
    
    # Crop the frame using the specified coordinates
    cropped_frame, width_offset, height_offset = crop_img(frame, x_start, x_end, y_start, y_end)
    img_info = (cropped_frame.shape[0], cropped_frame.shape[1], width_offset, height_offset)

    removePadOffset = RemovePadOffset((img_info[0], img_info[1]), faceBoxesCfg_yaml['imageSize'])

    in_img_file = preprocs(cropped_frame)  # Process the cropped frame
    faceDetections = faceDetector.detect(in_img_file)

    # Skip frames that were not detected
    if faceDetections.size != 0:
        faceDetections = removePadOffset(faceDetections)
    else:
        count += 1
        failed_frames.append(frame_number)  # Store the frame number

    # Draw bounding boxes on the original frame, adjust for the right half and height offset
    result_frame = faceBoxWrite(img_info, frame, faceDetections)
    result_frame_resized = cv2.resize(result_frame, output_size)

    out.write(result_frame_resized)
    cv2.imshow("FaceBoxDetection Frame", result_frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for the 'q' key to exit
        break

# Print the Number of not detected frames    
print("Not detected Frames: ", count)
print("Frames where face detection failed: ", failed_frames)

# Show Result
cap.release()
out.release()
cv2.destroyAllWindows()
