import onnxruntime
import numpy as np
import cv2
import yaml
import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import *
from FaceBoxesV2.transforms import *

def Contrast_frame(img, contrast=1.0, brightness=0):
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return img

def apply_clahe(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_channel = clahe.apply(y_channel)

    ycrcb_img = cv2.merge((y_channel, cr, cb))
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return img

def preprocess():
    preprocs_list = [ResizeImage(faceBoxesCfg_yaml['imageSize']),
                     LetterBox(faceBoxesCfg_yaml['imageSize'])]
    preprocs_list.append(ConvertColor('GRAY1ch'))
    preprocs_list.append(transforms.ToTensor())
    preprocs_list.append(transforms.Normalize(mean=0.485, std=0.229))
    preprocs_list += [ExpandBatchDim(), toDevice('cpu')]
    return preprocs_list

def crop_img(img, x_start, x_end, y_start, y_end):
    cropped_img = img[y_start:y_end, x_start:x_end]
    return cropped_img, x_start, y_start

def faceBoxWrite(img_info, img, detections, plotColor=(0, 255, 0), lineThickness=2):
    height, width, width_offset, height_offset = img_info

    for detection in detections:
        bbox = detection[2:]
        
        bbox[0] = int(bbox[0] * width) + width_offset
        bbox[1] = int(bbox[1] * height) + height_offset
        bbox[2] = int(bbox[2] * width) + width_offset
        bbox[3] = int(bbox[3] * height) + height_offset

        bbox = bbox.astype(int)
        cv2.rectangle(img, 
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    plotColor, lineThickness)
        cv2.putText(img, 
                    "face" + " : {0:.2f}".format(detection[1]), 
                    (bbox[0], bbox[1]), 
                    cv2.FONT_ITALIC, 
                    color = plotColor, 
                    fontScale = 0.5)
        print("face" + " : {0:.2f}".format(detection[1]))
    return img

with open('.\\FaceBoxesV2\\faceBoxesV2Cfg.yaml', 'r', encoding='utf-8') as file:
    faceBoxesCfg_yaml = yaml.safe_load(file)
with open('.\\FaceBoxesV2\\priorCfg.yaml', 'r', encoding='utf-8') as file:
    priorCfg_yaml = yaml.safe_load(file)

preprocs_list = preprocess()
preprocs = Compose(preprocs_list)

model_path = 'mdfd.onnx'
onnxruntime.get_device()
sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
faceDetector = FaceBoxesONNXDetector(model_path, faceBoxesCfg_yaml, priorCfg_yaml, 'cpu')

video_path = 'input_video03.mp4'
cap = cv2.VideoCapture(video_path)
output_size = (1280, 720)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video03.mp4', fourcc, 30.0, output_size)

x_start, x_end = 660, 1140
y_start, y_end = 0, 640

count = 0
failed_frames = [] 

count2 = 0
low_precision_frames =[]

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_number += 1
    
    original_frame = frame.copy()  # Store the original frame

    frame = Contrast_frame(frame, contrast=1, brightness=-42)
    frame = apply_clahe(frame)

    cropped_frame, width_offset, height_offset = crop_img(frame, x_start, x_end, y_start, y_end)
    img_info = (cropped_frame.shape[0], cropped_frame.shape[1], width_offset, height_offset)

    removePadOffset = RemovePadOffset((img_info[0], img_info[1]), faceBoxesCfg_yaml['imageSize'])

    in_img_file = preprocs(cropped_frame)  
    faceDetections = faceDetector.detect(in_img_file)

    if faceDetections.size != 0:
        faceDetections = removePadOffset(faceDetections)
        for detection in faceDetections:
            if detection[1] < 0.95:
                count2 += 1
                low_precision_frames.append(frame_number)
    else:
        count += 1
        failed_frames.append(frame_number)  

    # Draw bounding boxes on the original frame, adjusting coordinates for cropped region
    result_frame = faceBoxWrite(img_info, original_frame, faceDetections)
    result_frame_resized = cv2.resize(result_frame, output_size)

    out.write(result_frame_resized)
    cv2.imshow("FaceBoxDetection Frame", result_frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Not detected Frames: ", count)
print("Frames where face detection failed: ", failed_frames)

print("Low precision Frames: ", count2)
print("Frames where face detection has low precision: ", low_precision_frames)

cap.release()
out.release()
cv2.destroyAllWindows()
