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

def img_size(img):
    # Return size for image & video
   img_height, img_width = img.shape[:2]
   return (img_height, img_width)
    # Return size for image & video
    # img_height, img_width = img.shape[:2]
    # right_half = img[img_height-300:,img_width // 2:]
    # img_height, img_width = img.shape[:2]
    # return (img_height, img_width)

def faceBoxWrite(img_info, img, detections, plotColor = (0, 255, 0), lineThickness = 2):
    # Bbox rectangle Write for image & video
    height = img_info[0]
    width = img_info[1]
    for detection in detections:
        
        bbox = detection[2:]
        bbox[0] = int(bbox[0] * width)
        bbox[1] = int(bbox[1] * height)
        bbox[2] = int(bbox[2] * width)
        bbox[3] = int(bbox[3] * height)

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

# Model inference Result
#image = cv2.imread('000001.jpg')  
#removePadOffset = RemovePadOffset(img_size(image), faceBoxesCfg_yaml['imageSize'])

#Model Video Capture
video_path = 'input_video01.mp4'
cap = cv2.VideoCapture(video_path)
output_size = (1280, 720)

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output_Cropped_Right_Half.avi', fourcc, 30.0, output_size)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img_info = img_size(frame)
    removePadOffset = RemovePadOffset(img_info, faceBoxesCfg_yaml['imageSize'])

    in_img_file = preprocs(frame)
    faceDetections = faceDetector.detect(in_img_file)

    #For checking the coordinates
    print("faceDetections:", faceDetections)
    print("faceDetections shape:", faceDetections.shape)
    
    #Skip frames that were not detected
    if faceDetections.size != 0:     
        faceDetections = removePadOffset(faceDetections)
    
    #Resizing video 
    result_frame = faceBoxWrite(img_info, frame, faceDetections)
    result_frame_resized = cv2.resize(result_frame, output_size)

    out.write(result_frame_resized)
    cv2.imshow("FaceBoxDetection Frame", result_frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for the 'q' key to exit
        break
# in_img_file = preprocs(img)
# faceDetections = faceDetector.detect(in_img_file)
# faceDetections = removePadOffset(faceDetections)
#Result = faceBoxWrite(img_size(image), image, faceDetections)

# Show Result
cap.release()
out.release()
cv2.destroyAllWindows()