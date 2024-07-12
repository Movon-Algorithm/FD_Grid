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
    # img_crop = img[:,int(img_width/2):] #non-solved
    # img_height, img_width = img_crop.shape[:2]
    return (img_height, img_width)

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

# videoes
#20230820_082819_NOR_ch1
#20230820_233653_NOR_ch1
#Light_20230803_084717_NOR_ch1
#Light_20230803_084738_NOR_ch1

video = cv2.VideoCapture("20230820_082819_NOR_ch1.mp4")

while video.isOpened():
    check, image = video.read()

    removePadOffset = RemovePadOffset(img_size(image), faceBoxesCfg_yaml['imageSize'])

    in_img_file = preprocs(image)
    faceDetections = faceDetector.detect(in_img_file)

    try:
        faceDetections = removePadOffset(faceDetections)
    except: 
        pass

    Result = faceBoxWrite(img_size(image), image, faceDetections)

    if not check:
        print("Frame이 끝났습니다.")
        break
    
    cv2.imshow("Detected Faces",Result)

    if cv2.waitKey(25) == ord('q'):
        print("동영상 종료")
        break
    
    print(f"width: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}, height: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}, frame number: {int(video.get(cv2.CAP_PROP_POS_FRAMES))}, frame count: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")


video.release()
cv2.destroyAllWindows()



# Model inference Result
# image = cv2.imread('000001.jpg')
# removePadOffset = RemovePadOffset(img_size(video), faceBoxesCfg_yaml['imageSize'])

# in_img_file = preprocs(video)
# faceDetections = faceDetector.detect(in_img_file)
# faceDetections = removePadOffset(faceDetections)
# Result = faceBoxWrite(img_size(video), video, faceDetections)

#Show Result
# cv2.imshow("Detected Faces", Result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

