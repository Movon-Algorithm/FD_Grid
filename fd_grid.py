#240722 #ShinDaYeon #FD grid Project

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

def crop(img):
    #function which crop the origin image
    height, width = img_size(img)

    a=int(height*0.152)
    b=int(height*(1-0.01))
    c=int(width*(0.53))
    d=int(width*(1-0.02))

    crop_img = img[a:b, c:d]

    area = (b-a)*(d-c) #area of cropped image

    return crop_img, a, b, c, d, area

config = 1

def faceBoxWrite(img, detections, plotColor = (0, 255, 0), lineThickness = 2):
    # Bbox rectangle Write for image & video
    for detection in detections:
        
        img_cropped = crop(img)[0] 

        global bbox, config
        config = detection[1]

        bbox = detection[2:]
        bbox[0] = int(bbox[0] * img_size(img_cropped)[1])
        bbox[1] = int(bbox[1] * img_size(img_cropped)[0])
        bbox[2] = int(bbox[2] * img_size(img_cropped)[1])
        bbox[3] = int(bbox[3] * img_size(img_cropped)[0])
        bbox = bbox.astype(int)

        cv2.rectangle(img, 
                    (int(bbox[0]+crop(img)[3]), int(bbox[1]+crop(img)[1])), 
                    (int(bbox[2]+crop(img)[3]), int(bbox[3]+crop(img)[1])), 
                    plotColor, lineThickness)
        cv2.putText(img, 
                    "face" + " : {0:.2f}".format(detection[1]), 
                    (bbox[0]+crop(img)[3], bbox[1]+crop(img)[1]-10), 
                    cv2.FONT_ITALIC, 
                    color = plotColor, 
                    fontScale = 0.7)

    return img, config

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

# video number
video1 = '20230820_082819_NOR_ch1'
video2 = '20230820_233653_NOR_ch1'
video3 = 'Light_20230803_084717_NOR_ch1'
video4 = 'Light_20230803_084738_NOR_ch1'

for i in [video1, video2, video3, video4]: #iterate each video
    video = cv2.VideoCapture(f"{i}.mp4")
    fd_count = 0
    fd_pass = 0
    sum_of_confi = 0
    while video.isOpened():
        check, image = video.read()
        height, width = image.shape[:2]
        orig_area = height *width

        img_crop = crop(image)[0]

        removePadOffset = RemovePadOffset(img_size(img_crop), faceBoxesCfg_yaml['imageSize'])

        in_img_file = preprocs(img_crop)
        faceDetections = faceDetector.detect(in_img_file)

        ok = False
        try:
            faceDetections = removePadOffset(faceDetections)
            ok = True
            fd_count += 1
        except: 
            fd_pass += 1
            pass

        Result, confi = faceBoxWrite(image, faceDetections)

        if ok: #detect된 frame에 한해 confidence 측정
            sum_of_confi += confi
        
        if not check:
            print("Frame이 끝났습니다.")
            break

        cv2.imshow("Detected Faces", Result)

        # fps = video.get(cv2.CAP_PROP_FPS) #fps = 30.0
        # delay = int(1000/fps)
        if cv2.waitKey(25) == ord('q'): #waitKey 안의 숫자(delay) 변경하여도 속도 변화 없음 #coumpert 속도 성능때문이라고 추측
            print("q키를 눌러, 동영상 종료됨")
            break

        #frame number를 볼 수 있는 코드
        #print(f"width: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}, height: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}, frame number: {int(video.get(cv2.CAP_PROP_POS_FRAMES))}, frame count: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")

        if int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == int(video.get(cv2.CAP_PROP_POS_FRAMES)):
            break
    print(f"===== {i} video =====")
    print(f"width: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}, height: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}, a: {crop(image)[1]}, b: {crop(image)[2]}, c: {crop(image)[3]}, d: {crop(image)[4]}")
    print(f"average of confi: {sum_of_confi/fd_count: .3f}, fd_count: {fd_count}, fd_pass: {fd_pass}, percent of detect frame:{(fd_count/(fd_count+fd_pass))*100: .3f}%")
    print(f"The size of grid: {crop(image)[5]}/{orig_area}, percent of grid area: {crop(image)[5]/orig_area*100:.3f}%, starting point: {crop(image)[3], crop(image)[1]}")
    print(' ')
    # print(f"print fps(ms) {video.get(cv2.CAP_PROP_FPS)}ms\n") #렌 = 30.0
    video.release()
    cv2.destroyAllWindows()