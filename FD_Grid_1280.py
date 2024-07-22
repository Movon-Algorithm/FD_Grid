import onnxruntime
import numpy as np
import cv2
import yaml
import torchvision.transforms as transforms
from FaceBoxesV2.faceBoxesV2_detector_onnx import *
from FaceBoxesV2.transforms import *

def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    # Adjust the contrast and brightness of the image
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return img

def apply_clahe(img):
    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb_img)

    # Apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_channel = clahe.apply(y_channel)

    # Merge the channels back
    ycrcb_img = cv2.merge((y_channel, cr, cb))
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return img

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
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    plotColor, lineThickness)
        cv2.putText(img, 
                    "face" + " : {0:.2f}".format(detection[1]), 
                    (bbox[0], bbox[1]), 
                    cv2.FONT_ITALIC, 
                    color = plotColor, 
                    fontScale = 0.5)
        # if detection[1] < 0.95:
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
video_path = 'input_video04.mp4'
cap = cv2.VideoCapture(video_path)
output_size = (1280, 720)

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output01_Cropping.avi', fourcc, 30.0, output_size)

# Crop coordinates
x_start, x_end = 700, 1100 
y_start, y_end = 0, 650

count = 0
failed_frames = []  # List to store frame numbers where detection failed

count2 = 0
low_precision_frames =[]

frame_number = 0  # Initialize frame number
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_number += 1  # Increment frame number
    
    # Adjust contrast and brightness
    frame = adjust_contrast_brightness(frame, contrast=1, brightness=-42) #a=1, b=-8~-20(1) / a=1, b=-30(0)
    frame = apply_clahe(frame)
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # enhanced_frame = apply_clahe(gray_frame)
    # frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

    # Crop the frame using the specified coordinates
    cropped_frame, width_offset, height_offset = crop_img(frame, x_start, x_end, y_start, y_end)
    img_info = (cropped_frame.shape[0], cropped_frame.shape[1], width_offset, height_offset)

    removePadOffset = RemovePadOffset((img_info[0], img_info[1]), faceBoxesCfg_yaml['imageSize'])

    in_img_file = preprocs(cropped_frame)  # Process the cropped frame
    faceDetections = faceDetector.detect(in_img_file)

    # Skip frames that were not detected
    if faceDetections.size != 0:
        faceDetections = removePadOffset(faceDetections)
        for detection in faceDetections:
            if detection[1] < 0.95:
                count2 += 1
                low_precision_frames.append(frame_number)
    else:
        count += 1
        failed_frames.append(frame_number)  # Store the frame number

    # Draw bounding boxes on the original frame, adjust for the right half and height offset
    result_frame = faceBoxWrite(img_info, frame, faceDetections)
    result_frame_resized = cv2.resize(result_frame, output_size)

    #out.write(result_frame_resized)
    cv2.imshow("FaceBoxDetection Frame", result_frame_resized)

    if cv2.waitKey(100) & 0xFF == ord('q'):  # Wait for the 'q' key to exit
        break

# Print the Number of not detected frames    
print("Not detected Frames: ", count)
print("Frames where face detection failed: ", failed_frames)

print("Low precision Frames: ", count2)
print("Frames where face detection has low precision: ", low_precision_frames)
# Show Result
cap.release()
out.release()
cv2.destroyAllWindows()
