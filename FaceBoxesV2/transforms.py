

import cv2
# from utils.generals import *
import copy

class TranslationBox():
    def __init__(self, cropBoxPt, offset = 2):
        self.cropBoxPt = cropBoxPt
        self.cropBoxHeight = cropBoxPt[3] - cropBoxPt[1]
        self.cropBoxWidth = cropBoxPt[2] - cropBoxPt[0]
        self.offset = offset
    
    def __call__(self, labels):
        # ret = labels.copy()
        ret = copy.deepcopy(labels)
        # pdb.set_trace()
        #shift origin
        ret[:,self.offset + 0] = (labels[:, self.offset + 0] - self.cropBoxPt[0])
        ret[:,self.offset + 1] = (labels[:, self.offset + 1] - self.cropBoxPt[1])
        ret[:,self.offset + 2] = (labels[:, self.offset + 2] - self.cropBoxPt[0])
        ret[:,self.offset + 3] = (labels[:, self.offset + 3] - self.cropBoxPt[1])

        ret[:,self.offset + 0][:,self.offset + 0 < 0] = 0
        ret[:,self.offset + 1][:,self.offset + 1 < 0] = 0
        ret[:,self.offset + 2][:,self.offset + 2 < 0] = 0
        ret[:,self.offset + 3][:,self.offset + 3 < 0] = 0

        ret[:,self.offset + 0][ret[:,self.offset + 0] > (self.cropBoxWidth - 1)] = self.cropBoxWidth - 1
        ret[:,self.offset + 1][ret[:,self.offset + 1] > (self.cropBoxHeight - 1)] = self.cropBoxHeight - 1
        ret[:,self.offset + 2][ret[:,self.offset + 2] > (self.cropBoxWidth - 1)] = self.cropBoxWidth - 1
        ret[:,self.offset + 3][ret[:,self.offset + 3] > (self.cropBoxHeight - 1)] = self.cropBoxHeight - 1

        return ret



class PruneBox(): #only available for xywh format.
    def __init__(self, croppedImagePt, offset):
        self.croppedImagePt = croppedImagePt #(xmin, ymin, xmax, ymax) (Absolute scale)
        self.offset = offset
        
    def __call__(self, labels): #labels : (x_center, y_center, width, height)

        idx_list = []

        for idx, label in enumerate(labels):
            if not self.isInBox(label):
                idx_list.append(idx)
        
        labels = np.delete(labels, idx_list, axis = 0)

        return labels

    def isInBox(self, label):
        xFlag = True if ((label[self.offset + 0] > self.croppedImagePt[0]) and (label[self.offset + 0] < self.croppedImagePt[2])) else False
        yFlag = True if ((label[self.offset + 1] > self.croppedImagePt[1]) and (label[self.offset + 1] < self.croppedImagePt[3])) else False

        # pdb.set_trace()
        ret = (xFlag and yFlag)

        return ret

class Norm2Abs():

    def __init__(self, imagesize, offset = 2):
        self.imageSize = imagesize #(Height, Width)
        self.offset = offset

    def __call__(self, labels):
        labels = labels.astype(float)

        labels[:,self.offset + 0] = labels[:,self.offset + 0] * self.imageSize[1]
        labels[:,self.offset + 1] = labels[:,self.offset + 1] * self.imageSize[0]
        labels[:,self.offset + 2] = labels[:,self.offset + 2] * self.imageSize[1]
        labels[:,self.offset + 3] = labels[:,self.offset + 3] * self.imageSize[0]
        # if len(labels) >0:
        #     pdb.set_trace()
        # pdb.set_trace()
        return labels

class Abs2Norm():

    def __init__(self, imagesize, offset = 2):
        self.imageSize = imagesize #(Height, Width)
        self.offset = offset

    def __call__(self, labels):
        labels = labels.astype(float)
        # pdb.set_trace()
        labels[:, self.offset + 0] = labels[:, self.offset + 0] / self.imageSize[1]
        labels[:, self.offset + 1] = labels[:, self.offset + 1] / self.imageSize[0]
        labels[:, self.offset + 2] = labels[:, self.offset + 2] / self.imageSize[1]
        labels[:, self.offset + 3] = labels[:, self.offset + 3] / self.imageSize[0]
        # if len(labels) >0:
        #     pdb.set_trace()
        # pdb.set_trace()
        return labels

class XYWH2XYXY():
    def __init__(self, offset = 2):
        self.offset = offset
    def __call__(self, labels):
        # ret = labels.copy()
        ret = copy.deepcopy(labels)
        
        ret[:,self.offset + 0] = labels[:, self.offset + 0] - labels[:, self.offset + 2]/2
        ret[:,self.offset + 1] = labels[:, self.offset + 1] - labels[:, self.offset + 3]/2
        ret[:,self.offset + 2] = labels[:, self.offset + 0] + labels[:, self.offset + 2]/2
        ret[:,self.offset + 3] = labels[:, self.offset + 1] + labels[:, self.offset + 3]/2
        # pdb.set_trace()
        return ret

class XYXY2XYWH():
    def __init__(self, offset = 2):
        self.offset = offset
    def __call__(self, labels):
        # ret = labels.copy()
        ret = copy.deepcopy(labels)
        
        ret[:,self.offset + 0] = (labels[:, self.offset + 0] + labels[:, self.offset + 2])/2
        ret[:,self.offset + 1] = (labels[:, self.offset + 1] + labels[:, self.offset + 3])/2
        ret[:,self.offset + 2] = (labels[:, self.offset + 2] - labels[:, self.offset + 0])
        ret[:,self.offset + 3] = (labels[:, self.offset + 3] - labels[:, self.offset + 1])
        # pdb.set_trace()
        return ret

class CutBox():
    def __init__(self, offset = 2):
        self.offset = offset

    def __call__(self, labels):
        labels[:,self.offset + 0][labels[:,self.offset + 0] < 0] = 0
        labels[:,self.offset + 1][labels[:,self.offset + 1] < 0] = 0
        labels[:,self.offset + 2][labels[:,self.offset + 2] > (1.0)] = 1.0
        labels[:,self.offset + 3][labels[:,self.offset + 3] > (1.0)] = 1.0

        return labels

class ComposeCoordTransform():
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, labels):
        for t in self.transforms:
            labels = t(labels)
        # pdb.set_trace()
        return labels

class ShiftOrigin():
    def __init__(self, targetImageSize, sourceImagePoint):
        self.targetImageSize = targetImageSize
        self.sourceImageOrigin = sourceImagePoint[:2] #(xmin, ymin)
        self.sourceImageSize = (sourceImagePoint[3] - sourceImagePoint[1], sourceImagePoint[2] - sourceImagePoint[0]) #(height, width)
    def __call__(self, label):
        # pdb.set_trace()

        label[:, 2] = (label[:, 2] * self.sourceImageSize[1] + self.sourceImageOrigin[0]) / self.targetImageSize[1]
        label[:, 3] = (label[:, 3] * self.sourceImageSize[0] + self.sourceImageOrigin[1]) / self.targetImageSize[0]
        label[:, 4] = (label[:, 4] * self.sourceImageSize[1] + self.sourceImageOrigin[0]) / self.targetImageSize[1]
        label[:, 5] = (label[:, 5] * self.sourceImageSize[0] + self.sourceImageOrigin[1]) / self.targetImageSize[0]

        return label

class RemovePadOffset():
    def __init__(self, targetImageSize, sourceImageSize): #size format : (height, width)
        #get padded size
        self.targetImageSize = targetImageSize
        ratio = max(self.targetImageSize[0] / sourceImageSize[0], self.targetImageSize[1] / sourceImageSize[1])
        self.expandedSourceImageSize = int(round(sourceImageSize[0] * ratio)), int(round(sourceImageSize[1] * ratio)) # height, width
        dh, dw = self.expandedSourceImageSize[0] - targetImageSize[0], self.expandedSourceImageSize[1] - targetImageSize[1]

        dh /= 2
        dw /= 2
        
        assert dh * dw == 0

        self.dh = dh
        self.dw = dw


    def __call__(self, label):
        # pdb.set_trace()
        # print(label)
        # print(type(label))
        # if len(label) >1:
        #     pdb.set_trace()
        label[:, 2] = (label[:, 2] * self.expandedSourceImageSize[1]- self.dw) / self.targetImageSize[1]
        label[:, 3] = (label[:, 3] * self.expandedSourceImageSize[0]- self.dh) / self.targetImageSize[0]
        label[:, 4] = (label[:, 4] * self.expandedSourceImageSize[1]- self.dw) / self.targetImageSize[1]
        label[:, 5] = (label[:, 5] * self.expandedSourceImageSize[0]- self.dh) / self.targetImageSize[0]

        return label
        


def lmsCoordTransforms(lmsRatioCoord, cropBoxPt):
    cropBoxWidth = cropBoxPt[2] - cropBoxPt[0]
    cropBoxHeight = cropBoxPt[3] - cropBoxPt[1]

    # cv2.rectangle(frame, 
    #                 (cropBoxPt[0], cropBoxPt[1]), 
    #                 (cropBoxPt[2], cropBoxPt[3]), 
    #                 (255, 255, 255), 1)

    #lmsAbsCoord = np.zeros(lmsRatioCoord.shape, dtype = np.int)
    lmsAbsCoord = np.zeros(lmsRatioCoord.shape, dtype = int)
    for idx, point in enumerate(lmsRatioCoord):
        # pdb.set_trace()
        x_pred = int(point[0] * cropBoxWidth)
        y_pred = int(point[1] * cropBoxHeight)
        lmsAbsCoord[idx] = [x_pred+cropBoxPt[0], y_pred+cropBoxPt[1]]

    return lmsAbsCoord

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    # pdb.set_trace()
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if shape[0] <= 0 or shape[1] <= 0 : # 240p exception
        return img, (1, 1), (0, 0)
    # Compute padding
    ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # height, width ratios
    dh, dw = new_shape[0] - shape[0], new_shape[1] - shape[1]  # hw padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # print(top, bottom, left, right)
    # pdb.set_trace()
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    dw = int(dw)
    dh = int(dh)

    return img, ratio, (dh, dw)
    
'''
class ResizeImage():
    def __init__(self,new_img_size=(480, 640), fixRatio = True): #new_img_size : (Height, Width)
        self.new_img_size = new_img_size
        self.fixRatio = fixRatio
    def __call__(self, image):
        h0, w0 = image.shape[:2]  # orig hw
        rH = self.new_img_size[0] / h0  # ratio
        rW = self.new_img_size[1] / w0
        r = min(rH, rW)
        
        if (rH, rW) == (1, 1):

            return image

        if self.fixRatio:  # if sizes are not equal
            image = cv2.resize(image, None, None, fx = r, fy = r,
                            interpolation=cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA)
        else:
            image = cv2.resize(image, self.new_img_size)

        return image
'''

    

class LetterBox():
    def __init__(self,new_img_size=(480, 640), color = (114, 114, 114)): #new_img_size : (Height, Width)
        self.new_img_size = new_img_size
        self.color = color
    def __call__(self, image):
        h0, w0 = image.shape[:2]
        # pdb.set_trace()
        image, ratio, pad = letterbox(image, self.new_img_size, color = self.color,  auto=False, scaleup=True)
        
        return image

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, source):
        for t in self.transforms:
            source = t(source)
        return source

class Compose_pdb():
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, source):
        for t in self.transforms:
            print(type(source), source.shape, t)
            pdb.set_trace()
            source = t(source)
        return source


class ExpandBatchDim():
    def __init__(self):
        pass
    def __call__(self, image):
        image = image[None] # expand for batch dim

        return image

class toDevice():
    def __init__(self, device):
        self.device = device

    def __call__(self, image):

        image = image.to(self.device)

        return image

class PostProcs():
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, image):
        # pdb.set_trace()
        image = image[0].cpu()
        image = image * torch.tensor(self.std)[:, None, None] + torch.tensor(self.mean)[:, None, None]
        image = image.numpy().transpose(1,2,0)*255
        image = np.ascontiguousarray(image)
        image = image.astype(np.uint8)
        return image

class PostProcs_2():
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, image):
        image = image[0].cpu()
        # image = image * torch.tensor(self.std)[:, None, None] + torch.tensor(self.mean)[:, None, None]
        image = image.numpy().transpose(1,2,0)*255
        image = np.ascontiguousarray(image)
        image = image.astype(np.uint8)
        return image

def cropImage(image, detection, scale = 1.0):
    height, width = image.shape[:2]
    bbox = detection[2:]
    xmin = int(bbox[0] * width)
    ymin = int(bbox[1] * height)
    xmax = int(bbox[2] * width)
    ymax = int(bbox[3] * height)

    xSize = xmax - xmin
    ySize = ymax - ymin

    xmin -= int(xSize * (scale-1)/2)
    ymin -= int(ySize * (scale-1)/2)
    xmax += int(xSize * (scale-1)/2)
    ymax += int(ySize * (scale-1)/2)
    
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, width-1)
    ymax = min(ymax, height-1)
    # pdb.set_trace()
    image = image[ymin:ymax, xmin:xmax, :]
    # pdb.set_trace()
    return image, (xmin, ymin, xmax, ymax)


def cropImagePSR(image, faceDetection, scaleX = 3.0, scaleY = 2.0, yOffset = 0.4, doNotCropFlag = False):
    #faceDetection format : (face, confidence, xmin, ymin, xmax, ymax) , coordinate is normalizeds
    #outputSize : (height, width)
    

    bbox = faceDetection[2:]
    # pdb.set_trace()
    orgImageHeight, orgImageWidth = image.shape[:2]

    bbox = faceDetection[2:]

    # print(cropOriginX, cropOriginY)
    # xCenter = ((bbox[0] + bbox[2]) / 2) * orgImageWidth
    # yCenter = ((bbox[1] + bbox[3]) / 2) * orgImageHeight
    xCenter = ((bbox[0] + bbox[2]) / 2)
    yCenter = ((bbox[1] + bbox[3]) / 2)

    # xWidth = ((bbox[2] - bbox[0])) * orgImageWidth 
    # yWidth = ((bbox[3] - bbox[1])) * orgImageHeight
    xWidth = ((bbox[2] - bbox[0]))
    yWidth = ((bbox[3] - bbox[1]))

    yCenter += yWidth * yOffset

    newOutputSize = (yWidth * scaleY, xWidth * scaleX)

    newOutputImageHeight, newOutputImageWidth = newOutputSize
    
    if doNotCropFlag:
        xmin = 0
        ymin = 0
        xmax = orgImageWidth
        ymax = orgImageHeight

    else:
        xmin = xCenter - newOutputImageWidth / 2
        ymin = yCenter - newOutputImageHeight / 2
        xmax = xCenter + newOutputImageWidth / 2
        ymax = yCenter + newOutputImageHeight / 2

        # newOutputSize = (96, 96)

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, orgImageWidth-1)
    ymax = min(ymax, orgImageHeight-1)
    # pdb.set_trace()
    # cv2.imwrite("test_beforeCrop.png", image)
    image = image[ymin:ymax, xmin:xmax, :]
    # pdb.set_trace()
    # image = resizeImagePSR(image, newOutputSize)
    # image, ratio, pad = letterbox(image, newOutputSize)
    # pdb.set_trace()
    # cv2.imwrite("test_afterCrop.png", image)
    # pdb.set_trace()
    # print(pad)
    # psrInputSize = (pad[])
    return image, (xmin, ymin, xmax, ymax)

def resizeImagePSR(image, new_img_size, fixRatio = True):
    h0, w0 = image.shape[:2]  # orig hw
    rH = new_img_size[0] / h0  # ratio
    rW = new_img_size[1] / w0
    r = min(rH, rW)
    
    if (rH, rW) == (1, 1):

        return image

    if fixRatio:  # if sizes are not equal

        image = cv2.resize(image, None, None, fx = r, fy = r,
                        interpolation=cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA)
    else:
        image = cv2.resize(image, new_img_size)

    return image

class HWC2CHW():
    def __init__(self):
        pass
    def __call__(self, image):
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis = -1)
        image = image.transpose((2, 0, 1))


        return image

class Numpy2Tensor():
    def __init__(self, half = False):
        self.half = half
    def __call__(self, image):
        image = torch.tensor(image)

        image = image.half() if self.half else image.float()

        return image

class ScaleDown():
    def __init__(self, scale = 255):
        self.scale = 255
    def __call__(self, image):
        image /= self.scale

        return image
class ScaleUp():
    def __init__(self, scale = 255):
        self.scale = scale
    def __call__(self, image):
        # pdb.set_trace()
        image = image * self.scale
        # pdb.set_trace()

        return image

class ResizeImage():
    def __init__(self,new_img_size=(480, 640), fixRatio = True): #new_img_size : (Height, Width)
        self.new_img_size = new_img_size
        self.fixRatio = fixRatio
    def __call__(self, image):
        # pdb.set_trace()
        h0, w0 = image.shape[:2]  # orig hw
        
        if h0 <= 0 or w0 <= 0 : # 240p exception
            return image

        rH = self.new_img_size[0] / h0  # ratio
        rW = self.new_img_size[1] / w0
        r = min(rH, rW)
        
        if (rH, rW) == (1, 1):

            return image

        if self.fixRatio:  # if sizes are not equal
            image = cv2.resize(image, None, None, fx = r, fy = r,
                            interpolation=cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA)
        else:
            # pdb.set_trace()
            image = cv2.resize(image, (self.new_img_size[1],self.new_img_size[0]))

        return image

class ConvertColor():
    def __init__(self, outFormat = 'BGR'):
        self.outFormat = outFormat
    def __call__(self, image):
        # pdb.set_trace()
        if self.outFormat == 'BGR':
            pass
        elif self.outFormat == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif self.outFormat =='RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.outFormat == 'GRAY1ch':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.outFormat == 'GRAY2BGR':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise Exception('please check outFormat')


        # pdb.set_trace()

        return image
