import numpy as np
import onnxruntime
from FaceBoxesV2.prior_box_onnx import *
from FaceBoxesV2.box_utils_onnx import *

class FaceBoxesONNXDetector():
    def __init__(self, onnx_model_path, detectorCfg, priorCfg, device_id):
        if device_id == 'cpu':
            onnxruntime.get_device()
            self.sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        else:
            onnxruntime.get_device()
            # 디바이스 번호 넣어주고 싶었는데 작동안함 .. 20230615
            #options = onnxruntime.SessionOptions()
            #options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            #cuda_provider = onnxruntime.CudaExecutionProvider(device_id=device_id)
            #options.add_extension(cuda_provider)
            #self.sess = onnxruntime.InferenceSession(onnx_model_path, options)
            self.sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.priorCfg = priorCfg
        self.priors = PriorBox(priorCfg, image_size=detectorCfg['imageSize']).forward()
        self.thresh = detectorCfg['thresh']        
        self.inputChannel = detectorCfg['imageChannel']

    def detect(self, image):
        image = np.array(image.cpu(), dtype=np.float32)
        input_dict = {self.input_name: image}
        loc, conf = self.sess.run(None, input_dict)
        detections = []
        #bbox = []
        #boxes = decode(loc.data.squeeze(0), self.priors.data, self.priorCfg['variance'])
        boxes = decode(np.array(loc.data).squeeze(0), np.array(self.priors.data), self.priorCfg['variance'])


        conf = conf.reshape(-1, 2)
        scores = np.exp(conf[:, 1]) / np.sum(np.exp(conf), axis=1)
        inds = np.where(scores > self.thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        if order.shape[0] > 5000:
            order = order[:5000]
        boxes = boxes[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.3)
        dets = dets[keep, :]

        dets = dets[:750, :]

        for i in range(dets.shape[0]):
            xmin = dets[i][0]
            ymin = dets[i][1]
            xmax = dets[i][2]
            ymax = dets[i][3]
            scores = dets[i][4]
            detections.append([99.0, scores, xmin, ymin, xmax, ymax])
            #bbox.append([xmin, ymin, xmax, ymax])
        ret = np.array(detections)
        #print(ret)
        #return ret, bbox # x - bbox 불필요 @Chagnmin yi
        return ret