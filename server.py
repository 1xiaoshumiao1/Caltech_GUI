import codecs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
# import DeviceFalutModels
import time
import json
import sys
from mmdet.apis import init_detector, inference_detector

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import cv2
import numpy as np
import base64

# 服务端地址
host = ('192.168.0.195', 8000)

class Resquest(BaseHTTPRequestHandler):
    def handler(self):
        print("data:", self.rfile.readline().decode())
        self.wfile.write(self.rfile.readline())

    def do_GET(self):
        print(self.requestline)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        data = ""
        self.wfile.write(json.dumps(data).encode())

    # 接受post请求
    def do_POST(self):

        # 读取数据
        req_datas = self.rfile.read(int(self.headers['content-length']))
        req = json.loads(req_datas.decode())

        # 检测
        result = Detection(req)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # 返回结果
        self.wfile.write(json.dumps(result).encode('utf-8'))


class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass


# 检测
def Detection(ReqData):
    img_decode_as = ReqData['file'].encode('ascii')
    img_decode = base64.b64decode(img_decode_as)
    img_np_ = np.frombuffer(img_decode, np.uint8)
    img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR)


    # 解析参数
    threshold = ReqData['vis_threshold']
    if threshold < 0.1:
        threshold = 0.1

    time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    imgName = get_current_time() + '.jpg'
    imgPath = './recvPics/' + imgName
    # cv2.imwrite(imgPath, img)
    # 判断图片是否有效
    try:
        cv2.imwrite(imgPath, img)
    except:
        print('img broken!')
        return {"code": "503", "result": 'null', 'msg': '未处理的异常'}
    else:
        print('recvpic path: ', imgPath)

    result = inference_detector(model,img)
    print(result)
    bboxes = np.vstack(result)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
    labels = np.concatenate(labels)
    inds = bboxes[:, -1] > threshold
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    box_objects = []
    for bbox, label in zip(bboxes, labels):
        object_dict = {"Type": model.CLASSES[label],
                       "Confidence": float(bbox[-1]),
                       "X": int(bbox[0]),
                       "Y": int(bbox[1]),
                       "Width": int(bbox[2]-bbox[0]+1),
                       "Height": int(bbox[3]-bbox[1]+1),
                       }
        box_objects.append(object_dict)

    RetData = box_objects
    print('RetData: ', RetData)
    return RetData


def get_current_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

# 如需更改使用的模型,改写下面config,和checkpoint路径
def build_model():
    config = './config/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py'
    checkpoint = './checkpoints/pretrain_attention_epoch_12.pth'
    model = init_detector(config, checkpoint, device='cuda')
    return model


if __name__ == '__main__':
    myServer = ThreadingHttpServer(host, Resquest)
    model = build_model()
    print("Starting http server, listen at: %s:%s" % host)
    myServer.serve_forever()
