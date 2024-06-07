#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################
import time

import cv2
import argparse
import numpy as np

from jetcam.csi_camera import CSICamera

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=False, default="yolov3.cfg",
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=False, default="yolov3.weights",
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=False, default="yolov3.txt",
                help='path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# 初始化摄像头
camera0 = CSICamera(capture_device=0, width=224, height=224)  # 0 通常是默认摄像头的标识

start_time = time.time()

scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

last_frame = camera0.read()
last_box = list()

frame = camera0.read()

last_difference = cv2.mean((last_frame - frame) ** 2)[0]

image = frame

Width = image.shape[1]
Height = image.shape[0]

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

last_box = list()
for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]

    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    last_box.append((class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h)))
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow('obj', image)

epchos = 1

while True:
    # 读取摄像头的一帧画面
    frame = camera0.read()

    difference = cv2.mean((last_frame - frame) ** 2)[0]

    # 评估差异，可以根据需要设定阈值
    if difference < 10 <= last_difference:
        image = frame

        Width = image.shape[1]
        Height = image.shape[0]

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        last_box = list()
        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            last_box.append((class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h)))
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    elif last_difference < 500:
        image = frame
        for box in last_box:
            draw_prediction(image, box[0], box[1], box[2], box[3], box[4], box[5])
    else:
        image = frame

    # 显示帧
    epchos += 1
    cv2.imshow('obj', image)
    if epchos % 10 == 0:
        print(f"FPS:{10 / (time.time() - start_time)}")
        start_time = time.time()
    last_frame = frame
    last_difference = difference

    # 按 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # time.sleep(0.3)
    # print(nonzero_count)

# 释放摄像头
camera0.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
