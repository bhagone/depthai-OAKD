import numpy as np
import cv2
import depthai  # deals with camera and its data packets
from glob import glob

from util import frame_norm

mobilenet_path = glob("models/*.blob")[0]  # mobileNet

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(mobilenet_path)

cam_rgb.preview.link(detection_nn.input)

# output both rgb and nn inference to host device screen
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# intialize
device = depthai.Device(pipeline)
device.startPipeline()

# host side queues to receive results
q_rgb = device.getOutputQueue("rgb")
q_nn = device.getOutputQueue("nn")

# place-holders to consume the above result
frame = None
bboxes = []

while True:
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()

    # transform the input from rgb camera(1D array) into HWC format
    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    # transform the nn inputs too
    # (image_id, label, confidence, x_min, y_min, x_max, y_max)
    # the last four fields are the bouding boxes
    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[: np.where(bboxes == -1)[0][0]]
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > 0.8][:, 3:7]

    # display the result
    if frame is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            # (image, point1, point2, color, thickness)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("preview", frame)

    if cv2.waitKey(1) == ord("q"):
        break
