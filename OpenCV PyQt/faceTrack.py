import numpy as np
import cv2
import time

###################################################################################################################################
def caffe_config_net(proto, model):
    # Config net using cv2 (needs weights and config files)
    net = cv2.dnn.readNetFromCaffe(proto, model)
    
    return net

###################################################################################################################################
def caffe_processing(img, net):
    # Pass img/frame, net and output layers get output:

    start= time.time()

    # Modify img to input format:
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass to net:
    net.setInput(blob)
    detections = net.forward()

    end = time.time()
    
    process_time = end - start
    
    return detections, process_time

###################################################################################################################################
def caffe_detect(img, net, des_threshold=0.4):
    # Output format:
    # (pc, bx, by, bh, bw, .... class_preds)

    (H, W) = img.shape[:2]
    
    detections, process_time = caffe_processing(img, net)
    
    locations = np.zeros((5,))
    
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence >= des_threshold:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            
            box_info = [(i+1), startX, startY, endX, endY]
            locations = np.vstack([locations, box_info])
            
            # draw the bounding box of the face along with the associated probability
            text = "Face{}: {:.2f}%".format((i+1), confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)            
    
    return img, locations[1:], process_time

###################################################################################################################################

