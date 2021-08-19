import numpy as np
import cv2
import time

###################################################################################################################################
def caffe_config_net(proto, model):
    # Config net using cv2 (needs weights and config files)
    net = cv2.dnn.readNetFromCaffe(proto, model)
    
    return net

###################################################################################################################################
def net_processing(img, net):
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
def net_detect(img, net, des_threshold=0.4):
    # Output format:
    # (pc, bx, by, bh, bw, .... class_preds)

    (H, W) = img.shape[:2]
    
    detections, process_time = net_processing(img, net)
    
    locations = np.zeros((5,))
    
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]
      
        if confidence >= des_threshold:
            
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            
            box_info = [(i+1), startX, startY, endX, endY]
            locations = np.vstack([locations, box_info])
            
            # draw the bounding box of the face along with the associated probability
            (w, h) = ((endX - startX), (endY - startY))
    
            text = "Face{}: {:.2f}%".format((i+1), confidence * 100)
                
            # add litle background to class name info:
            backg = np.full((img.shape), (0,0,0), dtype=np.uint8)
            cv2.putText(backg, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            fx,fy,fw,fh = cv2.boundingRect(backg[:,:,2])
                
            # Draw obj bbox:
            cv2.rectangle(img, (startX, startY), (startX + w, startY + h), (0, 0, 255), 2) 
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), -1) 
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3) 
            cv2.putText(img, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)         
                
    return img, locations[1:], process_time

###################################################################################################################################

