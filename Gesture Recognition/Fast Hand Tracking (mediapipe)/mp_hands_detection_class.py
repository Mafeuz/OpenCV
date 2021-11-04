import numpy as np
import cv2
import mediapipe as mp

##############################################################################################################################
########################################### Mediapipe Hands Detection Class ##################################################

class mp_hands_detection:
    
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.mpDraw  = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        
        self.static_image_mode        = static_image_mode
        self.max_num_hands            = max_num_hands 
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        
        self.hands = self.mpHands.Hands(static_image_mode = static_image_mode, 
                                        max_num_hands = max_num_hands, 
                                        min_detection_confidence = min_detection_confidence,
                                        min_tracking_confidence = min_tracking_confidence)
        
    def hand_process(self, frame_bgr, draw=True):
        
        self.frame_bgr = frame_bgr
        self.draw = draw
    
        self.results = self.hands.process(cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB))
        
        self.landmarks_list = []
        self.landmarks_list_no_scaled = []
    
        if self.results.multi_hand_landmarks: 
            
            for handLandmarks in self.results.multi_hand_landmarks:

                self.landmarks_list = []
                self.landmarks_list_no_scaled = []

                for id, lm in enumerate(handLandmarks.landmark):
                    h, w, c = self.frame_bgr.shape
                    cX, cY  = int(lm.x*w), int(lm.y*h)

                    self.landmarks_list.append([id, cX, cY])
                    self.landmarks_list_no_scaled.append([id, lm.x, lm.y])

                if self.draw:
                    self.mpDraw.draw_landmarks(self.frame_bgr, handLandmarks, self.mpHands.HAND_CONNECTIONS)
                    cv2.circle(self.frame_bgr, (cX, cY), 1, (255, 0, 0), cv2.FILLED)
                    cv2.putText(self.frame_bgr, '{}'.format(id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 0, 0), 1)
                    
                return self.landmarks_list, self.landmarks_list_no_scaled
            
##############################################################################################################################

