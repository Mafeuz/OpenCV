######################################################################################################################################
### OpenCV Basic Image Transformations (Affine Combination, Translation, Rotation) ###################################################
######################################################################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

######################################################################################################################################
def affine_transform(img, pts1, pts2):
    
    h, w , c = img.shape

    M = cv2.getAffineTransform(pts1, pts2)
    output = cv2.warpAffine(img, M, (w, h))
    
    return output

######################################################################################################################################
def img_translation(img, dx, dy):
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    
    shifted = cv2.warpAffine(img, M, (0,0))
    
    return shifted

######################################################################################################################################
def img_rotation(img, angle, pivot, keep_full_img=False):
    
    (h, w) = img.shape[:2]
    (cX, cY) = pivot
    
    if keep_full_img:
        
        # Sum 1 to avoid [0,0,0] pixels in the source img
        # because later full black pixels will be taken out
        img = img + 1
        
        # Create space for keeping full img with stack (4 times img size and move it to the center):
        bboard = np.zeros_like(img)
        dbboard = np.hstack([bboard, bboard])
        stack = np.hstack([img, bboard])
        stack = np.vstack([stack, dbboard])
        
        # Move it to the center:
        M = np.float32([[1, 0, img.shape[1]//2],
                        [0, 1, img.shape[0]//2]])
        
        (h, w) = stack.shape[:2]
        (cX, cY) = (stack.shape[1]//2, stack.shape[0]//2)
        
        img = cv2.warpAffine(stack, M, (0,0))
                
        # Rotate new img:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Get img region extremes and move desired region to (0,0)
        loc = np.where((rotated != [0,0,0]).all(axis = 2))
                        
        extTop = np.min(loc[0])
        extLeft = np.min(loc[1])
        
        M = np.float32([[1, 0, -extLeft],
                        [0, 1, -extTop]])
        
        rotated = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]))
        
        loc = np.where((rotated != [0,0,0]).all(axis = 2))
        
        extRight = np.max(loc[0])
        extBot = np.max(loc[1])
        
        rotated = rotated[:extRight,:extBot,:] -1
    
    else:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
    
    return rotated

######################################################################################################################################
