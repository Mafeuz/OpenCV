import numpy as np
import cv2
import time

##############################################################################################################
def Display_Multiple_Images(images_array, scale=0.5):
        
    ##############################################################################################################
    # Resize images based on the shape of the first image:
    for i in range(len(images_array)):
        for j in range(len(images_array[i])):
            images_array[i][j] = cv2.resize(images_array[i][j], (images_array[0][0].shape[1], images_array[0][0].shape[0]))
    
    # Images Rescaling and Convert Gray-scale to BGR if needed:
    for i in range(len(images_array)):
        for j in range(len(images_array[i])):
            images_array[i][j] = cv2.resize(images_array[i][j], (0, 0), None, scale, scale)
            
            if (len(images_array[i][j].shape) == 2):
                images_array[i][j] = cv2.cvtColor(images_array[i][j], cv2.COLOR_GRAY2BGR)
                
            if (len(images_array[i][j].shape) == 3):
                if (images_array[i][j].shape[2] == 1):
                    images_array[i][j] = cv2.cvtColor(images_array[i][j], cv2.COLOR_GRAY2BGR)
    
    ##############################################################################################################
    # Let's equalize rows number of images:
    lens = [1]*len(images_array)
    
    for i in range(len(images_array)):
        lens[i] = len(images_array[i])
    
    # get max_len to add a blank image if necessary:
    max_len = np.max(lens)
    
    for i in range(len(images_array)):
        if (len(images_array[i]) < max_len):
            blank_image = np.zeros((images_array[0][0].shape[0], images_array[0][0].shape[1] , 3),dtype=np.uint8)
        
            for j in range(max_len - len(images_array[i])):
                images_array[i].append(blank_image)
                
    ##############################################################################################################
    # Stacking images:
    for i in range(len(images_array)):
        for j in range(len(images_array[i])):
            if (j == 0):
                h_stack = images_array[i][j]
            if (j > 0):
                h_stack = np.hstack([h_stack, images_array[i][j]])
        
        if (i == 0):                          
            v_stack = h_stack
        if (i > 0):                          
            v_stack = np.vstack([v_stack, h_stack])  
                                 
    return v_stack

##############################################################################################################
def color_filtering(frame, boundaries, max_contrast_output=False):

    # loop over the boundaries
    for (lower, upper) in boundaries:

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8") # Lower color limit
        upper = np.array(upper, dtype = "uint8") # Upper color limit

        mask = cv2.inRange(frame, lower, upper) # mask wit in range of lower to upper
        output_filter = cv2.bitwise_and(frame, frame, mask = mask)
        
        if max_contrast_output:
            output_filter[np.where((output_filter == [0,0,0]).all(axis = 2))] = [255,255,255]
            output_filter[np.where((output_filter != [255,255,255]).all(axis = 2))] = [0,0,0]
            output_filter = cv2.cvtColor(output_filter,cv2.COLOR_BGR2GRAY)
            output_filter = cv2.bitwise_not(output_filter)
        
    return output_filter

##############################################################################################################
def custom_canny(img, blur_kernel_size = (5,5), kernel_size = (3,3), canny_thresh = (100,100), dil_level = 0, ero_level = 0):
    
    # Canny Processing:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, blur_kernel_size, 1)
    canny_img = cv2.Canny(blur_img, canny_thresh[0], canny_thresh[1])
    
    kernel = np.ones(kernel_size)
    
    img_dilation = cv2.dilate(canny_img, kernel, iterations = dil_level)
    img_thresh = cv2.erode(img_dilation, kernel, iterations = ero_level)
    
    return img_thresh

##############################################################################################################

if __name__ == '__main__':
    print('Main')




