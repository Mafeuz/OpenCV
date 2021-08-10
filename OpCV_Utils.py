import numpy as np
import matplotlib.pyplot as plt
import cv2

######################################################################################################################################
def show_single_image_plt(img, title, fig_size=(15,15), show_axis=False):
    fig, axis = plt.subplots(figsize = fig_size)
  
    axis.imshow(img)
    axis.set_title(title, fontdict = {'fontsize': 22, 'fontweight': 'medium'})
    
    if not show_axis:
        axis.axis('off')
        
    pass

######################################################################################################################################
def show_multiple_images_plt(images_array, titles_array, fig_size = (15,15), show_axis=False):
    # Function for outputing plt subplots from images (RGB).
    # Each row of images must have the same number of elements as the others.
    # array form: [row1,row2,...rowN], row = [element1, element2,...elementN]
    
    if (len(images_array) > 1) & (len(images_array[0]) > 1):
        fig, axis = plt.subplots(len(images_array), len(images_array[0]), figsize = fig_size)

        for i in range(len(images_array)):
            for j in range(len(images_array[0])):
                axis[i][j].imshow(images_array[i][j])
                axis[i][j].set_anchor('NW')
                axis[i][j].set_title('{}'.format(titles_array[i][j]), fontdict = {'fontsize': 15, 'fontweight': 'medium'}, pad = 10)
                if not show_axis:
                    axis[i][j].axis('off')

    if (len(images_array) == 1):
        fig, axis = plt.subplots(1, len(images_array[0]), figsize = fig_size)
        for j in range(len(images_array[0])):
            axis[j].imshow(images_array[0][j])
            axis[j].set_anchor('NW')
            axis[j].set_title('{}'.format(titles_array[0][j]), fontdict = {'fontsize': 15, 'fontweight': 'medium'}, pad = 10)
            if not show_axis:
                axis[j].axis('off')

    if (len(images_array[0]) == 1):
        fig, axis = plt.subplots(len(images_array), 1, figsize = fig_size)
        for j in range(len(images_array)):
            axis[j].imshow(images_array[j][0])
            axis[j].set_anchor('NW')
            axis[j].set_title('{}'.format(titles_array[j][0]), fontdict = {'fontsize': 15, 'fontweight': 'medium'}, pad = 10)
            if not show_axis:
                axis[j].axis('off')
            
    pass

######################################################################################################################################
def stack_multiple_images(images_array, sep_lines=False, scale=0.5):
    # Function for rescaling and stacking cv2 BGR images together.
    # array form: [row1,row2,...rowN], row = [element1, element2,...elementN]
        
    ##################################################################################################################
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
    
    ##################################################################################################################
    # Let's equalize rows number of images:
    lens = [1]*len(images_array)
    
    for i in range(len(images_array)):
        lens[i] = len(images_array[i])
    
    # get max_len to add a blank image if necessary:
    max_len = np.max(lens)
    
    for i in range(len(images_array)):
        if (len(images_array[i]) < max_len):
            blank_image = np.ones((images_array[0][0].shape[0], images_array[0][0].shape[1] , 3),dtype=np.uint8)*150
        
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
                      
    ##############################################################################################################
    # Paint Separation Lines:
    if sep_lines:
        
        # Horizontal Lines:
        if (len(images_array) > 0):
            for i in range(len(images_array)-1):
                cv2.line(v_stack, (0, (i+1)*(v_stack.shape[0]//len(images_array))), 
                        (v_stack.shape[1], (i+1)*(v_stack.shape[0]//len(images_array))), (255, 0, 0), 2)
                
        # Vertical Lines:
        for i in range(max_len-1):
            cv2.line(v_stack, ((i+1)*(v_stack.shape[1]//max_len), 0), 
                     ((i+1)*(v_stack.shape[1]//max_len), v_stack.shape[0]), (255, 0, 0), 2)
            
    ##############################################################################################################
                                  
    return v_stack

######################################################################################################################################
def color_filtering(img, boundaries, binarization=False):

    # loop over the boundaries
    for (lower, upper) in boundaries:

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8") # Lower color limit
        upper = np.array(upper, dtype = "uint8") # Upper color limit

        mask = cv2.inRange(img, lower, upper) # mask wit in range of lower to upper
        output_filter = cv2.bitwise_and(img, img, mask = mask)
        
        # binarization:
        if binarization: 
            pass
         
    return output_filter

######################################################################################################################################
def custom_canny(img, blur_kernel_size = (5,5), kernel_size = (3,3), canny_thresh = (100,100), order = 1, dil_level = 0, ero_level = 0):
    # Canny Processing:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, blur_kernel_size, 1)
    canny_img = cv2.Canny(blur_img, canny_thresh[0], canny_thresh[1])
    
    kernel = np.ones(kernel_size)
    
    if (order == 1):
        img_dilation = cv2.dilate(canny_img, kernel, iterations = dil_level)
        img_thresh = cv2.erode(img_dilation, kernel, iterations = ero_level)
        
    if (order == -1):
        img_erosion = cv2.erode(canny_img, kernel, iterations = ero_level)
        img_thresh = cv2.dilate(img_erosion, kernel, iterations = dil_level)
    
    return img_thresh

######################################################################################################################################
def reorder_4points(points):
    # redorder 4 points ref to their coordinates
    # points format = np.array([[[p1x, p1y]], [[p2x, p2y]], [[p3x, p3y]], [[p4x, p4y]]])
    reordered_points = np.zeros_like(points)
    points = points.reshape((4,2))
    
    add = points.sum(1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    
    diff = np.diff(points, axis = 1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    
    return reordered_points

######################################################################################################################################
def img_warping_ref_obj(img, ref_points, ref_obj_W, ref_obj_H, pad=0):
    # ref_points format = np.array([[[p1x, p1y]], [[p2x, p2y]], [[p3x, p3y]], [[p4x, p4y]]])
    
    ###################################################
    # Reordering points if needed:
    reordered_points = np.zeros_like(ref_points)
    points = ref_points.reshape((4,2))
    
    add = points.sum(1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    
    diff = np.diff(points, axis = 1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    
    points = reordered_points
    ###################################################
    
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[ref_obj_W,0],[0,ref_obj_H],[ref_obj_W,ref_obj_H]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    img_warped = cv2.warpPerspective(img, matrix, (ref_obj_W, ref_obj_H))
    img_warped = img_warped[pad:img_warped.shape[0]-pad, pad:img_warped.shape[1]-pad]
    
    return img_warped

######################################################################################################################################
def img_homography(img, points1, points2, pad=0):
    # Points must have to be defined in the same resolution for the source and destination images.
    # points format = np.array([[[p1x, p1y]], [[p2x, p2y]], [[p3x, p3y]], [[p4x, p4y]]])
    
    ###################################################
    reordered_points1 = np.zeros_like(points1)
    points1 = points1.reshape((4,2))
    
    add = points1.sum(1)
    reordered_points1[0] = points1[np.argmin(add)]
    reordered_points1[3] = points1[np.argmax(add)]
    
    diff = np.diff(points1, axis = 1)
    reordered_points1[1] = points1[np.argmin(diff)]
    reordered_points1[2] = points1[np.argmax(diff)]
    
    points1 = reordered_points1
    ###################################################
    reordered_points2 = np.zeros_like(points2)
    points2 = points2.reshape((4,2))
    
    add = points2.sum(1)
    reordered_points2[0] = points2[np.argmin(add)]
    reordered_points2[3] = points2[np.argmax(add)]
    
    diff = np.diff(points2, axis = 1)
    reordered_points2[1] = points2[np.argmin(diff)]
    reordered_points2[2] = points2[np.argmax(diff)]
    
    points2 = reordered_points2
    ###################################################
    
    pts1 = np.float32(points1)
    pts2 = np.float32(points2)
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
 
    W = img.shape[1]
    H = img.shape[0]
    img_warp = cv2.warpPerspective(img, matrix, (W,H))
    img_warp = img_warp[pad:img_warp.shape[0]-pad, pad:img_warp.shape[1]-pad]
    
    return img_warp

######################################################################################################################################

if __name__ == '__main__':
    print('Main')




