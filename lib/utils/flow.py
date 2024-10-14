'''
Tools for calculating dense flow trajectories between frames
'''
import numpy as np
import cv2 as cv
from numpy.lib.stride_tricks import sliding_window_view

def getFlow(imPrev, imNew):
    '''
    Calculate optical flow (Farneback method)
    '''
    flow = cv.calcOpticalFlowFarneback(imPrev, imNew, flow=None, pyr_scale=.5, levels=3, winsize=9, iterations=1, poly_n=3, poly_sigma=1.1, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow


def next_points(points_in, flow_img):
    '''
    Based on optical flow and densely sampled image points (on multiple image scales), calculate where the point has moved in the image.

    Parameters:
    - points_in: (list:tuples) of sampled points in initial frame.
    - flow_img: (numpy.ndarray) representing optical flow image.
    '''
    median_flow = np.median(sliding_window_view(flow_img, (3,3,2,)),axis=(2,3,4))
    median_flow = np.stack(
        (np.pad(median_flow[:,:,0], 1),
         np.pad(median_flow[:,:,1], 1)), axis=2
        )
    points_out = []
    for point in points_in:
        new_point = (point[0] + median_flow[point[1],point[0],0], 
                     point[1] + median_flow[point[1],point[0],1])
        
        # test if new points are out of image bounds
        if new_point[0] < 0 or new_point[0] > median_flow.shape[1]:
            points_out.append((1, 1))
            # This is realistically where I would resample new points
            continue
        
        if new_point[1] < 0 or new_point[1] > median_flow.shape[0]:
            points_out.append((1, 1))
            # see comment above
            continue

        points_out.append(new_point)

    return np.array(points_out).astype('i')


def flow_sample(arr:np.ndarray, factor:int, inter: str):
    '''
    Sample points from the optical flow image and return the interpolated value according to downsizing factorisation size (i.e. the higher the factor, the smaller the output image, `inter` refers to the interpolation mathod)

    Parameters:
    - arr: (numpy.ndarray) of image array to sample.
    - factor: (int) downsizin factor, higher factor returns smaller output.
    - inter: str('mean','median','mode') mode of interpolating array chunks.
    
    Returns:
    - sampled_arr: (numpy.ndarray) output array that has been downsized using the interpolation method argument.
    '''
    sampled_arr = np.zeros((int(arr.shape[0]/factor),int(arr.shape[1]/factor)))
    with np.nditer(sampled_arr, flags=['multi_index']) as it:
        for x in it:
            idx = it.multi_index
            if inter == "mean":
                sampled_arr[idx] = arr[int(idx[0]*factor) : int(idx[0]*factor+factor),
                      int(idx[1]*factor) : int(idx[1]*factor+factor)].mean()
                
            elif inter == "max":
                sampled_arr[idx] = arr[int(idx[0]*factor) : int(idx[0]*factor+factor),
                      int(idx[1]*factor) : int(idx[1]*factor+factor)].max()
            
            elif inter == "median":
                sampled_arr[idx] = np.median(arr[int(idx[0]*factor) : int(idx[0]*factor+factor),
                      int(idx[1]*factor) : int(idx[1]*factor+factor)])

    return sampled_arr


def draw_arrows(mag,ang,img,factor,threshold,resize=1, thickness=1):
    '''
    Draw arrows on an image based on magnitude and angle information.

    parameters:
    - mag (numpy.ndarray): 2D array representing the magnitude of the flow.
    - ang (numpy.ndarray): 2D array representing the angle of the flow.
    - img (numpy.ndarray): Input image on which arrows will be drawn.
    - factor (float): downsizing factor to convert flow coordinates to image coordinates.
    - threshold (float): Magnitude threshold for discarding flows below a certain value.
    - resize (float): resize output image by this value.
    - thickness (int): thickness of arrows

    Returns:
    - numpy.ndarray: Image with arrows drawn based on the provided flow information.

    TODO: These flow arrows are being created assuming the image is being resized, add bool arg if the image is being resized.
    '''
    start_points = []
    end_points = []
    
    # Iterate over mag list and append start and end points
    for row in range(mag.shape[0]):
        for col in range(mag.shape[1]):
            # If flow is below a threshold, discard
            if mag[row,col] < threshold:
                continue
            # start points (center of row/column)
            start_points.append((int(col*factor+factor/2)*resize,
                                int(row*factor+factor/2)*resize))
            
            end_points.append(
                (int(col*factor+factor/2+mag[row,col]*np.cos(ang[row,col]))*resize,
                int(row*factor+factor/2+mag[row,col]*np.sin(ang[row,col]))*resize))

    # Set color and line thickness
    color = (0, 0, 255)

    # Draw arrows from the points list
    for point_no in range(len(start_points)):
        # Using cv2.arrowedLine() method  
        # Draw a red arrow line 
        # with thickness of 3 px and tipLength = 0.5 
        start_point = tuple([int(start_points[point_no][0]), int(start_points[point_no][1])])
        end_point = tuple([int(end_points[point_no][0]), int(end_points[point_no][1])])
        img = cv.arrowedLine(img, start_point, end_point,  
                            color, thickness, tipLength = 0.5)
    
    return img


def arrows(data:dict, factor:float=4.0, threshold:float=1.0, labels:np.ndarray=None, classes:np.ndarray=None, resize:float=1.0):
    '''
    Takes a rgb and flow data and draws arrows on RGB image according to magnitude and direction of flow vectors.

    ARGS:
    - data: (dict) containing 'rgb' and 'flow' keywords with np.array arguments representing image and flow data.
    - factor: (int) downscaling factor of flow array (larger number means greater downscaling).
    - threshold: (float) flow magnitudes lower than threshold will not be drawn to canvas.
    - resize (float): resize output image by this value.
    
    Returns:
    - arrows: (np.array) array of shape `(len(data['flow][0]), cfg.image_height, cfg.image_width, channels)`
    '''
    arrow_imgs = []

    for frame_no in range(len(data['flow'][0])):
        # Grab the flow frames and create arrays of the magnitudes and angles
        flow_frame = data['flow'][0][frame_no]
        flow_img = np.array(flow_frame)
        # Convert flow image to magnitude and angles of the flow points
        mag, ang = cv.cartToPolar(flow_img[..., 0], flow_img[..., 1])

        # Sample the points using `flow_sample()`
        mag = flow_sample(mag,factor,"median")
        ang = flow_sample(ang,factor,"median")

        # Create RGB image
        img = np.array(data['rgb'][0,:, frame_no+1, ...])
        img = np.transpose(img, (1,2,0))/255
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (img.shape[1]*resize, img.shape[0]*resize))

        # DRAW ARROW
        arrow_imgs.append(draw_arrows(mag,ang,img,factor,threshold,resize))

    arrow_imgs = np.array(arrow_imgs)

    if labels and classes:
        return arrow_imgs, classes[labels[0]]
    else:
        return arrow_imgs