import numpy as np
import cv2 as cv
import lib.utils as utils
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import torch

def get_im(data, batch_no, frame_no):

    batch = ((data['rgb'][:,:, frame_no, ...])/255).clone().detach()
    batch = torch.permute(batch, (0, 1, 2, 3))

    img = batch[batch_no]
    img = np.array(img)
    img = np.transpose(img, (1, 2, 0))
    img = cv.cvtColor(img,  cv.COLOR_RGB2BGR)

    for frame in data['boxes'][batch_no][frame_no]:
        start = (int(frame[0][0]), int(frame[0][1]))
        end = (int(frame[1][0]), int(frame[1][1]))
        color = (255, 0, 0) 
        thickness = 2

        img = cv.rectangle(img, start, end, color, thickness)
        
    return img



def test_video_multimode(dataloader, classes, new_data=True, results=None):
    if new_data:
        results, labels = next(iter(dataloader))
    if not new_data and not results:
        print('No results to display')
        return        

    modes = ['rgb', 'flow', 'poses']
    img_list = []
    for mode in modes:
        #print(results[mode].shape)
        img = np.array(results[mode][0,:, 0,...])
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_list.append(img.astype('uint8'))

    img_full = np.concatenate((img_list[0], img_list[1], img_list[2]), axis=1)

    while True:
        try:
            cv.imshow(classes[labels[0]], img_full)
        except Exception as e:
            print(e)
            break

        if cv.waitKey(0) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

    return img_full


def test_image(dataloader, classes=None):
    results, labels = next(iter(dataloader))        

    #print(results[mode].shape)
    img = np.array(results['rgb'][0,:, 0,...])
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype('uint8')

    while True:
        try:
            if classes:
                cv.imshow(classes[labels[0]], img)
            else:
                cv.imshow("test image", img)
        except Exception as e:
            print(e)
            break

        if cv.waitKey(0) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

    return img, labels


def video_slideshow(data: dict, datatype: str):
    '''
    Display and move through video using 'n' key.

    Parameters:
    - data (dict): standard data dictionary created by dataloader.
    - datatype (str): key for data dictionary depending on what data you'd like to display. Options include: `['flow','rgb','arrow?']`
    - label (numpy.ndarray): video label (from dataloader)
    '''
    frame_no = 0

    video=np.array(data[datatype][0])

    # shape of the rgb array is wrong (maybe I should change it?)
    if datatype=='rgb':
        video = np.transpose(video, (1,2,3,0))/255
    else:
        video = np.transpose(video, (0,2,3,1))

    while True:
        if frame_no > 15:
            frame_no=0
    
        cv.imshow('debug',cv.cvtColor(video[frame_no],cv.COLOR_BGR2RGB))

        k = cv.waitKey(33)
        if k == ord('q') or k == 27:
            break
        elif k == ord('n'):
            frame_no+=1

    cv.destroyAllWindows()


def arrow_demo_webcam(resize:float=2,downsizing_factor:float=16,camera:int=0):
    '''
    Get a optical flow demo using arrows drawn to represent the optical flow.
    PRESS "Q" TO EXIT
    '''
    cap = cv.VideoCapture(camera)

    _, prev_frame = cap.read()

    if _:
        prev_grey = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    
        while True:
            ret, new_frame = cap.read()
            if ret:
                new_grey = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
                
                # calculate optical flow
                flow = utils.getFlow(prev_grey, new_grey)
                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

                # downsample magnitude and angle arrays
                mag = utils.flow_sample(mag, factor=downsizing_factor,inter='median')
                ang = utils.flow_sample(ang, factor=downsizing_factor, inter='median')

                # resize the image by a factor of `resize`
                new_frame=cv.resize(new_frame, (new_frame.shape[1]*resize,
                                                new_frame.shape[0]*resize))

                # draw flow fields
                #frame = cv.cvtColor(new_frame, cv.COLOR_BGR2RGB)
                arrow_frame = utils.draw_arrows(mag, ang, new_frame, factor=downsizing_factor,
                                                threshold=1, resize=resize)

                # show image frame!
                cv.imshow('window',arrow_frame)


            if cv.waitKey(10) & 0xFF == ord('q'):
                break

            prev_grey = new_grey

    cv.destroyAllWindows()
    cap.release()



# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size