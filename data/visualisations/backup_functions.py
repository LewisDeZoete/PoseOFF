#!/usr/bin/env python3


def write_videos(videos):
    size = (videos[0].shape[2], videos[0].shape[1])
    flow_video = cv2.VideoWriter('data/visualisations/videos/flow_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
    flow_skel_video = cv2.VideoWriter('data/visualisations/videos/flow_skels_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)

    RGB_video = cv2.VideoWriter('data/visualisations/videos/RGB_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
    RGB_skel_video = cv2.VideoWriter('data/visualisations/videos/RGB_skels_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
    
    flowpose_video = cv2.VideoWriter('data/visualisations/videos/flowpose_video.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    current_frame_index = 0
    flowpose = flow_pose_sample(flow_full, poses, window_size=19) # T, MV, C
    while True:
        # Get the flow and RGB frames
        flow_frame = videos[0][current_frame_index]
        RGB_frame = videos[1][current_frame_index]

        # frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        flow_frame = flow2image(flow_frame)
        RGB_frame = cv2.cvtColor(RGB_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Write normal videos
        flow_video.write(flow_frame.astype(np.uint8))
        RGB_video.write(RGB_frame.astype(np.uint8))

        # Draw skeletons
        flow_skel_frame = draw_skel(flow_frame, pose=poses[current_frame_index])
        RGB_skel_frame = draw_skel(RGB_frame, pose=poses[current_frame_index])

        # Write skeleton videos
        flow_skel_video.write(flow_skel_frame.astype(np.uint8))
        RGB_skel_video.write(RGB_skel_frame.astype(np.uint8))

        # Draw flowpose
        flowpose_frame = draw_flowpose(flowpose_frame=flowpose[current_frame_index],
                                       pose_frame=poses[current_frame_index],
                                       frame=flow_frame,
                                       window_size=19)
        flowpose_video.write(flowpose_frame.astype(np.uint8))
        
        current_frame_index += 1
        if current_frame_index >= get_frame_count(videos[0]):
            break
    
    flow_video.release()
    flow_skel_video.release()
    RGB_video.release()
    RGB_skel_video.release()
    flowpose_video.release()


def flow_pose_sample(flows, poses, window_size=5):
    """
    Samples the optical flow in windows surrounding the pose keypoints.
    flows: (T C H W)
    poses: (T (M V) C)
    Returns array of shape:
        ((window_size**2)*2,
        frames,
        keypoints,
        num_people)
    """
    half_k=window_size//2

    # Get the shape of the input tensors
    num_frames, _, height, width = flows.shape
    num_pose_frames, keypoints, channels = poses.shape

    # The keypoints are already scaled
    pose_points = np.nan_to_num(poses, nan=0).astype(int)

    # Prepare tensor to stack flow windows
    stacker = np.zeros((num_pose_frames, keypoints, window_size**2*2,))
    print(stacker.shape)

    # Create a grid of valid indices (filter out points close to the image border)
    valid_indices = ((pose_points[:, :, 0] >= half_k) & (pose_points[:, :, 0] < width - half_k) &
                        (pose_points[:, :, 1] >= half_k) & (pose_points[:, :, 1] < height - half_k))

    # Loop through the frames and sample flow in window around each valid keypoint
    for frame_no, flow in enumerate(flows):
        for keypoint_num in range(keypoints):
            if valid_indices[frame_no, keypoint_num]:
                x, y = pose_points[frame_no, keypoint_num, 0], pose_points[frame_no, keypoint_num, 1]
                # Get the window of optical flow
                flow_window = flow[:, y - half_k : y + half_k + 1, x - half_k : x + half_k + 1]


                stacker[frame_no, keypoint_num, :] = flow_window.flatten()
    return stacker

if __name__=="__main__":
    pass
    # quick_view(poseoff_frame)
    # quit()

    # # First, draw the poseoff arrows on the full frame (may take time)
    # frame = draw_optical_flow_arrows(flow_frame, frame=rgb_frame, step=10, scale=3, threshold=0.0)
    # full_frame = draw_optical_flow_arrows(flow_frame, step=20, scale=3)
    # frame_transparent = np.zeros((H, W, 4), dtype=np.uint8)

    # # Then, we draw the skeleton lines we want
    # frame = draw_bones(frame, pose)
    # full_frame = draw_bones(full_frame, pose)
    # frame_transparent = draw_bones(frame_transparent, pose)

    # # Finally, draw the skeleton keypoints themselves
    # frame = draw_skel(frame, pose, person_num=0, skip_points=[4,5], debug=False)
    # full_frame = draw_skel(full_frame, pose, debug=False)
    # frame_transparent = draw_skel(frame_transparent, pose, debug=False)
