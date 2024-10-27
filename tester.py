import torch

poses = torch.load('data/UCF-101/skeleton/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.pt',
                         map_location=torch.device('cpu'))
flows = torch.load('data/UCF-101/flow/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.pt', 
                             map_location=torch.device('cpu'))

num_frames, _, height, width = flows.shape

# Define the kernel size for neighborhood calculations
kernel_size = 3
half_k = kernel_size // 2
threshold = 0.5

stacker = torch.zeros((5,300,17,2))

def get_flow_curl(x,y,half_k,flow):
    # Extract the neighborhood
    neighborhood = flow[
        :, y - half_k : y + half_k + 1, x - half_k : x + half_k + 1
    ]

    # Calculate gradients
    du_dy = neighborhood[..., 0].diff(
        dim=1
    )  # Change in u (x-component) with respect to y
    du_dx = neighborhood[..., 0].diff(
        dim=2
    )  # Change in u (x-component) with respect to x
    dv_dy = neighborhood[..., 1].diff(
        dim=1
    )  # Change in v (y-component) with respect to y
    dv_dx = neighborhood[..., 1].diff(
        dim=2
    )  # Change in v (y-component) with respect to x

    # Calculate divergence
    div = du_dx + dv_dy

    # Calculate curl
    curl = dv_dx - du_dy

    return div, curl

for i, flow in enumerate(flows):
    # Reshape the pose points to be a continuous array
    pose_points = poses[:2,i].reshape(2,34)
    # Get visibility tensor to use as a mask
    vis = poses[2,i].reshape(34)
    vis = vis > threshold
    # Rescale values from (-0.5:0.5) to (0,height-1) since we're using it to index
    pose_points[0] = (pose_points[0]+0.5)*(width-1)
    pose_points[1] = (pose_points[1]+0.5)*(height-1)
    pose_points = pose_points.type(torch.int)

    # Get the divergence and curl for each keypoint
    # for keypoint_num in range(pose_points.shape[1]):
    #     x,y = poses[0,keypoint_num], poses[1,keypoint_num]
        
        # if y < half_k or y >= height - half_k or x < half_k or x >= width - half_k:
        #     pass
            
    # For now just simply getting the flow points
    flow_points = (flows[i,:, pose_points[1], pose_points[0]]*vis).reshape(2,17,2)

print(flow_points.shape)
print(flow_points[0])


# # Iterate over all the flow frame numbers
# for frame_no in range(1, num_frames+1):
#     # Iterate over all people in the whole tensor
#     for person_no in range(pose.shape[3]):
#         # (3, 17)
#         person_key_points = pose[:,frame_no,:, person_no]
#         # Get x,y values for all keypoints for that frame
#         for keypoint_no in range(person_key_points.shape[1]):
#         # x,y,vis = person_key_points[...]
#             x = person_key_points[0,keypoint_no]*width
#             y = person_key_points[1,keypoint_no]*height
#             vis = person_key_points[2,keypoint_no]
#             # If the point is too close to boundaries or the visibility is too low, disregard
#             if y < half_k or y >= height - half_k or x < half_k or x >= width - half_k or vis < threshold:
#                 div, curl = (0,0)
#                 break
            
#             # Extract the neighborhood
#             neighborhood = flow[
#                 :, y - half_k : y + half_k + 1, x - half_k : x + half_k + 1
#             ]

#             # Calculate gradients
#             du_dy = neighborhood[..., 0].diff(
#                 dim=1
#             )  # Change in u (x-component) with respect to y
#             du_dx = neighborhood[..., 0].diff(
#                 dim=2
#             )  # Change in u (x-component) with respect to x
#             dv_dy = neighborhood[..., 1].diff(
#                 dim=1
#             )  # Change in v (y-component) with respect to y
#             dv_dx = neighborhood[..., 1].diff(
#                 dim=2
#             )  # Change in v (y-component) with respect to x

#             # Calculate divergence
#             divergence = du_dx + dv_dy
#             divergences.append(divergence.mean().item())

#             # Calculate curl
#             curl = dv_dx - du_dy
#             curls.append(curl.mean().item())