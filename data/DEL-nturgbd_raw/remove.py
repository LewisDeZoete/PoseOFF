# -------------------------------------------
# NTU_RGBD
# -------------------------------------------

with open('ntu_rgbd.txt', 'r') as f:
    all_ntu = f.readlines()
    for line_number, line in enumerate(all_ntu):
        line = line.split('_')[0]
        all_ntu[line_number] = line

print(f'Total files NTU: {len(all_ntu)}')

with open('NTU_RGBD_samples_with_missing_skeletons.txt', 'r') as f:
    all_missing = f.readlines()
    for line_number, line in enumerate(all_missing):
        line = line.split('\n')[0]
        all_missing[line_number] = line

print(f'Total missing skeletons NTU: {len(all_missing)}')

for missing in all_missing:
    if missing in all_ntu:
        all_ntu.remove(missing)

print(f'Total available skeletons NTU: {len(all_ntu)}\n')

with open('ntu_rgbd-available.txt', 'w') as f:
    for line in all_ntu:
        f.write(line + '\n')



# -------------------------------------------
# NTU_RGBD120
# -------------------------------------------

with open('ntu_rgbd120.txt', 'r') as f:
    all_ntu = f.readlines()
    for line_number, line in enumerate(all_ntu):
        line = line.split('_')[0]
        all_ntu[line_number] = line

print(f'Total files NTU120: {len(all_ntu)}')

with open('NTU_RGBD120_samples_with_missing_skeletons.txt', 'r') as f:
    all_missing = f.readlines()
    for line_number, line in enumerate(all_missing):
        line = line.split('\n')[0]
        all_missing[line_number] = line

print(f'Total missing skeletons NTU: {len(all_missing)}')

for missing in all_missing:
    if missing in all_ntu:
        all_ntu.remove(missing)

print(f'Total available skeletons NTU: {len(all_ntu)}')

with open('ntu_rgbd120-available.txt', 'w') as f:
    for line in all_ntu:
        f.write(line + '\n')

# import cv2
# import numpy as np
# from decord import VideoReader, cpu

# def compute_normal_flow(img1, img2):
#     """
#     Computes the local normal flow from two consecutive images.
#     :param img1: First grayscale image (time t)
#     :param img2: Second grayscale image (time t+dt)
#     :return: Normal flow vectors (u_n, v_n)
#     """
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     # Compute spatial gradients (Ix, Iy) and temporal gradient (It)
#     Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
#     Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
#     It = cv2.subtract(img2.astype(np.float64), img1.astype(np.float64))
   
#     # Compute normal flow components
#     magnitude = np.sqrt(Ix**2 + Iy**2) + 1e-6  # Avoid division by zero
#     u_n = -It * (Ix / magnitude)
#     v_n = -It * (Iy / magnitude)
   
#     return u_n, v_n

# def visualize_normal_flow(img, u_n, v_n, step=10):
#     """
#     Visualizes the normal flow on the image.
#     :param img: Original image
#     :param u_n: Normal flow in x direction
#     :param v_n: Normal flow in y direction
#     :param step: Step size for drawing the flow vectors
#     """
#     h, w, _ = img.shape
#     # vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
   
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             end_x = int(x + u_n[y, x] * 5)
#             end_y = int(y + v_n[y, x] * 5)
#             cv2.arrowedLine(img, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)
   
#     return img


# video = '../Datasets/UCF-101/BaseballPitch/v_BaseballPitch_g09_c01.avi'
# vr = VideoReader(video, ctx=cpu(0))
# img1 = vr[36].asnumpy()
# img2 = vr[37].asnumpy()
# # # Load two consecutive grayscale images
# # img1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
# # img2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)


# if img1 is None or img2 is None:
#     raise ValueError("Error loading images")

# # Compute normal flow
# u_n, v_n = compute_normal_flow(img1, img2)

# # Visualize normal flow
# vis_flow = visualize_normal_flow(img1, u_n, v_n)
# vis_flow = cv2.cvtColor(vis_flow, cv2.COLOR_BGR2RGB)

# # # Show result
# cv2.imwrite('normal_flow.png', vis_flow)
# # cv2.imshow('Normal Flow', vis_flow)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()