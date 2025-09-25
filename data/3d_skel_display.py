# -*- coding:utf-8 -*-

# -----------------------------------
# 3D Skeleton Display
# Author: DuohanL
# Date: 2020/2/10 @home
# REPO: https://github.com/XiaoCode-er/3D-Skeleton-Display
# -----------------------------------

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image


trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


# Show 3D Skeleton with Axes3D for NTU RGB+D
class Draw3DSkeleton(object):

	def __init__(
			self,
			file_path,
			save_path=None,
			init_horizon=-45,
			init_vertical=20,
			x_rotation=None,
			y_rotation=None
 ):

		self.file_path = file_path
		self.save_path = save_path

		if not os.path.exists(self.save_path):
			os.mkdir(self.save_path)

		self.xyz = self.read_xyz()

		self.init_horizon = init_horizon
		self.init_vertical = init_vertical

		self.x_rotation = x_rotation
		self.y_rotation = y_rotation

	def _read_skeleton(self):
		with open(self.file_path, 'r') as f:
			skeleton_sequence = {}
			skeleton_sequence['numFrame'] = int(f.readline())
			skeleton_sequence['frameInfo'] = []
			print(f"There are {skeleton_sequence['numFrame']} skeletons to visualise")
			for t in range(skeleton_sequence['numFrame']):
				frame_info = {}
				frame_info['numBody'] = int(f.readline())
				frame_info['bodyInfo'] = []
				for m in range(frame_info['numBody']):
					body_info = {}
					body_info_key = [
						'bodyID', 'clipedEdges', 'handLeftConfidence',
						'handLeftState', 'handRightConfidence', 'handRightState',
						'isResticted', 'leanX', 'leanY', 'trackingState'
					]
					body_info = {
						k: float(v)
						for k, v in zip(body_info_key, f.readline().split())
					}
					body_info['numJoint'] = int(f.readline())
					body_info['jointInfo'] = []
					for v in range(body_info['numJoint']):
						joint_info_key = [
							'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
							'orientationW', 'orientationX', 'orientationY',
							'orientationZ', 'trackingState'
						]
						joint_info = {
							k: float(v)
							for k, v in zip(joint_info_key, f.readline().split())
						}
						body_info['jointInfo'].append(joint_info)
					frame_info['bodyInfo'].append(body_info)
				skeleton_sequence['frameInfo'].append(frame_info)
		return skeleton_sequence

	def read_xyz(self, max_body=2, num_joint=25):
		seq_info = self._read_skeleton()
		data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # (3,frame_nums,25 2)
		for n, f in enumerate(seq_info['frameInfo']):
			for m, b in enumerate(f['bodyInfo']):
				for j, v in enumerate(b['jointInfo']):
					if m < max_body and j < num_joint:
						data[:, n, j, m] = [v['x'], v['y'], v['z']]
					else:
						pass
		return data

	def _normal_skeleton(self, data):
		#  use as center joint
		center_joint = data[0, :, 0, :]

		center_jointx = np.mean(center_joint[:, 0])
		center_jointy = np.mean(center_joint[:, 1])
		center_jointz = np.mean(center_joint[:, 2])

		center = np.array([center_jointx, center_jointy, center_jointz])
		data = data - center

		return data

	def _rotation(self, data, alpha=0, beta=0):
		# rotate the skeleton around x-y axis
		r_alpha = alpha * np.pi / 180
		r_beta = beta * np.pi / 180

		rx = np.array([[1, 0, 0],
					   [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
					   [0, np.sin(r_alpha), np.cos(r_alpha)]]
					  )

		ry = np.array([
			[np.cos(r_beta), 0, np.sin(r_beta)],
			[0, 1, 0],
			[-1 * np.sin(r_beta), 0, np.cos(r_beta)],
		])

		r = ry.dot(rx)
		data = data.dot(r)

		return data
	
	def draw_skeleton_frames(self, frame_indices, adj=None):
		"""
		Save 3D skeleton plots for specific frames as images.

		Args:
			frame_indices (list): List of frame indices to process.
			adj (list): list of value that are used to add cmap colors to scatter plot
				points that represent the joints of the skeleton.
		"""

		fig = plt.figure(figsize=(10,8))
		ax = fig.add_subplot(111, projection='3d')

		data = np.transpose(self.xyz, (3, 1, 2, 0))

		# Apply rotation if specified
		if (self.x_rotation is not None) or (self.y_rotation is not None):
			if self.x_rotation > 180 or self.y_rotation > 180:
				raise Exception("Rotation angle should be less than 180.")
			data = self._rotation(data, self.x_rotation, self.y_rotation)

		# Normalize the skeleton data
		data = self._normal_skeleton(data)

		for frame_idx in frame_indices:
			ax.cla()
			ax.set_title(f"Frame: {frame_idx}")

			x = data[0, frame_idx, :, 0]
			y = data[0, frame_idx, :, 1]
			z = data[0, frame_idx, :, 2]

			ax.set_xlim3d([-0.9, 0.9])
			ax.set_ylim3d([-0.9, 0.9])
			ax.set_zlim3d([-0.8, 0.8])

			# Plot the skeleton
			for part in body:
				x_plot = x[part]
				y_plot = y[part]
				z_plot = z[part]
				ax.plot(
					x_plot,
					y_plot,
					z_plot,
					color='b',
					# marker='o',
					# markerfacecolor='r'
				)

			ax.scatter(x,y,z, s=adj*1000, c=adj, cmap='Reds', marker='o')

			ax.set_xlabel('X')
			ax.set_ylabel('Z')
			ax.set_zlabel('Y')
			ax.set_axis_off()

			# Save the plot as an image
			save_path = os.path.join(self.save_path, f"frame_{frame_idx}.png")
			plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
			# plt.close(fig)

			# Crop the saved image
			with Image.open(save_path) as img:
				cropped_img = img.crop((100, 100, img.width - 100, img.height - 100))  # Adjust crop box as needed
				cropped_img.save(save_path)  # Overwrite the original image
			print(f"\t-- Processed frame {frame_idx} - {save_path}")


if __name__ == '__main__':
	from graph.ntu_rgb_d import Graph

	# graph = Graph()
	# A = graph.A.mean(0).mean(1)

	dataset='ntu'
	model_type = 'base'
	evaluation="CV"
	dilation=3
	gcn_number=1

	# This value is created by passing samples through the model and saving the
	# adjacency matrix gradients
	grad_importance = np.load(f"grad_importance_{model_type}.npy")

	A = grad_importance.mean(axis=0).mean(axis=0)
	# A_norm = (A - np.min(A)) / (np.max(A) - np.min(A))
	A_norm = (A) / (0.2)

	# 26 - hopping, 42 - staggering, 31 - point at something, 24 - kicking
	# skel_name = 'S009C003P019R001A026.skeleton' # Hopping (-70, 90) rotation
	skel_name = 'S009C001P019R001A024.skeleton'
	rotation = {'x_rotation': -90, 'y_rotation': 150}
	sk = Draw3DSkeleton(
		file_path=f"../Datasets/NTU_RGBD/nturgb+d_skeletons/{skel_name}",
		save_path='./data/visualisations/skeletons',
		init_horizon=-45,
		init_vertical=20,
		**rotation
	)

	# Define frames to process and save
	frames_to_save = [0,10,20,30,40,50]
	# frames_to_save = [0,30,60,90,120]
	sk.draw_skeleton_frames(
		frame_indices=frames_to_save,
		adj=A_norm
	)
