# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-03 09:23:07
# @Email:  cshzxie@gmail.com

import numpy as np
import torch
import transforms3d
from pointnet2_ops import pointnet2_utils


# class Compose(object):
#     def __init__(self, transforms):
#         self.transformers = []
#         for tr in transforms:
#             transformer = eval(tr['callback'])
#             parameters = tr['parameters'] if 'parameters' in tr else None
#             self.transformers.append({
#                 'callback': transformer(parameters),
#                 'objects': tr['objects']
#             })  # yapf: disable
#
#     def __call__(self, data):
#         for tr in self.transformers:
#             transform = tr['callback']
#             objects = tr['objects']
#             rnd_value = np.random.uniform(0, 1)
#             if transform.__class__ in [NormalizeObjectPose]:
#                 data = transform(data)
#             else:
#                 for k, v in data.items():
#                     if k in objects and k in data:
#                         if transform.__class__ in [
#                             RandomMirrorPoints
#                         ]:
#                             data[k] = transform(v, rnd_value)
#                         else:
#                             data[k] = transform(v)
#
#         return data


class Compose(object):
    def __init__(self, transforms):
        # Define a dictionary mapping callback names to their actual classes
        available_transforms = {
            'RandomRotatePoints': RandomRotatePoints,
            'UpSamplePoints': UpSamplePoints,
            'RandomSamplePoints': RandomSamplePoints,
            'NormalizeObjectPose': NormalizeObjectPose,
            'RandomMirrorPoints': RandomMirrorPoints,
            'RandomTranslatePoints': RandomTranslatePoints,
            'RandomScalePoints': RandomScalePoints,
            'ToTensor': ToTensor
            # Add other transforms here as needed
        }

        self.transformers = []

        for tr in transforms:
            # Get the transformer class from the dictionary instead of eval
            transformer = available_transforms[tr['callback']]
            parameters = tr['parameters'] if 'parameters' in tr else None

            # Store the transformer and its target objects
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)  # Random value for certain transforms

            # For RandomRotatePoints, RandomTranslatePoints, and RandomScalePoints,
            # we apply the same transformation to both 'partial' and 'gt'
            if isinstance(transform, (RandomRotatePoints, RandomTranslatePoints, RandomScalePoints)):
                # Generate random parameters for rotation, translation, or scale
                rnd_params = None
                if isinstance(transform, RandomRotatePoints):
                    rnd_params = [
                        np.random.uniform(*transform.roll_range),
                        np.random.uniform(*transform.pitch_range),
                        np.random.uniform(*transform.yaw_range)
                    ]
                elif isinstance(transform, RandomTranslatePoints):
                    rnd_params = [
                        np.random.uniform(*transform.translate_range[0]),
                        np.random.uniform(*transform.translate_range[1]),
                        np.random.uniform(*transform.translate_range[2])
                    ]
                elif isinstance(transform, RandomScalePoints):
                    rnd_params = np.random.uniform(*transform.scale_range)

                # Apply the same transformation to all specified objects
                for obj_name in objects:
                    if obj_name in data:
                        data[obj_name] = transform(data[obj_name], rnd_params)

            elif isinstance(transform, NormalizeObjectPose):
                # Apply transformation to the entire data (all objects)
                data = transform(data)

            else:
                for obj_name in objects:
                    if obj_name in data:
                        # Handle RandomMirrorPoints separately with random value
                        if isinstance(transform, RandomMirrorPoints):
                            data[obj_name] = transform(data[obj_name], rnd_value)
                        else:
                            data[obj_name] = transform(data[obj_name])

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need <= 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr < need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


# class RandomMirrorPoints(object):
#     def __init__(self, parameters):
#         pass
#
#     def __call__(self, ptcloud, rnd_value):
#         trfm_mat = transforms3d.zooms.zfdir2mat(1)
#         trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
#         trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
#         if rnd_value <= 0.25:
#             trfm_mat = np.dot(trfm_mat_x, trfm_mat)
#             trfm_mat = np.dot(trfm_mat_z, trfm_mat)
#         elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
#             trfm_mat = np.dot(trfm_mat_x, trfm_mat)
#         elif rnd_value > 0.5 and rnd_value <= 0.75:
#             trfm_mat = np.dot(trfm_mat_z, trfm_mat)
#
#         ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
#         return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters=None):
        # No parameters needed for this simple mirroring transformation
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)  # Identity matrix (no mirroring)

        # Mirror only along the X-axis with 50% probability
        if rnd_value <= 0.5:
            trfm_mat_x = transforms3d.zooms.zfdir2mat(-1, [1, 0, 0])  # Mirror along X-axis
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)

        # Apply the transformation matrix to the point cloud
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data


# class RandomRotatePoints(object):
#     def __init__(self, parameters):
#         self.angle_range = parameters['angle_range']  # The range of possible rotation angles
#
#     def __call__(self, ptcloud):
#         angle = np.random.uniform(*self.angle_range)  # Randomly select an angle
#         rotation_matrix = transforms3d.axangles.axangle2mat([0, 0, 1], angle)  # Create a rotation matrix
#         ptcloud[:, :3] = np.dot(ptcloud[:, :3], rotation_matrix.T)  # Apply the rotation
#         return ptcloud


class RandomRotatePoints(object):
    def __init__(self, parameters):
        self.roll_range = parameters['roll_angle']
        self.pitch_range = parameters['pitch_angle']
        self.yaw_range = parameters['yaw_angle']

    def generate_rotation_matrix(self, roll=None, pitch=None, yaw=None):
        # If angles are not provided, generate them randomly
        if roll is None:
            roll = np.random.uniform(*self.roll_range)
        if pitch is None:
            pitch = np.random.uniform(*self.pitch_range)
        if yaw is None:
            yaw = np.random.uniform(*self.yaw_range)

        # Create rotation matrices for each axis
        roll_matrix = transforms3d.axangles.axangle2mat([1, 0, 0], roll)
        pitch_matrix = transforms3d.axangles.axangle2mat([0, 1, 0], pitch)
        yaw_matrix = transforms3d.axangles.axangle2mat([0, 0, 1], yaw)

        # Combine the rotation matrices
        rotation_matrix = np.dot(roll_matrix, np.dot(pitch_matrix, yaw_matrix))
        return rotation_matrix

    def __call__(self, ptcloud, rnd_angles=None):
        if rnd_angles is None:
            # If no precomputed angles, generate a new rotation matrix
            rotation_matrix = self.generate_rotation_matrix()
        else:
            # Use the precomputed angles to generate a rotation matrix
            roll, pitch, yaw = rnd_angles
            rotation_matrix = self.generate_rotation_matrix(roll, pitch, yaw)

        # Apply the rotation matrix to the point cloud
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], rotation_matrix.T)
        return ptcloud


class RandomTranslatePoints(object):
    def __init__(self, parameters):
        self.translate_range = parameters['translate_range']  # A 3D vector specifying the range for x, y, z translation

    def __call__(self, ptcloud, rnd_translate=None):
        if rnd_translate is None:
            # Generate random translation for x, y, z
            rnd_translate = [
                np.random.uniform(*self.translate_range[0]),
                np.random.uniform(*self.translate_range[1]),
                np.random.uniform(*self.translate_range[2])
            ]

        # Apply translation to the point cloud
        ptcloud[:, :3] += rnd_translate  # Apply translation to the x, y, z coordinates
        return ptcloud


class RandomScalePoints(object):
    def __init__(self, parameters):
        self.scale_range = parameters['scale_range']  # A range specifying the scaling factor (min, max)

    def __call__(self, ptcloud, rnd_scale=None):
        if rnd_scale is None:
            # Generate a random scaling factor
            rnd_scale = np.random.uniform(*self.scale_range)

        # Apply scaling to the point cloud
        ptcloud[:, :3] *= rnd_scale  # Apply scaling to the x, y, z coordinates
        return ptcloud


class RandomNoisePoints(object):
    def __init__(self, parameters):
        self.noise_std = parameters['noise_std']  # The standard deviation of the noise

    def __call__(self, ptcloud):
        noise = np.random.normal(scale=self.noise_std, size=ptcloud[:, :3].shape)  # Generate noise
        ptcloud[:, :3] += noise  # Apply the noise
        return ptcloud