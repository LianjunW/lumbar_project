# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from matplotlib import pyplot as plt
from dataset.aug_transforms import  aug_transform

KEYPOINT_COLOR = (0, 255, 0) # Green

import albumentations as A
a_transform = A.Compose([A.OneOf([
    A.IAAAdditiveGaussianNoise(),
    A.GaussNoise(),
], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3)])


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    cv2.imshow("dd",image)
    cv2.waitKey(100)

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        # print(self.data_format)

        # if self.data_format == 'zip':
        #     from utils import zipreader
        #     data_numpy = zipreader.imread(
        #         image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        #     data_numpy = cv2.imread(
        #         image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = self.datas[idx]
        # print("Input shape",data_numpy.shape)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        #Rand crop
        if random.random() < 0.3:
            rand_x = random.randint(0,15)
            rand_y = random.randint(0,10)
            data_numpy = data_numpy[rand_y:,rand_x:]
            joints[:,0] = joints[:,0] - rand_x
            joints[:,1] = joints[:,1] - rand_y
        if random.random() < 0.3:
            rand_w = random.randint(0,12) + data_numpy.shape[1]
            rand_h = random.randint(0,12) + data_numpy.shape[0]
            convas = np.zeros(shape=(rand_h,rand_w),dtype=data_numpy.dtype)
            rand_x = random.randint(0,rand_w - data_numpy.shape[1])
            rand_y = random.randint(0,rand_h - data_numpy.shape[0])
            convas[rand_y:rand_y + data_numpy.shape[0],rand_x:rand_x + data_numpy.shape[1]] = data_numpy
            data_numpy = convas
            joints[:,0] += rand_x
            joints[:,1] += rand_y

        w_sf = 512.0/data_numpy.shape[1]
        h_sf = 512.0/data_numpy.shape[0]
        data_numpy = cv2.resize(data_numpy,dsize=(512,512))
        joints[:,0] *= w_sf
        joints[:,1] *= h_sf

        # kepoints = []
        # for i in range(self.num_joints):
        #     kepoints.append((joints[i,0],joints[i,1]))
        #
        # a_transform = A.Compose([
        #     A.HorizontalFlip(p=0.3),
        #     A.ShiftScaleRotate(p = 0.3),
        #     A.PadIfNeeded()
        #     A.Resize(512,512),
        #     A.RandomCrop(width=512,height=512),
        #                          A.OneOf([
                                 #     A.HueSaturationValue(0.4),
                                 #     A.RandomBrightness(0.4),]),
                                 # A.Resize(512,512)],keypoint_params=A.KeypointParams(format='xy'))
        # if not self.is_train:
        #     a_transform = A.Compose([
        #         A.Resize(512, 512)], keypoint_params=A.KeypointParams(format='xy'))

        # while(True):
        #     a_transformed = a_transform(image = data_numpy.copy(),keypoints=kepoints)
        #     if(len(a_transformed['keypoints'])):
        #         break
        #     else:
        #         print("continue transform")
        # vis_keypoints(a_transformed['image'],a_transformed['keypoints'])
        # print()
        # for i in range(self.num_joints):
        #     joints[i,0] = kepoints[i][0]
        #     joints[i,1] = kepoints[i][1]
        #     kepoints.append((joints[i,0],joints[i,1]))
        #
        # input = a_transformed['image']
        if self.is_train:
            # sf = self.scale_factor
            # rf = self.rotation_factor
            # s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            # r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
            #     if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1]
                joints[:, 0] = data_numpy.shape[1] - joints[:, 0] - 1



            if random.random() < 0.4:
                max_rotate_degree = 10
                prob = random.random()
                degree = (prob - 0.5) * 2 * max_rotate_degree
                h, w = data_numpy.shape
                img_center = (w / 2, h / 2)
                R = cv2.getRotationMatrix2D(img_center, degree, 1)

                abs_cos = abs(R[0, 0])
                abs_sin = abs(R[0, 1])

                bound_w = int(h * abs_sin + w * abs_cos)
                bound_h = int(h * abs_cos + w * abs_sin)

                dsize = (bound_w, bound_h)

                R[0, 2] += dsize[0] / 2 - img_center[0]
                R[1, 2] += dsize[1] / 2 - img_center[1]
                tmp_input = cv2.warpAffine(data_numpy, R, dsize=dsize,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                x_start = random.randint(0,dsize[0] - 512)
                y_start = random.randint(0,dsize[1] - 512)
                for i in range(joints.shape[0]):
                    tmp_pt = self._rotate(joints[i,:],R)
                    joints[i,0] = tmp_pt[0] - x_start
                    joints[i,1] = tmp_pt[1] - y_start
                input = tmp_input[y_start:y_start+512,x_start:x_start+512]
            else:
                input = data_numpy.copy()
                input = input[:,:,np.newaxis]

            if random.random() < 0.5:
                input = aug_transform(image = input)

        #
        #         # joints, joints_vis = fliplr_joints(
        #         #     joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
        #         c[0] = data_numpy.shape[1] - c[0] - 1

            # trans = get_affine_transform(c, s, r, self.image_size)
            # input = cv2.warpAffine(
            #     data_numpy,
            #     trans,
            #     (int(self.image_size[0]), int(self.image_size[1])),
            #     flags=cv2.INTER_LINEAR)
            # for i in range(self.num_joints):
            #     if joints_vis[i, 0] > 0.0:
            #         joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        else:
            input = data_numpy.copy()
            input = input[:,:,np.newaxis]
        # print(input.shape)
        if self.transform:
            input = self.transform(input)
        # input = input[np.newaxis,:,:]
        # input = input[:,:,np.newaxis]


        target, target_weight = self.generate_target(joints, joints_vis)

        # print(target.shape)
        target = torch.from_numpy(target)

        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3
            show_target = np.zeros(self.heatmap_size[1],self.heatmap_size[0])
            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                # cv2.imshow("test",target[joint_id])
                # cv2.waitKey(10)
                show_target = show_target + target[joint_id]
                # print(target[joint_id])

        # cv2.imshow("test",show_target)
        # cv2.waitKey(50)
        return target, target_weight
    def _rotate(self, point, R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]