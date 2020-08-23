from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json
import pandas as pd
import glob

import numpy as np
from scipy.io import loadmat, savemat
# from aug_transforms import  aug_transform
from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)
import SimpleITK as sitk
import pandas as pd
import cv2

def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x

class LumbarDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, json_paths, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 11
        self.flip_pairs = [[1, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.datas = []
        self.json_paths = json_paths
        
        self.lab_to_idx = {
                            # "T12":0,
                            "T12-L1":0,
                            "L1":1,
                            "L1-L2":2,
                            "L2":3,
                            "L2-L3":4,
                            "L3":5,
                            "L3-L4":6,
                            "L4":7,
                            "L4-L5":8,
                            "L5":9,
                            "L5-S1":10,
                           }
        
        self.db = self._get_db()
        # self.json_paths = json_paths if isinstance(json_paths,list) else [json_paths]
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
        
        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        parse_annotations = pd.DataFrame()
        print("json Paths ", self.json_paths)
        # for jsonPath in self.json_paths:
        jsonPath = self.json_paths[0]

        trainPath = os.path.join(os.path.dirname(jsonPath),"train")
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        json_df = pd.read_json(jsonPath)
        for idx in json_df.index:
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation']
            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)

        dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))
        # 'studyUid','seriesUid','instanceUid'
        tag_list = ['0020|000d', '0020|000e', '0008|0018']
        dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
        for dcm_path in dcm_paths:
            try:
                studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list)
                row = pd.Series(
                    {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})
                dcm_info = dcm_info.append(row, ignore_index=True)
            except:
                continue
        result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
        # print(result.head())
        # result = result.set_index('dcmPath')
        result = result.set_index('dcmPath')['annotation']
        # print(result.head())
        parse_annotations = result
        # parse_annotations = parse_annotations.append(result)

            # print(result.head())
            # for row in result.iteritems():
            #     print(row[0], row[1][0]['data']['point'])
            #     img = dicom2array(row[0])
            #     points = row[1][0]['data']['point']
            #     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #     for item in points:
            #         corrd = item['coord']
            #         cv2.circle(bgr_img, center=(corrd[0], corrd[1]), radius=3, color=(100, 200, 100), thickness=-1)
            #     cv2.imshow("test", bgr_img)
            #     cv2.imwrite(row[0].replace('dcm', 'jpg'), bgr_img)
            #     cv2.waitKey(10)




        # file_name = os.path.join(self.root,
        #                          'annot',
        #                          self.image_set+'.json')
        # with open(file_name) as anno_file:
        #     anno = json.load(anno_file)

        gt_db = []
        # for a in anno:
        label_list = []
        for row in parse_annotations.iteritems():
            image_name = row[0]
            img = dicom2array(row[0])
            self.datas.append(img)
            # image_name = a['image']
            c = np.array([256,256])
            s = np.array([1,1])
            # c = np.array(a['center'], dtype=np.float)
            # s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            points = row[1][0]['data']['point']
            # print(points)
            #     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cnt = 0
            for i,item in enumerate(points):
                corrd = item['coord']

                l = item['tag']['identification']
                if l not in label_list:
                    label_list.append(l)

                if(l == "T11-T12"):
                    continue

                id = self.lab_to_idx[l]
                cnt += 1
                joints_3d[id,0] = corrd[0]
                joints_3d[id,1] = corrd[1]
                joints_3d_vis[id,0] = corrd[0]
                joints_3d_vis[id,1] = corrd[1]
            # print("cnt",cnt)

            # if self.image_set != 'test':
            #     joints = np.array(a['joints'])
            #     joints[:, 0:2] = joints[:, 0:2] - 1
            #     joints_vis = np.array(a['joints_vis'])
            #     assert len(joints) == self.num_joints, \
            #         'joint num diff: {} vs {}'.format(len(joints),
            #                                           self.num_joints)
            #
            #     joints_3d[:, 0:2] = joints[:, 0:2]
            #     joints_3d_vis[:, 0] = joints_vis[:]
            #     joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            image_dir = "/home/wang/PycharmProjects/tianchi"
            gt_db.append({
                'image': os.path.join(self.root, image_dir, image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        print(label_list)
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
