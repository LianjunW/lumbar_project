from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.dcm_utils import *

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_final_preds,get_max_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform
import numpy as np
import cv2
import models
import dataset
import glob
import SimpleITK as sitk
import pandas as pd
import cv2



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--img-file',
                        help='input your test img',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file

def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale
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
# def get_max_preds(batch_heatmaps):
#     batch_size = batch_heatmaps.shape[0]
#     num_joints = batch_heatmaps.shape[1]
#
#     heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
#
#     maxvals = np.amax(heatmaps_reshaped, 2)
#     maxvals = maxvals.reshape((batch_size, num_joints, 1))
#
#     pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
#     pred_mask = pred_mask.astype(np.float32)
#
#     preds = beta_soft_argmax(torch.from_numpy(batch_heatmaps)).numpy()
#     preds *= pred_mask
#
#     return preds, maxvals
def get_final_preds(config, batch_heatmaps):
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        preds, maxval = get_max_preds(batch_heatmaps)

    # Transform back
    # for i in range(preds.shape[0]):
    #     preds[i] = transform_preds(preds[i], center[i], scale[i],
    #                                [heatmap_width, heatmap_height])

    return preds, maxval

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')



    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    print(model)
    # config.TEST.MODEL_FILE = "/home/wang/PycharmProjects/tianchi/human-pose-estimation/output/lumbar/lp_net_50/my/checkpoint.pth.tar"



    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        print(final_output_dir)
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # Loading an image
    image_file = args.img_file
    test_set_paths = "../../submit/B_dcm_list.txt"
    save_root = "../../submit/pos_output"
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    with open(test_set_paths) as fin:
        lines = fin.readlines()
        for line in lines:
            img_file = line.strip()
            print(img_file)
        # img_file = "/home/wang/PycharmProjects/tianchi/lumbar_train150/train/study72/image15.dcm"
        # data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            data_numpy = dicom2array(img_file)
            input = data_numpy.copy()
            input = input[:,:,np.newaxis]
            h,w,_ = input.shape
            input = cv2.resize(input,(512,512))

            h_sf = (512/128)*(h/512)
            w_sf = 4*w/512.0

            # print(input.shape)
            # object detection box
            # need to be given [left_top, w, h]
            # box = [391, 99, 667-391, 524-99]
            # box = [743, 52, 955-743, 500-52]
            # box = [93, 262, 429-93, 595-262]
            # c, s = _box2cs(box, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])
            # print(c)
            # r = 0

            # trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
            # print(trans.shape)
            # input = cv2.warpAffine(
            #     data_numpy,
            #     trans,
            #     (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            #     flags=cv2.INTER_LINEAR)



            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
            ])

            input = transform(input).unsqueeze(0)
            # switch to evaluate mode
            model.eval()
            fn = os.path.basename(os.path.dirname(img_file)) + "_"+ os.path.basename(img_file)
            save_path = os.path.join(save_root,fn.replace("dcm","txt"))
            res_fout = open(save_path,'w')
            with torch.no_grad():

                # compute output heatmap
                output = model(input)
                # print(output.shape)
                preds, maxvals = get_final_preds(config, output.clone().cpu().numpy())

                image = data_numpy.copy()
                if(len(preds[0]) != 11):
                    print("point num not right:",line,len(preds[0]))
                for mat in preds[0]:
                    x, y = int(mat[0]*w_sf), int(mat[1]*h_sf)
                    res_fout.write(str(x) + "," + str(y) + "\n")
                    # x *=w_sf
                    # y *=h_sf
                    cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

                # vis result
                # cv2.imwrite("test_lp50.jpg", image)
                cv2.imshow('demo', image)
                # print(fn)
                cv2.imwrite(save_root +"/" + fn.replace("dcm","jpg"),image)
                cv2.waitKey(10)
                # cv2.destroyAllWindows()
            res_fout.close()
if __name__ == '__main__':
    main()

