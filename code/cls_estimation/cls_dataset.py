import os
import numpy as np
import random
import json
import glob
import SimpleITK as sitk
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
from  aug_transforms import aug_transform,ImgAug
import imgaug as ia
from imgaug import augmenters as iaa
from collections import  defaultdict
pad_to_fixsize = iaa.PadToFixedSize(width=192,height=96)
# aug_t = iaa.OneOf([iaa.Fliplr(p = 0.3),iaa.Flipup(p = 0.3)])

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
trainPath = r'./lumbar_train51/train'
jsonPath = r'./lumbar_train51/lumbar_train51_annotation.json'

trainPath = r'/home/wang/PycharmProjects/tianchi/lumbar_train150/train'
jsonPath = r'/home/wang/PycharmProjects/tianchi/lumbar_train150/lumbar_train150_annotation.json'


# annotation_info = pd.DataFrame(columns=('studyUid','seriesUid','instanceUid','annotation'))
# json_df = pd.read_json(jsonPath)
# for idx in json_df.index:
#     studyUid = json_df.loc[idx,"studyUid"]
#     seriesUid = json_df.loc[idx,"data"][0]['seriesUid']
#     instanceUid =  json_df.loc[idx,"data"][0]['instanceUid']
#     annotation =  json_df.loc[idx,"data"][0]['annotation']
#     row = pd.Series({'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid,'annotation':annotation})
#     annotation_info = annotation_info.append(row,ignore_index=True)
#
#
#
# dcm_paths = glob.glob(os.path.join(trainPath,"**","**.dcm"))
# # 'studyUid','seriesUid','instanceUid'
# tag_list = ['0020|000d','0020|000e','0008|0018']
# dcm_info = pd.DataFrame(columns=('dcmPath','studyUid','seriesUid','instanceUid'))
# for dcm_path in dcm_paths:
#     try:
#         studyUid,seriesUid,instanceUid = dicom_metainfo(dcm_path,tag_list)
#         row = pd.Series({'dcmPath':dcm_path,'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid })
#         dcm_info = dcm_info.append(row,ignore_index=True)
#     except:
#         continue
# result = pd.merge(annotation_info,dcm_info,on=['studyUid','seriesUid','instanceUid'])
# # print(result.head())
# # result = result.set_index('dcmPath')
# result = result.set_index('dcmPath')['annotation']
# print(result.head())
# for row in result.iteritems():
#     print(row[0],row[1][0]['data']['point'])
#     img = dicom2array(row[0])
#     points = row[1][0]['data']['point']
#     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     for item in points:
#         corrd = item['coord']
#         cv2.circle(bgr_img,center=(corrd[0],corrd[1]),radius=3,color=(100,200,100), thickness=-1)
#     cv2.imshow("test",bgr_img)
#     # cv2.imwrite(row[0].replace('.dcm','_labe.jpg'),bgr_img)
#     cv2.waitKey(0)

class ClsDataset(Dataset):
    def __init__(self,json_path,data_root,transform = None):
        self.input_arrays = []
        self.labels = []
        self.describe = []
        self.transform = transform
        self.init_db(json_path,data_root)

    def init_db(self,json_path,data_root):
        pos_instanceUids = []
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        json_df = pd.read_json(json_path)
        for idx in json_df.index:
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation']
            if instanceUid in pos_instanceUids:
                print("WWRONGWRONGWRONGWRONGWRONGWRONGWRONGWRONGWRONGWRONGRONG")
            pos_instanceUids.append(instanceUid)
             
            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)
        dcm_paths = glob.glob(os.path.join(data_root,"**","**.dcm"))
        # 'studyUid','seriesUid','instanceUid'
        tag_list = ['0020|000d','0020|000e','0008|0018','0020|0037','0008|103e']
        dcm_info = pd.DataFrame(columns=('dcmPath','studyUid','seriesUid','instanceUid'))
        for dcm_path in dcm_paths:
            try:
                studyUid,seriesUid,instanceUid,direction,describe = dicom_metainfo(dcm_path,tag_list)
                row = pd.Series({'dcmPath':dcm_path,'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid })
                dcm_info = dcm_info.append(row,ignore_index=True)
                self.describe.append(describe)
                s = 0
                d = [float(i) for i in direction.split("\\")]
                v = [0, 1, 0, 0, 0, -1]
                for i in range(len(v)):
                    s = s + d[i] * v[i]
                if s > 1.8:
                    img = dicom2array(dcm_path)
                    self.input_arrays.append(img)
                    if instanceUid in pos_instanceUids:
                        self.labels.append(1)
                        # print(describe)
                    else:
                        self.labels.append(0)
            except:
                continue
        # result = pd.merge(annotation_info,dcm_info,on=['studyUid','seriesUid','instanceUid'])
        # result = result.set_index('dcmPath')['annotation']
    def __getitem__(self, idx):
        img = self.input_arrays[idx]
        img = img[:,:,np.newaxis]
        if self.transform is not None:
            img = self.transform(img)
        return img,self.labels[idx],self.describe[idx]


        return self.input_arrays[idx],self.labels[idx]
    def __len__(self):
        return len(self.labels)
A_train_trainsforms = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])
# transforms.Pad()


class DiscClsDataset(Dataset):
    def __init__(self, json_path, data_root, transform=None):
        self.input_arrays = []
        self.labels = []
        self.describe = []
        self.transform = transform
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
        self.datas = np.empty(shape=(0,6))
        self.init_db(json_path, data_root)
        for key in self.label_dict.keys():
            print(key,self.label_dict[key])
        for k in self.lab_to_idx.keys():
            print(k,": ")
            ddd = self.label_count_dict[self.lab_to_idx[k]]
            ddd = sorted(ddd.items(),reverse=True ,key=lambda d: d[1])
            print(ddd)
            # for key in ddd.keys():
            #     print(key,ddd[key])

    def init_db(self, json_path, data_root):
        pos_instanceUids = []
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        json_df = pd.read_json(json_path)
        self.label_dict = defaultdict(int)
        self.label_count_dict = []
        for i in range(11):
            self.label_count_dict.append(defaultdict(int))


        for idx in json_df.index:
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation']
            if instanceUid in pos_instanceUids:
                print("WWRONGWRONGWRONGWRONGWRONGWRONGWRONGWRONGWRONGWRONGRONG")
            pos_instanceUids.append(instanceUid)

            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)
        dcm_paths = glob.glob(os.path.join(data_root, "**", "**.dcm"))
        # 'studyUid','seriesUid','instanceUid'
        tag_list = ['0020|000d', '0020|000e', '0008|0018', '0020|0037', '0008|103e']
        dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
        for dcm_path in dcm_paths:
            try:
                studyUid, seriesUid, instanceUid, direction, describe = dicom_metainfo(dcm_path, tag_list)
                row = pd.Series(
                    {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})
                dcm_info = dcm_info.append(row, ignore_index=True)
                # self.describe.append(describe)
                # s = 0
                # d = [float(i) for i in direction.split("\\")]
                # v = [0, 1, 0, 0, 0, -1]
                # for i in range(len(v)):
                #     s = s + d[i] * v[i]
                # if s > 1.8:
                #     img = dicom2array(dcm_path)
                #     self.input_arrays.append(img)
                #     if instanceUid in pos_instanceUids:
                #         self.labels.append(1)
                #         # print(describe)
                #     else:
                #         self.labels.append(0)
            except:
                continue
        result = pd.merge(annotation_info,dcm_info,on=['studyUid','seriesUid','instanceUid'])
        result = result.set_index('dcmPath')['annotation']
        self.indexs = []
        img_idx = 0
        for row in result.iteritems():
            image_name = row[0]
            img = dicom2array(row[0])
            self.input_arrays.append(img)


            # self.datas.append(img)
            #
            # joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            # joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            points = row[1][0]['data']['point']

            # tags = row[1][0]['data']['tag']
            # print(points)
            #     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cnt = 0
            labels = []
            joints = np.ones((11,6))
            joints[:,4] *= img_idx
            img_idx += 1
            for i, item in enumerate(points):
                corrd = item['coord']
                tag = item['tag']
                l = None
                t = tag['identification']

                if t == "T11-T12":
                    continue
                # print(t)
                point_id = self.lab_to_idx[t]
                joints[point_id][:2] = corrd
                joints[point_id][5] = point_id
                if "-" in tag['identification']:
                    l = tag['disc']
                else:
                    l = tag['vertebra']
                self.label_count_dict[point_id][l] += 1
                self.label_dict[l] += 1
                try:
                    l = int(l[1])
                except:
                    print(l)
                    l = 1
                if l > 5:
                    l = 5
                if l < 1:
                    l = 1
                joints[self.lab_to_idx[t]][2] = l - 1
            mean_dis = (joints[10][1] -joints[0][1])/10
            for i in range(11):
                if i > 0:
                    upper = joints[i][1] - joints[i-1][1]
                else:
                    upper = mean_dis*1.2
                if i < 10:
                    bottom = joints[i+1][1] - joints[i][1]
                else:
                    bottom = mean_dis*1.2
                # print(upper,bottom)
                limit = max(upper,bottom)
                # print(limit)
                if (abs(limit - mean_dis) > mean_dis * 0.5):
                    limit = mean_dis
                joints[i][3] = limit
            self.datas = np.concatenate((self.datas,joints),axis=0)


                # l = item['tag']['identification']
                # if l not in label_list:
                #     label_list.append(l)
                #
                # if (l == "T11-T12"):
                #     continue
                #
                # id = self.lab_to_idx[l]
                # cnt += 1
                # joints_3d[id, 0] = corrd[0]
                # joints_3d[id, 1] = corrd[1]
                # joints_3d_vis[id, 0] = corrd[0]
                # joints_3d_vis[id, 1] = corrd[1]


        # result = pd.merge(annotation_info,dcm_info,on=['studyUid','seriesUid','instanceUid'])
        # result = result.set_index('dcmPath')['annotation']

    def __getitem__(self, idx):
        item = self.datas[idx]
        img_id = int(item[4])
        limit = int(item[3])
        # print(limit)
        c_x = int(item[0])
        c_y = int(item[1])
        lab = int(item[2])
        p_id = int(item[5])
        base_x_p = 1.6
        base_y_p = 0.8
        if p_id%2 == 1:
            base_x_p *= 0.5
            base_y_p *= 0.5

        p = random.random()*0.5
        x_p = (base_x_p+p)
        y_p = (base_y_p+p)
        start_x = int(c_x - x_p*limit)
        start_y = int(c_y - y_p*limit)

        img = self.input_arrays[img_id][int(start_y):int(start_y+2*y_p*limit),int(start_x):int(start_x+limit*2*x_p)]
        h,w = img.shape
        if h > 96 or w > 192:
            img = cv2.resize(img,(192,96))

        img = pad_to_fixsize(image = img)
        if self.transform is not None:
            img = self.transform(img)
        one_hot = np.zeros(11)
        one_hot[p_id]  = 1
        # test_t = target[:,j].unsqueeze(1)
        # one_hot.scatter_(1,target[:,j].unsqueeze(1).long(),1)

        return img,lab,one_hot





        # return None
        # img = self.input_arrays[idx]
        # img = img[:, :, np.newaxis]
        # if self.transform is not None:
        #     img = self.transform(img)
        # return img, self.labels[idx], self.describe[idx]
        #
        # return self.input_arrays[idx], self.labels[idx]

    def __len__(self):
        return len(self.datas)
class ImgAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug = iaa.SomeOf((0,3),[iaa.Fliplr(p = 0.3),
                                      iaa.Flipud(p = 0.3),
                                      sometimes(iaa.LinearContrast((0.75, 1.5))),
                                      iaa.OneOf([
                                          iaa.GaussianBlur((0, 3.0)),
                                          iaa.AverageBlur(k=(2, 7)),
                                          iaa.MedianBlur(k=(3, 11)),
                                      ])
                                      ])

        # self.aug = aug_transform
    def __call__(self,img):
        return self.aug(image = img)
        # array_img = self.aug(image = numpy.array(img))
        # return Image.fromarray(array_img)

    def __repr__(self):
        return self.__class__.__name__

train_transforms = transforms.Compose([ImgAug(),transforms.ToPILImage(),transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

if __name__ == "__main__":
    labs = []
    dataset = DiscClsDataset(jsonPath,trainPath)
    img_w = 0
    img_h = 0
    for i,(img,lab,p_id) in enumerate(dataset):
        # print(lab)
        try:
            # cv2.imshow("src_img",src_img)

            cv2.imshow("img",img)
            # print(lab,p_id)
            if lab not in labs:
                labs.append(lab)
            # print(img.shape)
            h,w = img.shape
            img_h = max(img_h,h)
            img_w = max(img_w,w)
            cv2.waitKey(0)
        except:
            continue

    print(labs)
    print("Max w:{}, Max h:{}".format(img_w,img_h))
    # for i,(img,lab,d) in enumerate(dataset):
        # cv2.imshow("img",img)
        # print("lab:{:2} : {}".format(lab,d.encode('UTF-8','ignore').decode('UTF-8')))
        # cv2.waitKey(0)
