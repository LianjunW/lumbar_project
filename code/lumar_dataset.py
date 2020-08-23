from torch.utils.data import Dataset
import os
import pandas as pd
import glob
import SimpleITK as sitk
import numpy as np
import random
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
class LumarDataset(Dataset):
    def __init__(self,json_file):
        super().__init__()
        self.json_file = json_file
        self.datas = []
        self.points = []
        self.rois = [] 
        self.wh = []
        self.sample_size = 64
        self.parse_json() 
    def __len__(self):
        return len(self.rois)
    def in_rect(self,p,x,y):
        # print(p,x,y)
        return all([p[0] > x , p[1] > y , p[0] < x + self.sample_size , p[1] < y + self.sample_size])
        # return p[0] > x and p[1] > y and p[0] < x + self.sample_size and p[1] < y + self.sample_size
        
        
    def __getitem__(self, idx):
        roi = self.rois[idx] 
        img = self.datas[idx]
        h,w = img.shape
        item_size = self.sample_size
        if random.random() < 0.8:
            #get item from roi
            print(roi[0],min(roi[2],w) - item_size)
            xmin = roi[0]
            xmax = roi[2] - item_size
            if xmin > xmax:
                center =int((roi[0] + roi[2])/2)
                xmin = max(center - 80,0)
                xmax = min(center + 80,w-self.sample_size)
            x1 = random.randint(xmin,xmax)

            ymin = roi[1]
            ymax = roi[3] - item_size
            if ymin < ymax:
                center = int((roi[1] + roi[3])/2)
                ymin = max(center - 80,0)
                ymax = min(center + 80, w - self.sample_size)
            y1 = random.randint(ymin,ymax)
        else:
            x1 = random.randint(0,w-item_size)
            y1 = random.randint(0,h-item_size)
        item_img = img[y1:item_size+y1,x1:x1+item_size]
        item_points = []
        for p in self.points[idx]:
            if self.in_rect(p,x1,y1):
               item_points.append(p)
        return item_img,item_points 
    def parse_json(self):
        jsonPath = self.json_file
        trainPath = os.path.join(os.path.dirname(jsonPath), "train")
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

        gt_db = []
        # for a in anno:
        
        for row in result.iteritems():
            image_name = row[0]
            img = dicom2array(row[0])
            self.wh.append(img.shape) 
            self.datas.append(img)
            # c = c - 1

            joints_3d = np.zeros((11, 3), dtype=np.float)
            points = row[1][0]['data']['point']
            x1 = 100000
            y1 = 100000
            x2 = 0 
            y2 = 0 
            
            for i, item in enumerate(points):
                corrd = item['coord']
                joints_3d[i, 0] = corrd[0]
                joints_3d[i, 1] = corrd[1]
                x1 = min(corrd[0],x1)
                y1 = min(corrd[1],y1)
                x2 = max(corrd[0],x2)
                y2 = max(corrd[1],y2)
            self.points.append(joints_3d)

            x1 = max(0,x1-20)
            y1 = max(0,y1-20)
            x2 = min(img.shape[1],x2 + 20)
            y2 = min(img.shape[0],y2 + 20)
            self.rois.append([x1,y1,x2,y2])                
        
        
        
jpath = "/home/wang/PycharmProjects/tianchi/lumbar_train150/lumbar_train150_annotation.json" 
test_dataset = LumarDataset(jpath)
for idx,(img,pts) in enumerate(test_dataset):
    print(idx,img.shape)
    cv2.imshow("img",img)
    cv2.waitKey(0)



