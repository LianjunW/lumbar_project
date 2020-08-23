import os
import json
import glob
import SimpleITK as sitk
import pandas as pd
import cv2
import albumentations as A
from aug_transforms import aug_transform
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


annotation_info = pd.DataFrame(columns=('studyUid','seriesUid','instanceUid','annotation'))
json_df = pd.read_json(jsonPath)
for idx in json_df.index:
    studyUid = json_df.loc[idx,"studyUid"]
    seriesUid = json_df.loc[idx,"data"][0]['seriesUid']
    instanceUid =  json_df.loc[idx,"data"][0]['instanceUid']
    annotation =  json_df.loc[idx,"data"][0]['annotation']
    row = pd.Series({'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid,'annotation':annotation})
    annotation_info = annotation_info.append(row,ignore_index=True)

dcm_paths = glob.glob(os.path.join(trainPath,"**","**.dcm"))
# 'studyUid','seriesUid','instanceUid'
tag_list = ['0020|000d','0020|000e','0008|0018']
dcm_info = pd.DataFrame(columns=('dcmPath','studyUid','seriesUid','instanceUid'))
for dcm_path in dcm_paths:
    try:
        studyUid,seriesUid,instanceUid = dicom_metainfo(dcm_path,tag_list)
        row = pd.Series({'dcmPath':dcm_path,'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid })
        dcm_info = dcm_info.append(row,ignore_index=True)
    except:
        continue
result = pd.merge(annotation_info,dcm_info,on=['studyUid','seriesUid','instanceUid'])
# print(result.head())
# result = result.set_index('dcmPath')
result = result.set_index('dcmPath')['annotation']
print(result.head())
for row in result.iteritems():
    print(row[0],row[1][0]['data']['point'])
    img = dicom2array(row[0])
    points = row[1][0]['data']['point']
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for item in points:
        corrd = item['coord']
        cv2.circle(bgr_img,center=(corrd[0],corrd[1]),radius=3,color=(100,200,100), thickness=-1)
    cv2.imshow("test",bgr_img)

    # t_img =  a_transform(image = img)
    t_img = aug_transform(image = img)
    cv2.imshow("t",t_img)
    # cv2.imwrite(row[0].replace('.dcm','_labe.jpg'),bgr_img)
    cv2.waitKey(0)
