import os
import glob
import SimpleITK as sitk
import pandas as pd
import json
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
# test_root = "/home/wang/PycharmProjects/tianchi/lumbar_testA50"
res_path = "/home/wang/PycharmProjects/tianchi/output/study201_image14.txt"
test_root = "/home/wang/PycharmProjects/tianchi/B/lumbar_testB50"

datas = []

def convert_json(res_path):
    fn = os.path.basename(res_path).replace("_","/").replace("txt","dcm")
    dcm_path = os.path.join(test_root,fn)
    # 'studyUid','seriesUid','instanceUid'
    tag_list = ['0020|000d','0020|000e','0008|0018','0020|0011']
    # list_tag = ['0020|000d', '0020|000e', '0008|0018', '0020|0011', '0020|0037', '0020|0032', '0008|1030', '0008|103e']
    studyUid, seriesUid, instanceUid,zid = dicom_metainfo(dcm_path, tag_list)
    zid = int(zid)
    annotations = []
    labs = ["T12-L1","L1","L1-L2","L2","L2-L3","L3","L3-L4","L4","L4-L5","L5","L5-S1"]

    names = ["disc","vertebra"]
    points = []
    with open(res_path,'r') as fin:
        lines = fin.readlines()
        id = 0
        for line in lines:
            x,y = line.strip().split(',')
            l = labs[id]
            tmp_d = {"tag":{"identification":l,names[id%2]:"v2"},"coord":[int(x),int(y)],"zIndex":zid}
            points.append(tmp_d)
            id += 1
    data_item ={"studyUid":studyUid,"version":"v0.1",
                "data":[{"seriesUid":seriesUid,"instanceUid":instanceUid,
                         'annotation':[{"annotator":72 ,"data":{"point":points}}]}]}
    datas.append(data_item)


txt_paths = glob.glob("/home/wang/PycharmProjects/tianchi/output/*.txt")
txt_paths = glob.glob("/home/wang/PycharmProjects/tianchi/output_B_mysoft/*.txt")

for txt_p in txt_paths:
    convert_json(txt_p)

with open("json_res_no_cls.json","w+") as f:
    json.dump(datas,f)
# a = json.dumps(datas)

# print(a)
# print(datas)
# dict = {"studyUid":studyUid,data}




