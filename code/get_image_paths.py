import  os
import pandas as pd
import SimpleITK as sitk
import cv2
import glob
from collections import defaultdict
from lib.utils.dcm_utils import *

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
    res = []
    for tag in list_tag:
        try:
            tmp = reader.GetMetaData(tag)
            res.append(tmp)
        except:
            res.append("nothing!")
            continue
    return res
    # return [reader.GetMetaData(t) for t in list_tag]


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

json_path = "../data/DatasetB/testB50_series_map.json"
dcm_root = "../data/DatasetB/lumbar_testB50"

annotation_info = pd.DataFrame(columns=('studyUid','seriesUid'))
json_df = pd.read_json(json_path)
studyuid_seriesuid = defaultdict(str)
seriesuids = []
for idx in json_df.index:
    studyUid = json_df.loc[idx,"studyUid"]
    seriesUid = json_df.loc[idx,"seriesUid"]
    studyuid_seriesuid[studyUid] = seriesUid
    seriesuids.append(seriesUid)
    print(studyUid,seriesUid)
study_roots = glob.glob(os.path.join(dcm_root, "*"))
describe_8_103e = []
describe = []
# pos_lab = []
# neg_lab = []

pos_tags = defaultdict(int)
neg_tags = defaultdict(int)
parse_annotations = pd.DataFrame()

seriesUid2instance_dict = defaultdict(list)

save_root = "../submit/"

for study_root in study_roots:
    dd = defaultdict(list)
    dcm_paths = glob.glob(os.path.join(study_root + "/*.dcm"))

    # instance_dict = defaultdict(list)

    for dcm_path in dcm_paths:
        sopuids = []
        # for fn in os.listdir(dcm_root):
        #     dcm_path = os.path.join(dcm_root,fn)
        print(dcm_path)
        try:
            img = dicom2array(dcm_path)
        except:
            continue
        print(img.shape)
        list_tag = ['0020|000d', '0020|000e', '0008|0018', '0020|0011', '0020|0037', '0020|0032', '0008|1030',
                    '0008|103e']
        # study uid ,series uid, sop uid(frame id) ,第几个序列
        # list_tag = ['0008|0018', '0020|0011','0008|103e']
        # instance id,
        r = dicom_metainfo(dcm_path, list_tag)
        series_id = r[1]
        # describe_8_103e.append(r[2])
        print(r)
        tmp = []
        if  r[1] in seriesuids:
            tmp.append(os.path.abspath(dcm_path))
            r5 = [float(i) for i in r[5].split("\\")]
            tmp.append(r5[0])
            seriesUid2instance_dict[series_id].append(tmp)
            # dd[r[1]].append(tmp)
parse_annotations = pd.DataFrame()
dd = seriesUid2instance_dict
dcm_list_path = os.path.join(save_root,"B_dcm_list.txt")
fout = open(dcm_list_path,'w')

for key in dd.keys():
    dd[key].sort(key=lambda x:x[-1])
    d_path = (dd[key][int(len(dd[key]) / 2)][0])
    fout.write(d_path + "\n")

    dd_frame = pd.DataFrame(dd[key], columns=('imgpath' , 'x_axis'))

    # print(dd_frame)
    parse_annotations = parse_annotations.append(dd_frame)

# fout.close()
# parse_annotations.to_csv("B_series.csv")

        # cv2.putText(img,dstr,(10,180),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.8,thickness=1,color=255)
        # cv2.imshow("img",img)
        # save_path = dcm_path.replace("dcm","jpg")
        # cv2.imwrite(save_path,img)
        # cv2.waitKey(10)
    # find_t2 = False
    # t2_serials = []
    # for key in dd.keys():
    #
    #     dd[key].sort(key=lambda x: x[-1])
    #
    #     have_pos = False
    #     for item in dd[key]:
    #         if item[1] == 1:
    #             have_pos = True
    #
    #     if (have_pos):
    #         pos_tags[dd[key][0][2]] += 1
    #     else:
    #         neg_tags[dd[key][0][2]] += 1
    #
    #     # print(dd[key])
    #     if "t2" in dd[key][0][2] or "T2" in dd[key][0][2]:
    #         t2_serials.append(key)
    # if len(t2_serials) < 1:
    #     for key in dd.keys():
    #         if "count" not in key.lower():
    #             t2_serials.append(key)
    # for key in t2_serials:
    #     dd_frame = pd.DataFrame(dd[key], columns=('imgpath', 'lab', 'tag', 'series_id', 'x_axis'))
    #     print(dd_frame)
    #     parse_annotations = parse_annotations.append(dd_frame)



    # seriesUid = json_df.loc[idx,"data"][0]['seriesUid']
    # instanceUid =  json_df.loc[idx,"data"][0]['instanceUid']
    # annotation =  json_df.loc[idx,"data"][0]['annotation']
    # row = pd.Series({'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid,'annotation':annotation})
    # annotation_info = annotation_info.append(row,ignore_index=True)
