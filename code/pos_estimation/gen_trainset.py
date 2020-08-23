import _init_paths
from utils.dcm_utils import *
import  os
import pandas as pd
import glob
import json

dcm_root = "/home/wang/PycharmProjects/tianchi/lumbar_train150/train"
json_path = "/home/wang/PycharmProjects/tianchi/lumbar_train150/lumbar_train150_annotation.json"

def gen_data(dcm_root, json_path):
    # trainPath = os.path.join(os.path.dirname(json_path),"train")
    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
    json_df = pd.read_json(json_path)
    for idx in json_df.index:
        studyUid = json_df.loc[idx, "studyUid"]
        seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
        instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
        annotation = json_df.loc[idx, "data"][0]['annotation']
        row = pd.Series(
            {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
        annotation_info = annotation_info.append(row, ignore_index=True)

    dcm_paths = glob.glob(os.path.join(dcm_root, "**", "**.dcm"))
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
    print(result.head())
    parse_annotations = result
    save_path = json_path[:-5] + "_parsered" + ".json"
    result.to_json(save_path)
    # with open("test.json","w+") as f:
    #     json.dump(result,f)




