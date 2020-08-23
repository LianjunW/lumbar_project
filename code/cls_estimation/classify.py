from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import SGD,Adam,lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils import *
import torch
from cls_dataset import *
from torch.utils.data import DataLoader
import argparse
# from dataset import *
from torch import nn
import os
import shutil
import time
from torchvision import  models
from pthflops import count_ops
from torchsummary import summary
from loss import *
from PIL import Image
from cls_dataset import test_transforms,pad_to_fixsize
from cls_train import  MyModel
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
device = "cpu"
if torch.cuda.is_available():
    device="cuda"
    print("cuda")
model_path = "/home/wang/PycharmProjects/tianchi/trained_models/best_checkpoint.pth.tar"
model = MyModel(5)
# model = models.resnet18()
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
# model.fc = torch.nn.Linear(512, 5)
model.to(device)
net = model
net.load_state_dict(torch.load(model_path)['state_dict'])
# net=torch.load(model_path).to(device)
toPIl = transforms.ToPILImage()
net.eval()
def eval(img,p_id):
    # img = Image.open(img_path)  # RGB (w,h)
    # img2  = cv2.imread(img_path)
    # print(img.shape)
    h,w = img.shape
    if h > 96 or w > 192:
        img = cv2.resize(img, (192, 96))
    p_id = torch.from_numpy(p_id)
    p_id = p_id.cuda().float()

    img = pad_to_fixsize(image=img)
    img = test_transforms(img)
    # pil_img = Image.fromarray(img)
    # pil_img = Image.fromarray(img2)

    # pil_img = parcel_transform_test(pil_img)
    # pil_img = torch.unsqueeze(pil_img, 0)
    img = torch.unsqueeze(img,0)
    p_id = torch.unsqueeze(p_id,0)
    # print(img.shape)
    out = net(img.to(device),p_id)


    max_res = torch.max(out,1)
    print(to_numpy(out))
    return max_res[1].item() + 1
    # print(to_numpy(out))
# img = None
# eval(img)
test_root = "/home/wang/PycharmProjects/tianchi/B/lumbar_testB50"
datas = []
label_dict = defaultdict(int)
label_count_dict = []
for i in range(11):
    label_count_dict.append(defaultdict(int))


def classify(res_path):
    fn = os.path.basename(res_path).replace("_","/").replace("txt","dcm")
    dcm_path = os.path.join(test_root,fn)
    # 'studyUid','seriesUid','instanceUid'
    tag_list = ['0020|000d','0020|000e','0008|0018','0020|0011']
    # list_tag = ['0020|000d', '0020|000e', '0008|0018', '0020|0011', '0020|0037', '0020|0032', '0008|1030', '0008|103e']
    studyUid, seriesUid, instanceUid,zid = dicom_metainfo(dcm_path, tag_list)
    zid = int(zid)
    annotations = []
    labs = ["T12-L1","L1","L1-L2","L2","L2-L3","L3","L3-L4","L4","L4-L5","L5","L5-S1"]
    pre_tags = ["v1", 0,    "v1",   0,   "v1",   0,   "v1",  0,    "v3", 0,   "v3" ]

    names = ["disc","vertebra"]
    points = []
    y_list = []
    x_list = []
    with open(res_path,'r') as fin:
        lines = fin.readlines()
        id = 0
        for line in lines:
            x,y = line.strip().split(',')
            y_list.append(int(y))
            x_list.append(int(x))

            
            id += 1
    mean_dis = (y_list[-1] - y_list[0])/10
    img = dicom2array(dcm_path)
    for i in range(len(y_list)):
        if i > 0:
            upper = y_list[i] - y_list[i - 1]
        else:
            upper = mean_dis
        if i < 10:
            bottom = y_list[i + 1] - y_list[i]
        else:
            bottom = mean_dis

        limit = max(upper, bottom)
        if (abs(limit-mean_dis) > mean_dis*0.5):
            limit = mean_dis
        x = x_list[i]
        y = y_list[i]
        limit = int(limit)
        start_x = int(x - 2 * limit)
        start_y = int(y - limit)
        roi_img = img[start_y: start_y + 2 * limit, start_x: start_x + limit * 4]

        cv2.imshow("roi",roi_img)
        cv2.imshow("img",img)
        one_hot = np.zeros(11)
        one_hot[i] = 1
        cls_num = eval(img,one_hot)
        print(cls_num)
        label_dict["v"+str(cls_num)] += 1

        label_count_dict[i]["v"+str(cls_num)] += 1
        l = labs[i]
        if pre_tags[i] == 0:
            disc_tag = "v" + str(cls_num)
        else:
            disc_tag = pre_tags[i]
        tmp_d = {"tag": {"identification": l, names[i % 2]: disc_tag}, "coord": [int(x), int(y)], "zIndex": zid}
        points.append(tmp_d)
        cv2.waitKey(10)

    data_item ={"studyUid":studyUid,"version":"v0.1",
                "data":[{"seriesUid":seriesUid,"instanceUid":instanceUid,
                         'annotation':[{"annotator":72 ,"data":{"point":points}}]}]}
    datas.append(data_item)


txt_paths = glob.glob("/home/wang/PycharmProjects/tianchi/output_B_mysoft/*.txt")
for txt_p in txt_paths:
    classify(txt_p)
with open("json_res_cls.json","w+") as f:
    json.dump(datas,f)
for key in label_dict.keys():
    print(key,label_dict[key])
labs = ["T12-L1","L1","L1-L2","L2","L2-L3","L3","L3-L4","L4","L4-L5","L5","L5-S1"]
for i in range(len(labs)):
    print(labs[i], ": ")
    ddd = label_count_dict[i]
    ddd = sorted(ddd.items(), reverse=True, key=lambda d: d[1])
    print(ddd)