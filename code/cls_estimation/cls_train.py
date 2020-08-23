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


best_pred = 0.0
def get_args():
    parser = argparse.ArgumentParser(
        description="Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)")

    parser.add_argument("-d", "--data_path", type=str, default="data", help="the root folder of dataset")
    parser.add_argument("-e", "--epochs", default=160, type=int, help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("-l", "--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("-m", "--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("-w", "--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--log_path", type=str, default="tensorboard/keyframe")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


class MyModel(nn.Module):
    def __init__(self,cls_num):
        super(MyModel,self).__init__()
        self.cls_num = cls_num

        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        model.fc = torch.nn.Linear(512, 100)
        self.base_model = model
        self.p_fc = nn.Sequential(torch.nn.Linear(11,11),torch.nn.Sigmoid())
        self.fc1 = nn.Sequential(torch.nn.Linear(111,cls_num),torch.nn.ReLU(),torch.nn.Linear(cls_num,cls_num))


        # model.to(device)
    def forward(self, input,p_id):

        x = self.base_model(input)
        # print(x.shape)
        # print(p_id.shape)
        p_id = self.p_fc(p_id)
        x = torch.cat([x,p_id],1)

        # print(x.shape)
        x = self.fc1(x)

        return x


class ResModel(nn.Module):
    def __init__(self, cls_num):
        super(ResModel, self).__init__()
        self.cls_num = cls_num
        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        model.fc = torch.nn.Linear(512, cls_num)
        self.base_model = model
        # self.fc1 = nn.Sequential(torch.nn.Linear(111, cls_num), torch.nn.ReLU(), torch.nn.Linear(cls_num, cls_num))
        # model.to(device)
    def forward(self, input, p_id = None):
        x = self.base_model(input)
        return x
import time

timestr = time.strftime('%m%d%H%M')

def main(opt):
    # train_lists_path = "./data/train_07201150.txt"
    # val_list_path = "./data/val_07201150.txt"
    # train_lists_path = "./data/train_07271026.txt"
    # val_list_path = "./data/val_07271026.txt"
    #
    # train_lists_path = "./data/train_0728.txt"
    # val_list_path = "./data/val_0728.txt"
    cls_num = 5
    train_json_path = "/home/wang/PycharmProjects/tianchi/lumbar_train150/lumbar_train150_annotation.json"
    train_data_root = "/home/wang/PycharmProjects/tianchi/lumbar_train150/train"

    val_json_path = "/home/wang/PycharmProjects/tianchi/lumbar_train51/lumbar_train51_annotation.json"
    val_data_root = "/home/wang/PycharmProjects/tianchi/lumbar_train51/train"
    train_dataset = DiscClsDataset(train_json_path, train_data_root, transform=train_transforms)
    val_dataset = DiscClsDataset(val_json_path, val_data_root, transform=test_transforms)
    # train_dataset = Anomalydataset(train_lists_path)
    # val_dataset = Anomalydataset(val_list_path)

    train_dataloader = DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size = opt.batch_size,drop_last=True)
    device = "cpu"
    if torch.cuda.is_available():
        device="cuda"
        print("cuda")
    scheduler = LR_Scheduler('cos',
                             base_lr=opt.lr,
                             num_epochs=opt.epochs,
                             iters_per_epoch=len(train_dataloader),
                             warmup_epochs=5)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    opt.log_path = opt.log_path + time.strftime("%m%d%H%M")
    writer = SummaryWriter(opt.log_path,filename_suffix=time.strftime("%m%d%H%M"))

    # model = models.resnet18()
    # model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
    # model.fc = torch.nn.Linear(512,cls_num)

    model = MyModel(cls_num = cls_num)
    # model = ResModel(cls_num = cls_num)

    model.to(device)
    # model.cuda()
    # criterion = MyBCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    best_acc1 = 0

    for epoch in range(opt.epochs):
        # adjust_learning_rate(optimizer, epoch, opt.lr)
        train(train_dataloader, model, criterion, optimizer, epoch, writer, scheduler)
        acc1 = validate(val_dataloader, model, criterion, epoch, writer)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        }, is_best, opt.saved_path)










def train(train_loader, model, criterion, optimizer, epoch, writer,scheduler):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    rec1 = AverageMeter("Rec@1", ":6.2f")
    # top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1,rec1],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    num_iter_per_epoch = len(train_loader)
    global best_pred
    scheduler(optimizer, 0, epoch, best_pred)
    for i, (inputs, target,p_id) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            inputs = inputs.cuda().float()
            target = target.cuda()
            p_id = p_id.cuda().float()

        # compute output
        output = model(inputs,p_id)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        # print(acc1)
        # rec = recall(output,target)
        losses.update(loss.detach().item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        # rec1.update(rec,inputs.size(0))
        writer.add_scalar('Train/Loss', losses.avg, epoch * num_iter_per_epoch + i)
        writer.add_scalar('Train/Top1_acc', top1.avg, epoch * num_iter_per_epoch + i)
        # writer.add_scalar('Train/Recall', rec1.avg, epoch * num_iter_per_epoch + i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    rec1 = AverageMeter("Rec@1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1,rec1],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    global best_pred
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target,p_id) in enumerate(val_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda().float()
                target = target.cuda().long()
                p_id = p_id.cuda().float()

            # compute output
            output = model(inputs,p_id)
            # print(output.shape)
            # print(target.shape)
            # target = target.unsqueeze(dim = 1)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            # rec = recall(output,target)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            # rec1.update(rec,inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5 == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f}"
              .format(top1=top1))
        writer.add_scalar('Test/Loss', losses.avg, epoch)
        writer.add_scalar('Test/Top1_acc', top1.avg, epoch)
        # writer.add_scalar('Test/Recall', rec1.avg, epoch)
    best_pred = max(top1.avg, best_pred)
    return top1.avg


def save_checkpoint(state, is_best, saved_path, filename="checkpoint.pth.tar"):
    filename = timestr + "_" + filename
    file_path = os.path.join(saved_path, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(saved_path, "best_checkpoint.pth.tar"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def recall(output,target):
    _, predicted = output.max(1)
    # loss = criterion(outputs, targets)
    # test_correct += predicted.eq(targets).sum().item()
    # (predicted[targets==1] == 1).sum().item()/(targets==1).sum().item()
    TP = ((predicted == 1) & (target == 1)).sum().item()
    # TN = ((predicted == 0) & (target == 0)).sum().item()
    FN = ((predicted == 0) & (target == 1)).sum().item()
    # FP = ((predicted == 1) & (target == 0)).sum().item()
    r = TP/(TP+FN + 0.0000001)*100.0
    return r
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum().float()
        res = correct.mul_(100.0 / batch_size)
        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # # res = [] for k in topk:
        # #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #     res.append()
        return res

if __name__ == "__main__":
    opt = get_args()
    main(opt)