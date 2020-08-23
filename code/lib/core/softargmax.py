import torch
import torch.nn as nn
from torch.nn import functional as F


class SoftArgmax2D(nn.Module):
    """
    Creates a module that computes Soft-Argmax 2D of a given input heatmap.
    Returns the index of the maximum 2d coordinates of the give map.
    :param beta: The smoothing parameter.
    :param return_xy: The output order is [x, y].
    """

    def __init__(self, beta: int = 100, return_xy: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x H x W.
        :return: The index of the maximum 2d coordinates is of size B x N x 2.
        """
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        softmax: torch.Tensor = F.softmax(
            heatmap.view(batch_size, num_channel, height * width), dim=2
        ).view(batch_size, num_channel, height, width)
        # print("softmax",softmax[0,0,:,:])
        xx, yy = torch.meshgrid(list(map(torch.arange, [height, width])))
        # print("xx",xx.shape)
        # print("yy",yy)
        approx_x = (
            softmax.mul(xx.float().to(device))
                .view(batch_size, num_channel, height * width)
                .sum(2)
                .unsqueeze(2)
        )
        # print(approx_x)
        approx_y = (
            softmax.mul(yy.float().to(device))
                .view(batch_size, num_channel, height * width)
                .sum(2)
                .unsqueeze(2)
        )
        # print(approx_y)

        output = [approx_x, approx_y] if self.return_xy else [approx_y, approx_x]
        output = torch.cat(output, 2)
        # print(output)
        return output
class MySoftArgmax2D(nn.Module):
    """
    Creates a module that computes Soft-Argmax 2D of a given input heatmap.
    Returns the index of the maximum 2d coordinates of the give map.
    :param beta: The smoothing parameter.
    :param return_xy: The output order is [x, y].
    """

    def __init__(self, beta: int = 100, return_xy: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x H x W.
        :return: The index of the maximum 2d coordinates is of size B x N x 2.
        """
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        init_headmap = heatmap[:,-2:,:,:]
        init_h_softmax  = F.softmax(init_headmap.view(batch_size,2,height*width),dim = 2).view(batch_size,2,height,width)
        yy,xx = torch.meshgrid(list(map(torch.arange, [height, width])))
        # print("xx",xx)
        # print("yy",yy)
        init_approx_x = (
            init_h_softmax.mul(xx.float().to(device))
                .view(batch_size, 2, height * width)
                .sum(2)
                .unsqueeze(2)
        )
        # print(init_approx_x)
        init_approx_y = (
            init_h_softmax.mul(yy.float().to(device))
                .view(batch_size, 2, height * width)
                .sum(2)
                .unsqueeze(2)
        )
        # print(init_approx_y)
        last_y = init_approx_y[:,0,:]


        output = torch.zeros(size=(11,2))

        output[9:11,0] = init_approx_x.squeeze()
        output[9:11,1] = init_approx_y.squeeze()
        # out[10]
        dis_sum = 0
        dis_num = 0
        dis = output[10,1] - output[9,1]
        for i in range(8,-1,-1):
            last_y = output[i+1][1].long()
            tmp_dis = torch.ceil(output[i + 2, 1] - output[i + 1, 1])
            if (abs(tmp_dis - dis) > 0.35*dis):
                dis_sum += tmp_dis
                dis_num += 1
                dis = dis_sum / dis_num
            # print("dis shape",dis.shape)
            # print("dis",dis)
            # print(last_y[0])
            start_y = last_y - 1.8*dis
            end_y = torch.round(start_y + (0.9*2)*dis).long()
            start_y = torch.round(start_y).long()
            # print("start y",start_y)
            tmp_headmap = heatmap[:,i,:,:].squeeze(0)
            # print(tmp_headmap.shape)
            # print(start_y,last_y)
            mask = torch.zeros(size=(height,width))
            mask[start_y:end_y+1,:] = 1

            tmp_headmap = tmp_headmap.mul(mask)
            tmp_headmap_softmax = F.softmax(tmp_headmap.view(height*width),dim = 0).view(height,width)
            tmp_approx_x = (
                tmp_headmap_softmax.mul(xx.float().to(device))
                    .view(batch_size, 1, height * width)
                    .sum(2)
                    .unsqueeze(2)
            )
            tmp_approx_y = (
                tmp_headmap_softmax.mul(yy.float().to(device))
                    .view(batch_size, 1, height * width)
                    .sum(2)
                    .unsqueeze(2)
            )

            # print(tmp_approx_x,tmp_approx_y)
            output[i,0] = tmp_approx_x
            output[i,1] = tmp_approx_y

        # print(output)
        output = output.unsqueeze(0)
        return output
        # output = [approx_x, approx_y] if self.return_xy else [approx_y, approx_x]
        # output = torch.cat(output, 2)
        # return output