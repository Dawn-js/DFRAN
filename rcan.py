import torch.nn as nn
import torch.nn.functional as F

# class CALayer(nn.Module):
#     def __init__(self, num_channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv1d(num_channels, num_channels//reduction, 1, 1, 0),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_channels//reduction, num_channels, 1, 1, 0),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
    
class CALayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 替换为1D自适应最大池化层
        self.conv_du = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.max_pool(x)  # 替换为1D自适应最大池化层
        y = F.interpolate(y, size=x.size()[-1], mode='nearest')  # 使用nn.Upsample函数进行上采样
        y = self.conv_du(y)
        return x * y



class RCAB(nn.Module):
    def __init__(self, num_channels, reduction, res_scale):
        super().__init__()

        body = [
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
        ]
        body.append(CALayer(num_channels, reduction))

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, num_channels, num_blocks, reduction, res_scale=1.0):
        super().__init__()

        body = list()
        for _ in range(num_blocks):
            body += [RCAB(num_channels, reduction, res_scale)]
        body += [nn.Conv1d(num_channels, num_channels, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res