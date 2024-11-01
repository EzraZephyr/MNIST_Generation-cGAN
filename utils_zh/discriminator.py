import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, ndf, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.Conv2d(1 + num_classes, ndf, 4, 2, 1, bias=False),  # 输入通道数修改为1 + num_classes
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # 将1维的灰度图像与标签的编码进行拼接 后通过多层卷积和激活函数逐步提取输入图像的特征 最终通过sigmoid输出一个概率值

    def forward(self, input, labels):
        batch_size, _, height, width = input.size()
        label_onehot = torch.zeros(batch_size, self.num_classes, height, width, device=labels.device)
        label_onehot.scatter_(1, labels.view(batch_size, 1, 1, 1).repeat(1, 1, height, width), 1)
        # 创建onehot编码 提取标签并调整为(batch_size,1,1,1) 将长宽扩展到图片真实的维度 用于拼接

        input_with_labels = torch.cat((input, label_onehot), dim=1)
        output = self.main(input_with_labels)
        return output.view(-1)
