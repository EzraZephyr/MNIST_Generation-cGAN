import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngf, noise_dim, num_classes):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes

        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ELU(True),

            nn.ConvTranspose2d(ngf, 1, 5, 1, 0, bias=False),
            nn.Tanh()
        )
        # 将输入的随机噪声拼接上标签的编码 使用ELU激活函数
        # 最后一层通过Tanh将输出归一化到-1到1的区间让生成的图像与输入数据的分布一致

    def forward(self, noise, labels):

        batch_size = labels.size(0)
        label_onehot = torch.zeros(batch_size, self.num_classes, 1, 1, device=labels.device)
        # 创建全0张量并扩展成和图像的空间维度相同

        label_onehot[torch.arange(batch_size), labels] = 1
        # 通过索引选择批次中每个图像对应的标签位置设为1 形成onehot编码

        input = torch.cat([noise, label_onehot], dim=1)
        output = self.main(input)
        # 将噪声张量与onehot编码的标签在通道维度拼接 一起进行传播

        output = output[:, :, :28, :28]
        return output
