import csv
import os
import torch
from torch import nn, optim
from utils import discriminator
from utils import generator
from utils import dataloader
from matplotlib import pyplot as plt

def train():
    train_loader = dataloader.data_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_dim = 100
    ndf, ngf = 128, 128
    num_epochs = 100
    num_classes = 10
    # 十个类别

    netD = discriminator.Discriminator(ndf, num_classes).to(device)
    netG = generator.Generator(ngf, noise_dim, num_classes).to(device)
    # 初始化判别器和生成器

    criterion = nn.BCELoss().to(device)

    train_csv = '../mnist_train.csv'
    with open(train_csv, 'w', newline='') as f:
        fieldnames = ['Epoch', 'd_loss', 'g_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # 创建用于记录损失的CSV文件

    optimizer_D = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # 使用Adam优化器优化判别器和生成器

    print("Start Training")

    for epoch in range(num_epochs):

        loss_D, loss_G = 0, 0

        for i, data in enumerate(train_loader, 0):

            netD.zero_grad()
            real_images = data[0].to(device)
            real_labels = data[1].to(device)
            batch_size = real_images.size(0)
            # 获取真实图像和标签

            real_targets = torch.full((batch_size,), 0.9, device=device)
            # 创建一个全0.9张量 标签平滑

            output_real = netD(real_images, real_labels)
            loss_d_real = criterion(output_real, real_targets)
            loss_d_real.backward()
            # 计算真实样本的损失并反向传播

            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = netG(noise, fake_labels)
            # 生成随机噪声和随机标签进入生成器生成假样本

            fake_targets = torch.full((batch_size,), 0.0, device=device)
            output_fake = netD(fake_images.detach(), fake_labels)
            loss_d_fake = criterion(output_fake, fake_targets)
            loss_d_fake.backward()
            # 创建一个全0张量 将生成的假样本进入判别器 计算损失反向传播

            optimizer_D.step()

            real_targets.fill_(1.0)
            # 生成器希望判别器认为假样本为真 所以标签需要使用1

            output_fake = netD(fake_images, fake_labels)
            loss_g = criterion(output_fake, real_targets)
            loss_g.backward()
            optimizer_G.step()
            # 计算生成器的损失并反向传播 更新参数

            loss_D += (loss_d_real.item() + loss_d_fake.item())
            loss_G += loss_g.item()

        loss_mean_d = loss_D / len(train_loader)
        loss_mean_g = loss_G / len(train_loader)

        print(f'Epoch: {epoch+1}, Loss_D: {loss_mean_d:.4f}, Loss_G: {loss_mean_g:.4f}')

        with open(train_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch+1, 'd_loss': loss_mean_d, 'g_loss': loss_mean_g})
            # 将当前epoch的损失写入文件

        if (epoch + 1) % 10 == 0:
            save_images(epoch + 1, netG, device, num_classes)

    if not os.path.exists('../model'):
        os.makedirs('../model')
    torch.save(netD.state_dict(), '../model/netD.pth')
    torch.save(netG.state_dict(), '../model/netG.pth')

    return netG

def save_images(epoch, netG, device, num_classes=10):

    noise = torch.randn(num_classes, 100, 1, 1, device=device)
    labels = torch.arange(0, num_classes, device=device, dtype=torch.long)

    with torch.no_grad():
        fake_images = netG(noise, labels)

    fake_images = fake_images.cpu().detach().numpy()

    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            img = fake_images[idx, 0, :, :]
            ax[i, j].imshow(img, cmap='gray')
            ax[i, j].set_title(f"Label: {labels[idx].item()}")
            ax[i, j].axis('off')

    if not os.path.exists('../images'):
        os.makedirs('../images')
    plt.savefig(f'../images/epoch_{epoch}.png', bbox_inches='tight')
    plt.close()
    # 生成并保存每个标签下的图像

if __name__ == '__main__':
    netG = train()
    save_images(100, netG, device='cuda' if torch.cuda.is_available() else 'cpu')
    # 保存最终样本
