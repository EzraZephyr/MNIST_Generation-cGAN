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
        # Concatenate the input random noise with the encoded label, using ELU activation function
        # The final layer applies Tanh to normalize the output between -1 and 1,
        # aligning the generated image distribution with that of the input data

    def forward(self, noise, labels):

        batch_size = labels.size(0)
        label_onehot = torch.zeros(batch_size, self.num_classes, 1, 1, device=labels.device)
        # Create a tensor of zeros and expand it to match the spatial dimensions of the image

        label_onehot[torch.arange(batch_size), labels] = 1
        # Use indexing to set the position corresponding to each image's label in the batch to 1,
        # forming a one-hot encoding

        input = torch.cat([noise, label_onehot], dim=1)
        output = self.main(input)
        # Concatenate the noise tensor with the one-hot encoded labels along the channel dimension for joint propagation

        output = output[:, :, :28, :28]
        return output
