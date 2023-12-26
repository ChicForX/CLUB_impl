import torch
import torch.nn as nn
import torch.nn.functional as F

# consists of 6 parts

# prior generator, g_psi(Normal)
class PriorGenerator(nn.Module):
    def __init__(self, DIM_Z, noise_dim=50):
        super(PriorGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, noise_dim*2)
        self.bn1 = nn.BatchNorm1d(noise_dim*2)
        self.fc2 = nn.Linear(noise_dim*2, noise_dim)
        self.bn2 = nn.BatchNorm1d(noise_dim)
        self.fc3 = nn.Linear(noise_dim, DIM_Z)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# z discriminator, d_eta(z)
class ZDiscriminator(nn.Module):
    def __init__(self, DIM_Z):
        super(ZDiscriminator, self).__init__()
        self.fc1 = nn.Linear(DIM_Z, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# encoder, p_phi(z|x)
class Encoder(nn.Module):
    def __init__(self, dim_z, dim_img):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # output of conv
        self.flat_size = 64 * 4 * 4  # [28 / (2 * 2 * 2)] round up

        self.fc = nn.Linear(self.flat_size, dim_z * 4)
        self.fc_bn = nn.BatchNorm1d(dim_z * 4)

        self.fc_mean = nn.Linear(dim_z * 4, dim_z)
        self.fc_log_sigma = nn.Linear(dim_z * 4, dim_z)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc_bn(self.fc(x)))

        z_mean = self.fc_mean(x)
        z_log_sigma_sq = self.fc_log_sigma(x)
        return z_mean, z_log_sigma_sq

# utility decoder, p_theta(u|z)
class UtilityDecoder(nn.Module):
    def __init__(self, DIM_Z, DIM_U):
        super(UtilityDecoder, self).__init__()
        self.fc1 = nn.Linear(DIM_Z, DIM_Z * 4)
        self.bn1 = nn.BatchNorm1d(DIM_Z * 4)
        self.fc2 = nn.Linear(DIM_Z * 4, DIM_U)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.softmax(self.fc2(x), dim=1)
        return x

# uncertainty decoder, p_xi(x|s,z)
class UncertaintyDecoder(nn.Module):
    def __init__(self, DIM_Z, DIM_S):
        super(UncertaintyDecoder, self).__init__()
        self.fc = nn.Linear(DIM_Z + DIM_S, 7 * 7 * 128)
        self.bn1 = nn.BatchNorm1d(7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2, padding=1)
    def forward(self, z, s):
        # Expand the dimensions of s to match the second dimension of z
        s_expanded = s.unsqueeze(1)
        # concat z, s
        z_with_s = torch.cat([z, s_expanded], dim=1)

        x = F.leaky_relu(self.bn1(self.fc(z_with_s)))
        x = x.view(-1, 128, 7, 7)    # reshape x for conv
        x = F.leaky_relu(self.bn2(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x))
        return x

# utility  discriminator, d_omega(u)
class UtilityDiscriminator(nn.Module):
    def __init__(self, DIM_U):
        super(UtilityDiscriminator, self).__init__()

        self.fc1 = nn.Linear(DIM_U, DIM_U * 4)
        self.bn1 = nn.BatchNorm1d(DIM_U * 4)
        self.fc2 = nn.Linear(DIM_U * 4, DIM_U * 4)
        self.bn2 = nn.BatchNorm1d(DIM_U * 4)
        self.fc3 = nn.Linear(DIM_U * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = torch.sigmoid(self.fc3(x))
        return x
