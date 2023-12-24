import torch
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_images = train_data.data.numpy()
test_images = test_data.data.numpy()

# add color channels
x_train = np.zeros((train_images.shape[0], train_images.shape[1], train_images.shape[2], 3), dtype=np.uint8)
x_test = np.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 3), dtype=np.uint8)

# color labels: sensitivity
s_train = np.zeros(train_images.shape[0], dtype=np.uint8)
s_test = np.zeros(test_images.shape[0], dtype=np.uint8)

# class labels: utility
u_train = train_data.targets.numpy()
u_test = test_data.targets.numpy()

def assign_color_channels(data, color_labels):
    num_samples = data.shape[0]
    idx = np.random.permutation(num_samples)
    split = num_samples // 3
    data[idx[:split], :, :, 0] = data[idx[:split]]
    data[idx[split:2*split], :, :, 1] = data[idx[split:2*split]]
    data[idx[2*split:], :, :, 2] = data[idx[2*split:]]
    color_labels[idx[:split]] = 0
    color_labels[idx[split:2*split]] = 1
    color_labels[idx[2*split:]] = 2

assign_color_channels(x_train, s_train)
assign_color_channels(x_test, s_test)

# numpy to tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
s_train_tensor = torch.tensor(s_train, dtype=torch.long)
s_test_tensor = torch.tensor(s_test, dtype=torch.long)
u_train_tensor = torch.tensor(u_train, dtype=torch.long)
u_test_tensor = torch.tensor(u_test, dtype=torch.long)

# dataset
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, u_train_tensor, s_train_tensor)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, u_test_tensor, s_test_tensor)

# data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
