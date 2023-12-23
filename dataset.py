import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_images = train_data.data.numpy()
test_images = test_data.data.numpy()

x_train = np.zeros((train_images.shape[0], train_images.shape[1], train_images.shape[2], 3), dtype=np.uint8)
x_test = np.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 3), dtype=np.uint8)

s_train = np.zeros(train_images.shape[0], dtype=np.uint8)
s_test = np.zeros(test_images.shape[0], dtype=np.uint8)

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

# split
x_train, x_val, s_train, s_val = train_test_split(x_train, s_train, test_size=0.2, random_state=42)

# numpy to tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
s_train_tensor = torch.tensor(s_train, dtype=torch.long)
s_val_tensor = torch.tensor(s_val, dtype=torch.long)
s_test_tensor = torch.tensor(s_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(x_train_tensor, s_train_tensor)
val_dataset = torch.utils.data.TensorDataset(x_val_tensor, s_val_tensor)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, s_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
