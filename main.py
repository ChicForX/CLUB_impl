import torch
from config import config_dict
from dataset import train_loader, val_loader
from model import CLUB
import time

# hyperparams
lr = config_dict['lr']
dim_img = config_dict['dim_img']
dim_z = config_dict['dim_z']
dim_s = config_dict['dim_s']
dim_u = config_dict['dim_u']
dim_noise = config_dict['dim_noise']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = config_dict['epochs']

# train
def train():
    start_time = time.time()
    for epoch in range(epochs):
        start_time_epoch = time.time()
        for imgs in train_loader:
            # TODO

            # 1. train encoder, utility decoder & uncertainty decoder

            # 2. train z discriminator

            # 3. train encoder & prior generator adversarially

            # 4. train utility discriminator

            # 5. train prior generator & utility decoder adversarially


        print(f"One epoch execution time: {(time.time() - start_time_epoch):.5f} seconds")

# validate
def validate():
    # TODO
    return


if __name__ == '__main__':
    # init
    model = CLUB(dim_z, dim_u, dim_noise, dim_img)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train

    # validate

