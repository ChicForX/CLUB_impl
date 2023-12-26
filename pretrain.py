import torch
import os
import torch.optim as optim
import torch.nn as nn
from layer_utils import kl_div_for_gaussian


def pre_train_model(model, train_loader, dim_z, alpha, beta, device, pretrain_epoch=20, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_criterion = nn.MSELoss()
    utility_criterion = nn.CrossEntropyLoss()

    info_str = f"d_{dim_z}_beta_{beta}_alpha_{alpha}"
    model_path = f"saved_models/club_pretrain_{info_str}.pth"

    if not os.path.exists(model_path):
        print(f"Pre-Training with {info_str}")
        model.train()
        for epoch in range(pretrain_epoch):
            for x_batch, u_batch, s_batch in train_loader:
                x_batch, u_batch, s_batch = x_batch.to(device), u_batch.to(device), s_batch.to(device)
                optimizer.zero_grad()

                z_mean, z_log_sigma_sq = model.encoder(x_batch)
                u_hat = model.utility_decoder(z_mean)
                x_hat = model.uncertainty_decoder(z_mean, s_batch)

                reconstruction_loss = reconstruction_criterion(x_hat, x_batch)
                utility_loss = utility_criterion(u_hat, u_batch)
                kl_loss = kl_div_for_gaussian(z_mean, z_log_sigma_sq)

                total_loss = reconstruction_loss + utility_loss + alpha * kl_loss

                total_loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {total_loss.item()}")

        torch.save(model.state_dict(), model_path)
    else:
        print(f"Loading model from file with {info_str}")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
