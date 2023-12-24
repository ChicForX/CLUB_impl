import torch
from config import config_dict
from dataset import train_loader, test_loader
from model import CLUB
import time
from layer_utils import reparameterize, utility_loss, rec_loss, wce_loss, kl_div_for_gaussian
from s_evaluator import S_Evaluator
import os
import optim

# hyperparams
lr = config_dict['lr']
dim_img = config_dict['dim_img']
dim_z = config_dict['dim_z']
dim_s = config_dict['dim_s']
dim_u = config_dict['dim_u']
dim_noise = config_dict['dim_noise']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = config_dict['epochs']
alpha = config_dict['alpha']
beta = config_dict['beta']

# train
def train(model, optimizer_encoder, optimizer_utility_decoder, optimizer_uncertainty_decoder,
          optimizer_prior_generator, optimizer_z_discriminator, optimizer_utility_discriminator):
    for epoch in range(epochs):
        for i, (x, u, s) in train_loader:  # imgs, class, colors
            # Training consists of 5 steps
            # ----1. train encoder, utility decoder & uncertainty decoder ----#
            optimizer_encoder.zero_grad()
            optimizer_utility_decoder.zero_grad()
            optimizer_uncertainty_decoder.zero_grad()

            z_mean, z_log_sigma_sq = model.encoder(x)
            z = reparameterize(z_mean, z_log_sigma_sq)
            u_utility = model.utility_decoder(z)
            x_obf = model.uncertainty_decoder(z, s)

            reconstruction_loss = rec_loss(x, x_obf)
            u_loss = utility_loss(u, u_utility)
            kl_loss = kl_div_for_gaussian(z_mean, z_log_sigma_sq)
            total_loss_1 = reconstruction_loss + u_loss + alpha * kl_loss

            total_loss_1.backward()
            optimizer_encoder.step()
            optimizer_utility_decoder.step()
            optimizer_uncertainty_decoder.step()

            # -------------------- 2. train z discriminator ------------------#
            optimizer_z_discriminator.zero_grad()

            noise = torch.randn(dim_noise)
            z_hat = model.prior_generator(noise)
            z_mean, z_log_sigma_sq = model.encoder(x)
            z = reparameterize(z_mean, z_log_sigma_sq)
            d_hat = model.z_discriminator(z_hat)
            d = model.z_discriminator(z)

            total_loss_2 = wce_loss(d_hat, d, beta + alpha)

            total_loss_2.backward()
            optimizer_z_discriminator.step()

            # ------- 3. train encoder & prior generator adversarially -------#
            optimizer_encoder.zero_grad()
            optimizer_prior_generator.zero_grad()

            z_hat = model.prior_generator(noise)
            z_mean, z_log_sigma_sq = model.encoder(x)
            z = reparameterize(z_mean, z_log_sigma_sq)
            d_hat = model.z_discriminator(z_hat)
            d = model.z_discriminator(z)

            total_loss_3 = -wce_loss(d_hat, d, beta + alpha)

            total_loss_3.backward()
            optimizer_encoder.step()
            optimizer_prior_generator.step()

            # ---------------- 4. train utility discriminator ----------------#
            optimizer_utility_discriminator.zero_grad()

            z_hat = model.prior_generator(noise)
            u_hat = model.utility_decoder(z_hat)
            d_hat = model.utility_discriminator(u_hat)
            d = model.utility_discriminator(u)

            total_loss_4 = wce_loss(d_hat, d)

            total_loss_4.backward()
            optimizer_utility_discriminator.step()

            # --- 5. train prior generator & utility decoder adversarially ---#
            optimizer_prior_generator.zero_grad()
            optimizer_utility_decoder.zero_grad()

            z_hat = model.prior_generator(noise)
            u_hat = model.utility_decoder(z_hat)
            d = model.utility_discriminator(u_hat)

            total_loss_5 = -wce_loss(d, torch.zeros_like(d))  # log(1-D)

            total_loss_5.backward()
            optimizer_prior_generator.step()
            optimizer_utility_decoder.step()

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Enc Loss: {total_loss_1:.4f}, "
                  f"Util Dec Loss: {total_loss_2:.4f}, "
                  f"Uncert Dec Loss: {total_loss_3:.4f}, "
                  f"Prior Gen Loss: {total_loss_4:.4f}, "
                  f"Util Disc Loss: {total_loss_5:.4f}")

# test
def test(model, test_loader):
    model.eval()

    with torch.no_grad():
        total_utility_accuracy = 0
        total_sensitivity_accuracy = 0

        for x, u, s in test_loader:
            z_test = model.encoder(x)
            x_hat = model.uncertainty_decoder(z_test, s)
            u_hat = model.utility_decoder(z_test)

            # TODO
            # utility

            # sensitivity
            # train s evaluator
            s_evaluator = S_Evaluator(dim_img, dim_s)
            train_s_evaluator(s_evaluator, train_loader, test_loader, dim_img, dim_s)

            # cal sensitivity acc

            # cal sensitivity mae

            # mutual information


        return

def train_s_evaluator(model, train_loader, test_loader, dim_img, dim_s):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model_path = f"saved_models/mnist_eval_model_s.pth"
    force_train = False

    if not os.path.exists(model_path) or force_train:
        print("Training sensitive attribute evaluator")
        model.train()
        for epoch in range(100):
            for x_batch, s_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, s_batch)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), model_path)
    else:
        print("Loading sensitive attribute evaluator from file")
        model.load_state_dict(torch.load(model_path))

    # Evaluating the model on the test set
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        for x_batch, _, s_batch in test_loader:
            s_hat_test = model(x_batch)
            evaluator_accuracy = (torch.argmax(s_hat_test, dim=-1) == s_batch).float().mean().item()
            total_accuracy += evaluator_accuracy
        average_accuracy = total_accuracy / len(test_loader)
        print(f"Evaluator accuracy = {average_accuracy * 100}")




if __name__ == '__main__':
    # init
    model = CLUB(dim_z, dim_u, dim_noise, dim_img)
    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    optimizer_utility_decoder = torch.optim.Adam(model.utility_decoder.parameters(), lr=lr)
    optimizer_uncertainty_decoder = torch.optim.Adam(model.uncertainty_decoder.parameters(), lr=lr)
    optimizer_prior_generator = torch.optim.Adam(model.prior_generator.parameters(), lr=lr)
    optimizer_z_discriminator = torch.optim.Adam(model.z_discriminator.parameters(), lr=lr)
    optimizer_utility_discriminator = torch.optim.Adam(model.utility_discriminator.parameters(), lr=lr)

    # train & validate
    train(model, optimizer_encoder, optimizer_utility_decoder, optimizer_uncertainty_decoder,
          optimizer_prior_generator, optimizer_z_discriminator, optimizer_utility_discriminator)

    # test
    test(model, test_loader)
