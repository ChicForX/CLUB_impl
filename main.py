import torch
import torch.nn.functional as F
from config import config_dict
from dataset import train_loader, test_loader
from model import CLUB
from layer_utils import reparameterize, utility_loss, rec_loss, wce_loss, kl_div_for_gaussian
from s_evaluator import S_Evaluator
from pretrain import pre_train_model
import eval_utils as eval
from mine import MINE
import os

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
pretrain_epoch = config_dict['pretrain_epoch']
lr_pretrain = config_dict['lr_pretrain']
eval_path = config_dict['eval_path']
mi_epochs = config_dict['mi_epochs']
batch_size = config_dict['batch_size']


# train
def train(model, optimizer_encoder, optimizer_utility_decoder, optimizer_uncertainty_decoder,
          optimizer_prior_generator, optimizer_z_discriminator, optimizer_utility_discriminator):
    for epoch in range(epochs):
        model.train()

        for i, (x, u, s) in enumerate(train_loader):  # imgs, class, colors
            x, u, s = x.to(device), u.to(device), s.to(device)

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
            u_loss = utility_loss(u_utility, u)
            kl_loss = kl_div_for_gaussian(z_mean, z_log_sigma_sq)
            total_loss_1 = reconstruction_loss + u_loss + alpha * kl_loss

            total_loss_1.backward()
            optimizer_encoder.step()
            optimizer_utility_decoder.step()
            optimizer_uncertainty_decoder.step()

            # -------------------- 2. train z discriminator ------------------#
            optimizer_z_discriminator.zero_grad()

            noise = torch.randn(x.size(0), dim_noise).to(device)
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
            d = model.utility_discriminator(F.one_hot(u, num_classes=10).float())  # batch_size, 10

            total_loss_4 = wce_loss(d_hat, d, alpha+beta)

            total_loss_4.backward()
            optimizer_utility_discriminator.step()

            # --- 5. train prior generator & utility decoder adversarially ---#
            optimizer_prior_generator.zero_grad()
            optimizer_utility_decoder.zero_grad()

            z_hat = model.prior_generator(noise)
            u_hat = model.utility_decoder(z_hat)
            d = model.utility_discriminator(u_hat)

            total_loss_5 = -wce_loss(d, torch.zeros_like(d), alpha+beta)  # log(1-D)

            total_loss_5.backward()
            optimizer_prior_generator.step()
            optimizer_utility_decoder.step()

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Enc Loss: {total_loss_1:.4f}, "
                  f"Util Dec Loss: {total_loss_2:.4f}, "
                  f"Uncert Dec Loss: {total_loss_3:.4f}, "
                  f"Prior Gen Loss: {total_loss_4:.4f}, "
                  f"Util Disc Loss: {total_loss_5:.4f}")

        model.eval()
        with torch.no_grad():
            if not os.path.exists(eval_path):
                os.makedirs(eval_path, exist_ok=True)
            # save imgs of each epoch
            eval.save_images(x, x_obf, filename=f"{eval_path}reconst-{epoch + 1}.png")


# test
def tst(model, test_loader):
    model.eval()

    # train s evaluator
    s_evaluator = S_Evaluator(dim_img, dim_s).to(device)
    eval.train_s_evaluator(s_evaluator, train_loader, test_loader, device)

    # train MINE
    mine = MINE().to(device)
    eval.train_mine(model, mine, test_loader, mi_epochs, device)

    with torch.no_grad():
        total_utility_acc = 0
        total_sensitivity_acc = 0
        total_sens_mae = 0
        total_mi_u_z = 0
        total_mi_s_z = 0

        for x, u, s in test_loader:
            x, u, s = x.to(device), u.to(device), s.to(device)
            z_mean, z_log_sigma_sq = model.encoder(x)
            z_test = reparameterize(z_mean, z_log_sigma_sq)
            x_hat = model.uncertainty_decoder(z_test, s)
            u_hat = model.utility_decoder(z_test)

            # utility accuracy
            total_utility_acc += eval.cal_accuracy(u_hat, u)

            # sensitivity
            s_hat = s_evaluator(x_hat)
            # cal sensitivity acc
            total_sensitivity_acc += eval.cal_sens_acc(s_hat, s)
            # cal sensitivity mae
            total_sens_mae += eval.cal_sens_mae(s_hat, s)

            # mutual information
            # s & z
            total_mi_s_z += mine.get_mi(z_test, s)
            # u & z
            total_mi_u_z += mine.get_mi(z_test, u)


        # save imgs
        eval.save_images(x, x_hat, filename=f"{eval_path}tst_output.png")

        eval.obs_reconstruction(x, s, model, filename=f"{eval_path}tst_obs_channel_reconst_epoch1.png", channel=0)

        avg_utility_acc = total_utility_acc / len(test_loader)
        avg_sensitivity_acc = total_sensitivity_acc / len(test_loader)
        avg_sensitivity_mae = total_sens_mae / len(test_loader)
        avg_mi_s_z = total_mi_s_z / len(test_loader)
        avg_mi_u_z = total_mi_u_z / len(test_loader)

        print(f"Utility Accuracy: {avg_utility_acc}")
        print(f"Sensitivity Accuracy: {avg_sensitivity_acc}")
        print(f"Mean Absolute Error: {avg_sensitivity_mae}")
        print(f"Mutual Information of s & z: {avg_mi_s_z}")
        print(f"Mutual Information of u & z: {avg_mi_u_z}")


if __name__ == '__main__':
    # init
    model = CLUB(dim_z, dim_u, dim_noise, dim_img, dim_s)
    model.to(device)
    pre_train_model(model, train_loader, dim_z, alpha, beta, device, pretrain_epoch, lr_pretrain)

    # init optimizers for training
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
    tst(model, test_loader)
