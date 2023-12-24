import torch
import torch.nn.functional as F


# reparameterize
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# cal loss
# mse
def utility_loss(label_true, label_pred):
    return F.mse_loss(label_pred, label_true, reduction='mean')


#reconstrucion loss
def rec_loss(input, output):
    return F.mse_loss(output, input, reduction='mean')


# weighted binary cross entropy
def wce_loss(y_pred, y, coef=torch.ones()):
    return F.binary_cross_entropy(y_pred, y, weight=coef, reduction='mean')


# kl divergence between prior(Normal) and encoder
def kl_div_for_gaussian(z_mean, z_log_sigma_sq):
    # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + z_log_sigma_sq - z_mean.pow(2) - z_log_sigma_sq.exp(), dim=1)
    return kl_div.mean()

