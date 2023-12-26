import torch
import torch.nn.functional as F


# reparameterize
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# cal loss
# mse
def utility_loss(y_pred_logits, y):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(y_pred_logits, y)


#reconstrucion loss
def rec_loss(input, output):
    return F.mse_loss(output, input, reduction='mean')


# weighted binary cross entropy
def wce_loss(y_pred, y, weight):
    return weight * F.binary_cross_entropy(y_pred, y, reduction='mean')


# kl divergence between prior(Normal) and encoder
def kl_div_for_gaussian(z_mean, z_log_sigma_sq):
    # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + z_log_sigma_sq - z_mean.pow(2) - z_log_sigma_sq.exp(), dim=1)
    return kl_div.mean()


