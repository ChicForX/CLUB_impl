import torch

# reparameterize
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# cal loss
# mse
def utility_loss(label_true, label_pred):
    return F.mse_loss(label_pred, label_true, reduction='mean')

#reconstrucion loss
def rec_loss(input, output, alpha):
    return alpha * F.mse_loss(output, input, reduction='mean')

# weighted binary cross entropy
def wce_loss(y_pred, y, coef):
    return F.binary_cross_entropy(y_pred, y, weight=coef, reduction='mean')

