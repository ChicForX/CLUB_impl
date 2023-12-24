import torch
import torchvision.utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# evaluation
# accuracy
def cal_accuracy(y_pred, y):
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct = (predicted_classes == y).float()
    accuracy = correct.mean()
    return accuracy * 100

# save figures


# sensitivity
def cal_sens_mae():





# mutual information between u and z

# mutual information between s and z

# save img by 5*5
def save_images(x_hat, filename, nrow=5, ncol=5):
    x_hat = x_hat[:nrow * ncol]

    x_hat = x_hat.cpu().detach()
    grid = torchvision.utils.make_grid(x_hat, nrow=nrow)

    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = transforms.ToPILImage()(ndarr)
    im.save(filename)