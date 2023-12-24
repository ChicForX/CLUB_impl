import torch


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

