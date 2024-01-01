import torch
import torchvision.utils
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from config import config_dict

model_path = config_dict['model_path']
def show_mnist_images(loader):
    images, labels, colors = next(iter(loader))

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        # [height,width,channel]
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()


# evaluation
# accuracy
def cal_accuracy(y_pred, y):
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct = (predicted_classes == y).float()
    accuracy = correct.mean()
    return accuracy * 100


# sensitivity evaluator
def train_s_evaluator(model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()


    force_train = False

    if not os.path.exists(model_path) or force_train:
        print("Training sensitive attribute evaluator")
        model.train()
        for epoch in range(5):
            for x_batch, _, s_batch in train_loader:
                x_batch, s_batch = x_batch.to(device), s_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                s_batch = s_batch.view(-1, 1)
                loss = criterion(outputs, s_batch.float())
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}: Loss {loss.item()}")

        torch.save(model.state_dict(), model_path)
    else:
        print("Loading sensitive attribute evaluator from file")
        model.load_state_dict(torch.load(model_path))

    # Evaluating the model on the test set
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        for x_batch, _, s_batch in test_loader:
            x_batch, s_batch = x_batch.to(device), s_batch.to(device)
            s_hat_test = model(x_batch)
            evaluator_accuracy = (torch.argmax(s_hat_test, dim=-1) == s_batch).float().mean().item()
            total_accuracy += evaluator_accuracy
        average_accuracy = total_accuracy / len(test_loader)
        print(f"Evaluator accuracy = {average_accuracy * 100}")


# sensitivity mae
def cal_sens_mae(s_hat, s):
    absolute_error = torch.abs(s_hat - s)
    mae = torch.mean(absolute_error)
    return mae

# sensitivity accuracy
def cal_sens_acc(s_hat, s):
    predicted_classes = torch.argmax(s_hat, dim=1)
    correct_predictions = (predicted_classes == s)
    accuracy = torch.mean(correct_predictions.float()) * 100
    return accuracy


# mutual information
def train_mine(model, mine, train_loader, epochs, device):
    print("Training mine")
    optimizer = torch.optim.Adam(mine.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, u_batch, _ in train_loader:
            x_batch, u_batch = x_batch.to(device), u_batch.to(device)
            z_batch, _ = model.encoder(x_batch)
            total_loss += mine.train_mine_inside_epoch(z_batch, u_batch, optimizer)

        print(f"Epoch {epoch}, MI Estimate: {-total_loss / len(train_loader)}")


# save img by 5*5
def save_images(x_hat, filename, nrow=5, ncol=5):
    x_hat = x_hat[:nrow * ncol]

    x_hat = x_hat.cpu().detach()
    grid = torchvision.utils.make_grid(x_hat, nrow=nrow)

    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = transforms.ToPILImage()(ndarr)
    im.save(filename)
