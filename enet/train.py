import torch
import torch.nn as nn
from tqdm import tqdm
from nodes import ENet
from functions import get_class_weights, loader
from params import (
    root_path,
    lr,
    batch_size,
    print_every,
    eval_every,
    bc_train,
    bc_eval,
    epochs,
)

# Uncomment the following 2 lines for google colab
# !wget https://www.dropbox.com/s/pxcz2wdz04zxocq/CamVid.zip?dl=1 -O CamVid.zip
# !unzip CamVid.zip

# TODO: Implement resource setup to run locally

print("1. initiating... a 12 class ENet")
enet = ENet(12)  # instantiate a 12 class ENet
print("*checking for gpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enet = enet.to(device)

print("2. generating class weights")
class_weights = get_class_weights(12)

# defining hyper parameters
# print("3. defining hyper params")

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
optimizer = torch.optim.Adam(enet.parameters(), lr=lr, weight_decay=2e-4)

print("4. initiating training loop")
train_losses = []
eval_losses = []

# define pipeline objects
pipe = loader(f"{root_path}/train/", f"{root_path}/trainannot/", batch_size)
eval_pipe = loader(f"{root_path}/val/", f"{root_path}/valannot/", batch_size)

# train loop
for e in range(1, epochs + 1):
    train_loss = 0
    print("-" * 15, "Epoch %d" % e, "-" * 15)

    enet.train()

    for _ in tqdm(range(bc_train)):
        X_batch, mask_batch = next(pipe)

        # assign data to cpu/gpu
        X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()

        out = enet(X_batch.float())

        # loss calculation
        loss = criterion(out, mask_batch.long())
        # update weights
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print()
    train_losses.append(train_loss)

    if (e + 1) % print_every == 0:
        print("Epoch {}/{}...".format(e, epochs), "Loss {:6f}".format(train_loss))

    if e % eval_every == 0:
        with torch.no_grad():
            enet.eval()

            eval_loss = 0

            # Validation loop
            for _ in tqdm(range(bc_eval)):
                inputs, labels = next(eval_pipe)

                inputs, labels = inputs.to(device), labels.to(device)

                out = enet(inputs)

                out = out.data.max(1)[1]

                eval_loss += (labels.long() - out.long()).sum()

            print()
            print("Loss {:6f}".format(eval_loss))

            eval_losses.append(eval_loss)

    if e % print_every == 0:
        checkpoint = {"epochs": e, "state_dict": enet.state_dict()}
        torch.save(
            checkpoint, "{}/ckpt-enet-{}-{}.pth".format(root_path, e, train_loss)
        )
        print("Model saved!")

print(
    "Epoch {}/{}...".format(e, epochs),
    "Total Mean Loss: {:6f}".format(sum(train_losses) / epochs),
)
