"""ENet Model for Semantic Segmentation Script implemented in PyTorch."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from nodes import ENet
from functions import decode_segmap
from params import root_path


# TODO: Implement resource setup to run locally
enet = ENet(12)  # instantiate a 12 class ENet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enet = enet.to(device)

# make sure you have the model renamed in the path before executing this
state_dict = torch.load(f"{root_path}/ckpt-enet-5.pth")["state_dict"]
enet.load_state_dict(state_dict)

# TODO: make the inference process interactive and easy to test
fname = "Seq05VD_f05100.png"
tmg_ = plt.imread(f"{root_path}/test/" + fname)
tmg_ = cv2.resize(tmg_, (512, 512), cv2.INTER_NEAREST)
tmg = torch.tensor(tmg_).unsqueeze(0).float()
tmg = tmg.transpose(2, 3).transpose(1, 2).to(device)

enet.to(device)
with torch.no_grad():
    out1 = enet(tmg.float()).squeeze(0)

# load the labeled (inferred) image
smg_ = Image.open(f"{root_path}/testannot/" + fname)
smg_ = cv2.resize(np.array(smg_), (512, 512), cv2.INTER_NEAREST)

# move the output to cpu TODO: why?
out2 = out1.cpu().detach().numpy()
mno = 8  # Should be between 0 - n-1 | where n is the number of classes

figure = plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.axis("off")
plt.imshow(tmg_)
plt.subplot(1, 3, 2)
plt.title("Output Image")
plt.axis("off")
plt.imshow(out2[mno, :, :])
plt.show()

# get class labels from the output
b_ = out1.data.max(0)[1].cpu().numpy()

# decode the images
true_seg = decode_segmap(smg_)
pred_seg = decode_segmap(b_)

# plot the decoded segments
figure = plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.axis("off")
plt.imshow(tmg_)
plt.subplot(1, 3, 2)
plt.title("Predicted Segmentation")
plt.axis("off")
plt.imshow(pred_seg)
plt.subplot(1, 3, 3)
plt.title("Ground Truth")
plt.axis("off")
plt.imshow(true_seg)
plt.show()
