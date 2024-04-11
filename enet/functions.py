import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from params import root_path


def loader(training_path, segmented_path, batch_size, h=320, w=1000):
    filenames_t = os.listdir(training_path)
    total_files_t = len(filenames_t)

    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)

    assert total_files_t == total_files_s

    if str(batch_size).lower() == "all":
        batch_size = total_files_s

    # idx = 0
    while 1:
        # Choosing random indexes of images and labels
        batch_idxs = np.random.randint(0, total_files_s, batch_size)

        inputs = []
        labels = []

        for jj in batch_idxs:
            # Reading normalized photo
            img = plt.imread(training_path + filenames_t[jj])
            # Resizing using nearest neighbor method
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            inputs.append(img)

            # Reading semantic image
            img = Image.open(segmented_path + filenames_s[jj])
            img = np.array(img)
            # Resizing using nearest neighbor method
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            labels.append(img)

        inputs = np.stack(inputs, axis=2)
        # Changing image format to C x H x W
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)

        labels = torch.tensor(labels)

        yield inputs, labels


def get_class_weights(num_classes, c=1.02):
    pipe = loader(f"{root_path}/train/", f"{root_path}/trainannot/", batch_size="all")
    _, labels = next(pipe)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights


def decode_segmap(image):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colours = np.array(
        [
            Sky,
            Building,
            Pole,
            Road_marking,
            Road,
            Pavement,
            Tree,
            SignSymbol,
            Fence,
            Car,
            Pedestrian,
            Bicyclist,
        ]
    ).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for n in range(0, 12):
        r[image == n] = label_colours[n, 0]
        g[image == n] = label_colours[n, 1]
        b[image == n] = label_colours[n, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb
