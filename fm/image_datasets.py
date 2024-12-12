import random

import cv2
# from mpi4py import MPI
import numpy as np
from torch.utils.data import Dataset
import os
join = os.path.join

# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

class ImageDataset(Dataset):
    def __init__(
        self,
        config,
        phase="train",
    ):
        super().__init__()
        self.config = config
        self.phase = phase
        self.img_size = config.img_size     # only used for training transform
        image_paths = config.train_path if phase == "train" else config.val_path
        image_paths_list = [image_paths] if isinstance(image_paths, str) else image_paths
        self.image_names = []
        for image_path in image_paths_list:
            if not os.path.exists(image_path):
                continue
            self.image_names.extend([os.path.join(image_path, image_name) for image_name in sorted(os.listdir(image_path))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_names[idx]
        img_H = imread_uint(path, self.config.in_channels)
        # apply resize
        if self.config.resize_size > 0: # currently only for downsampling in generation 
            height, width, _ = img_H.shape
            if height < width:
                new_height = self.config.resize_size
                new_width = int((width / height) * new_height)
            else:
                new_width = self.config.resize_size
                new_height = int((height / width) * new_width)
            # resize the image to the given size, the shorter side is resized to the given size
            # cv2.INTER_AREA is equivalent to PIL's Image.BOX
            img_H = cv2.resize(img_H, (new_width, new_height), interpolation=cv2.INTER_AREA)
        H, W, _ = img_H.shape
        if self.phase == "train":
            if self.config.random_crop: # apply random crop
                rnd_h = random.randint(0, max(0, H - self.img_size))
                rnd_w = random.randint(0, max(0, W - self.img_size))
                arr = img_H[rnd_h:rnd_h + self.img_size, rnd_w:rnd_w + self.img_size, :]
            elif self.config.center_crop: # center crop
                arr = img_H[(H - self.img_size) // 2:(H + self.img_size) // 2, 
                            (W - self.img_size) // 2:(W + self.img_size) // 2, :]
            else:   # no crop
                arr = img_H
        elif self.phase == "val":
            if self.config.random_crop_val: # apply random crop
                rnd_h = random.randint(0, max(0, H - self.config.val_img_size))
                rnd_w = random.randint(0, max(0, W - self.config.val_img_size))
                arr = img_H[rnd_h:rnd_h + self.config.val_img_size, rnd_w:rnd_w + self.config.val_img_size, :]
            elif self.config.center_crop_val: # center crop
                arr = img_H[(H - self.config.val_img_size) // 2:(H + self.config.val_img_size) // 2, 
                            (W - self.config.val_img_size) // 2:(W + self.config.val_img_size) // 2, :]
            else:   # no crop
                arr = img_H
        else:
            raise NotImplementedError
        # data augmentation is done in the training loop

        arr = arr.astype(np.float32) / 127.5 - 1 # [-1, 1]
        out_dict = {'img_name': os.path.splitext(os.path.basename(self.image_names[idx]))[0]}
        # import pdb; pdb.set_trace()
        return np.transpose(arr, [2, 0, 1]), out_dict

