import os
from glob import glob
from pathlib import Path
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.config = config
        self.image_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
        ])

        if is_train:
            if category:
                self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "train", "good", "*.png"))
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)

        # 若是灰階，擴充成三通道
        if image.shape[0] == 1:
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)

        if self.is_train:
            label = 'good'
            return image, label

        # ====== 測試階段 ======
        if self.config.data.mask:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                label = 'good'
            else:
                # 這邊支援 _mask.png 或一般 .png
                mask_path_1 = image_file.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
                mask_path_2 = image_file.replace("/test/", "/ground_truth/")  # 無 _mask 的版本

                # 若第一個不存在就嘗試第二個，若都不存在則用空 mask
                if os.path.exists(mask_path_1):
                    target = Image.open(mask_path_1)
                elif os.path.exists(mask_path_2):
                    target = Image.open(mask_path_2)
                else:
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])

                # 若 target 是 PIL 圖就轉 tensor
                if isinstance(target, Image.Image):
                    target = self.mask_transform(target)
                label = 'defective'
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                label = 'good'
            else:
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                label = 'defective'

        return image, target, label

    def __len__(self):
        return len(self.image_files)
