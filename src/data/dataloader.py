import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os.path as P
import pandas as pd
import random
from PIL import Image
from typing import Optional
import pytorch_lightning as pl
from torchvision.transforms.transforms import Resize


class CustomLoader(Dataset):
    def __init__(self, data_dir, transform, stage, same_percentage=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transform
        self.stage = stage
        self.same_percentage = same_percentage
        self.resize = transforms.Resize(size=(220, 180))

        if stage == "fit" or stage == "val":
            self.image_dir = P.join(data_dir, "train")
            self.landmarks_dir = P.join(data_dir, "trainK_68")
            self.mask_dir = P.join(data_dir, "train_mask")
            pairs_file_1 = P.join(data_dir, "celeba-pairs-train.csv")
            pairs_file_2 = P.join(data_dir, "celeba-pairs-train-sameid.csv")

            pairs_file_train_diff = pd.read_csv(pairs_file_1, dtype=str)
            self.size = len(pairs_file_train_diff)
            self.pairs_diff = []
            for i in range(self.size):
                pair = [
                    pairs_file_train_diff.iloc[i]["from"],
                    pairs_file_train_diff.iloc[i]["to"],
                ]
                self.pairs_diff.append(pair)

            pairs_file_train_same = pd.read_csv(pairs_file_2, dtype=str)
            self.pairs_same = []
            for i in range(self.size):
                pair = [
                    pairs_file_train_same.iloc[i]["from"],
                    pairs_file_train_same.iloc[i]["to"],
                ]
                self.pairs_same.append(pair)
        elif stage == "test":
            self.image_dir = P.join(data_dir, "test")
            self.landmarks_dir = P.join(data_dir, "testK_68")
            self.mask_dir = P.join(data_dir, "test_mask")
            pairs_file_1 = P.join(data_dir, "celeba-pairs-test.csv")

            pairs_file_test = pd.read_csv(pairs_file_1, dtype=str)
            self.size = len(pairs_file_test)
            self.pairs_test = []
            for i in range(self.size):
                pair = [
                    pairs_file_test.iloc[i]["from"],
                    pairs_file_test.iloc[i]["to"],
                ]
                self.pairs_test.append(pair)

    def __getitem__(self, index):
        same_identity = False
        if self.stage == "fit":
            index = random.randint(0, self.size - 1)
            p = random.random()
            if p < self.same_percentage:
                P1_name, P2_name = self.pairs_same[index]
                same_identity = True
            else:
                P1_name, P2_name = self.pairs_diff[index]
        elif self.stage == "test":
            P1_name, P2_name = self.pairs_test[index]
        else:
            index = random.randint(0, 4000)
            P1_name, P2_name = self.pairs_same[index]

        P1_image_path = P.join(self.image_dir, P1_name + ".jpg")
        P1_landmarks_path = P.join(self.landmarks_dir, P1_name + ".pt")
        P2_image_path = P.join(self.image_dir, P2_name + ".jpg")
        P2_landmarks_path = P.join(self.landmarks_dir, P2_name + ".pt")
        P2_mask_path = P.join(self.mask_dir, P2_name + ".jpg")

        P1_img = Image.open(P1_image_path).convert("RGB")
        P2_img = Image.open(P2_image_path).convert("RGB")

        BP1_img = torch.load(P1_landmarks_path)
        BP2_img = torch.load(P2_landmarks_path)

        P2_mask = Image.open(P2_mask_path).convert("RGB")

        P1 = self.transforms(P1_img)
        P2 = self.transforms(P2_img)
        BP1 = self.resize(BP1_img.float())
        BP2 = self.resize(BP2_img.float())

        mask = self.transforms(P2_mask)
        msk = torch.zeros_like(mask)
        msk[mask > 0] = 1

        return {
            "P1": P1,
            "BP1": BP1,
            "P2": P2,
            "BP2": BP2,
            "masked_P2": P2 * (1 - msk),
            "P1_path": P1_name,
            "P2_path": P2_name,
            "same": same_identity,
            "mask": mask,
        }

    def __len__(self):
        if self.stage == "test":
            return self.size
        if self.stage == "fit":
            return 4096
        else:
            return 1


class AnonyDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=32, same_percentage=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(220, 180)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.bs = batch_size
        self.same_percentage = same_percentage

    def setup(self, stage: Optional[str] = None):
        pass
        # print("STAGE", stage)

        # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        #     self.data_train = CustomLoader(self.data_dir, self.transform, stage)

        # if stage == "validation" or stage is None:
        #     self.data_val = CustomLoader(self.data_dir, self.transform, stage)

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.data_test = CustomLoader(self.data_dir, self.transform, stage)

    def train_dataloader(self):
        data_train = CustomLoader(
            self.data_dir, self.transform, "fit", same_percentage=self.same_percentage
        )
        return DataLoader(
            data_train, batch_size=self.bs, num_workers=4, pin_memory=False
        )

    def val_dataloader(self):
        data_val = CustomLoader(self.data_dir, self.transform, "val")
        return DataLoader(data_val, batch_size=1, num_workers=4)

    def test_dataloader(self):
        data_test = CustomLoader(self.data_dir, self.transform, "test")
        return DataLoader(data_test, batch_size=1)
