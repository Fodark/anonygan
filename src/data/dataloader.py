import os
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
import numpy as np
import cv2
import json

legal_lndmk = [
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,27,28,29,30,60,61,62,63,64,65,66,67,
]
legal_mask = torch.tensor([False] * 68)
for i in legal_lndmk:
    legal_mask[i] = True


def create_landmark_and_mask(landmarks, lfw=False):
    final_tensor = (
        torch.zeros((68, 220, 180)) if not lfw else torch.zeros((68, 256, 256))
    )
    for idx, l in enumerate(landmarks):
        if l["y"] > 255:
            l["y"] = 255
        if l["x"] > 255:
            l["x"] = 255
        final_tensor[idx][l["y"]][l["x"]] = 1.0

    img_msk = (
        np.zeros((220, 180, 3), np.uint8)
        if not lfw
        else np.zeros((256, 256, 3), np.uint8)
    )
    contours = []
    # leftest jaw point, highest left eyebrow point
    first_x, first_y = landmarks[0]["x"], landmarks[19]["y"]
    # righest jaw point, highest right eyebrow point
    last_x, last_y = landmarks[16]["x"], landmarks[24]["y"]

    contours.append((first_x, first_y))
    # complete jawline
    for p in range(17):
        x, y = landmarks[p]["x"], landmarks[p]["y"]
        contours.append((x, y))

    contours.append((last_x, last_y))
    # fill the polygon defined by above points of white
    cv2.fillPoly(img_msk, pts=[np.array(contours)], color=(255, 255, 255))
    result_mask = Image.fromarray(img_msk)
    result_mask = transforms.ToTensor()(result_mask)

    return final_tensor, result_mask


class CustomLoader(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        stage,
        same_percentage=0.5,
        same_identity=False,
        reduced_landmarks=False,
        iciap=False,
        lfw=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transform
        self.stage = stage
        self.same_percentage = same_percentage
        self.same_identity = same_identity
        self.resize = (
            transforms.Resize(size=(220, 180))
            if not lfw
            else transforms.Resize(size=(256, 256))
        )
        self.reduced_landmarks = reduced_landmarks
        self.iciap = iciap
        self.lfw = lfw

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

            self.image_dir_fallback = P.join(data_dir, "train")
            self.landmarks_dir_fallback = P.join(data_dir, "trainK_68")
            self.mask_dir_fallback = P.join(data_dir, "train_mask")

            if not self.iciap:
                pairs_file_1 = P.join(data_dir, "celeba-pairs-test-new.csv")

                pairs_file_test = pd.read_csv(pairs_file_1, dtype=str)
                self.size = len(pairs_file_test)
                self.pairs_test = []
                for i in range(self.size):
                    pair = [
                        pairs_file_test.iloc[i]["from"],
                        pairs_file_test.iloc[i]["to"],
                    ]
                    self.pairs_test.append(pair)
            else:
                if not self.lfw:
                    data_root = "/media/dvl1/HDD_DATA/iciap/test_celeba"
                else:
                    data_root = "/media/dvl1/HDD_DATA/iciap/ciagan_lfw"
                self.data_root = data_root
                pairs_file_1 = P.join(data_root, "fair.csv")

                pairs_file_test = pd.read_csv(pairs_file_1, dtype=str)
                # self.size = len(pairs_file_test)
                self.pairs_test = []
                if not self.lfw:
                    for r in pairs_file_test.iterrows():
                        condition = r[1]["condition"]
                        source_id = r[1]["source_id"]

                        source_p = P.join(data_root, "sources", source_id)
                        for src in os.listdir(source_p):
                            source = src.split(".")[0]
                            pair = [condition, f"{source_id}/{source}"]
                            self.pairs_test.append(pair)
                else:
                    for r in pairs_file_test.iterrows():
                        condition = r[1]["condition"].split(".")[0]
                        source = r[1]["source_id"].split(".")[0]

                        pair = [condition, source]
                        self.pairs_test.append(pair)
                self.size = len(self.pairs_test)

    def __getitem__(self, index):
        same_identity = False
        if self.stage == "fit":
            index = random.randint(0, self.size - 1)
            p = random.random()
            if p < self.same_percentage:
                P1_name, P2_name = self.pairs_same[index]
                if self.same_identity:
                    P2_name = P1_name
                same_identity = True
            else:
                if not self.same_identity:
                    P1_name, P2_name = self.pairs_diff[index]
                else:
                    P1_name, P2_name = self.pairs_same[index]
        elif self.stage == "test":
            P1_name, P2_name = self.pairs_test[index]
        else:
            index = random.randint(0, 4000)
            P1_name, P2_name = self.pairs_same[index]

        if not self.iciap:
            P1_image_path = P.join(self.image_dir, P1_name + ".jpg")
            P1_landmarks_path = P.join(self.landmarks_dir, P1_name + ".pt")
            if not os.path.isfile(P1_image_path):
                P1_image_path = P.join(self.image_dir_fallback, P1_name + ".jpg")
                P1_landmarks_path = P.join(self.landmarks_dir_fallback, P1_name + ".pt")
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

            if self.reduced_landmarks:
                BP1 = BP1[legal_mask]
                BP2 = BP2[legal_mask]

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
        else:
            if not self.lfw:
                P2_basename = P2_name.split("/")[1]
                P1_image_path = P.join(
                    self.data_root, "condition", "fair/" + P1_name + ".png"
                )
                P2_image_path = P.join(self.data_root, "sources", P2_name + ".png")
            else:
                P1_image_path = P.join(
                    self.data_root, "lfw-only-second", P1_name + ".jpg"
                )
                P2_image_path = P.join(
                    self.data_root, "lfw-only-second", P2_name + ".jpg"
                )

            P1 = self.transforms(Image.open(P1_image_path).convert("RGB"))
            P2 = self.transforms(Image.open(P2_image_path).convert("RGB"))

            landmarks1 = json.load(
                open(P.join(self.data_root, "landmarks", P1_name + ".json"), "r")
            )
            landmarks2 = json.load(
                open(
                    P.join(
                        self.data_root, "landmarks", P2_name.split("/")[1] + ".json"
                    ),
                    "r",
                )
            )

            BP1, _ = create_landmark_and_mask(landmarks1, lfw=self.lfw)
            BP2, mask = create_landmark_and_mask(landmarks2, lfw=self.lfw)

            if self.reduced_landmarks:
                BP1 = BP1[legal_mask]
                BP2 = BP2[legal_mask]

            BP1, BP2, mask = self.resize(BP1), self.resize(BP2), self.resize(mask)

            return {
                "P1": P1,
                "BP1": BP1,
                "P2": P2,
                "BP2": BP2,
                "masked_P2": P2 * (1 - mask),
                "P1_path": P1_name,
                "P2_path": P2_basename,
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
    def __init__(
        self,
        data_dir: str = "./",
        batch_size=32,
        same_percentage=0.5,
        same_identity=False,
        reduced_landmarks=False,
        iciap=False,
        lfw=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(220, 180) if not lfw else (256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.bs = batch_size
        self.same_percentage = same_percentage
        self.same_identity = same_identity
        self.reduced_landmarks = reduced_landmarks
        self.iciap = iciap
        self.lfw = lfw

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
            self.data_dir,
            self.transform,
            "fit",
            same_percentage=self.same_percentage,
            same_identity=self.same_identity,
            reduced_landmarks=self.reduced_landmarks,
        )
        return DataLoader(
            data_train,
            batch_size=self.bs,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        data_val = CustomLoader(
            self.data_dir,
            self.transform,
            "val",
            reduced_landmarks=self.reduced_landmarks,
        )
        return DataLoader(
            data_val, batch_size=1, num_workers=4, persistent_workers=True
        )

    def test_dataloader(self):
        data_test = CustomLoader(
            self.data_dir,
            self.transform,
            "test",
            reduced_landmarks=self.reduced_landmarks,
            iciap=self.iciap,
            lfw=self.lfw,
        )
        return DataLoader(
            data_test, batch_size=1, num_workers=20, persistent_workers=True
        )
