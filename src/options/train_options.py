import argparse
import os
import torch


class TrainOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument(
            "--data_root", type=str, default="/media/dvl1/SSD_DATA/anonygan-dataset"
        )
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--arcface_arch", type=str, default="r50")
        self.parser.add_argument("--lambda_gan", type=int, default=5)
        self.parser.add_argument("--lambda_pose", type=int, default=5)
        self.parser.add_argument("--lambda_id", type=int, default=10)
        self.parser.add_argument("--lambda_wfm", type=int, default=5)
        self.parser.add_argument("--lambda_rec", type=int, default=10)
        self.parser.add_argument("--beta1", type=float, default=0.5)
        self.parser.add_argument("--same_percentage", type=float, default=0.5)
        self.parser.add_argument("--double_discriminator", action="store_true")
        self.parser.add_argument("--pretrained_id_discriminator", action="store_true")
        self.parser.add_argument("--no_wfm", action="store_true")
        self.parser.add_argument("--no_l1", action="store_true")
        self.parser.add_argument("--name", type=str)
        self.parser.add_argument("--lr", type=float, default=2e-4)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
