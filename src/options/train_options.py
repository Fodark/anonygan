import argparse


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
        self.parser.add_argument("--id_discriminator", action="store_true")
        self.parser.add_argument("--pretrained_id_discriminator", action="store_true")
        self.parser.add_argument("--no_wfm", action="store_true")
        self.parser.add_argument("--no_l1", action="store_true")
        self.parser.add_argument("--name", type=str)
        self.parser.add_argument("--g_lr", type=float, default=2e-4)
        self.parser.add_argument("--lr", type=float, default=2e-4)
        self.parser.add_argument("--ch_input", type=int, default=6)
        self.parser.add_argument("--train_same_identity", action="store_true")
        self.parser.add_argument("--no_ch_att", action="store_true")
        self.parser.add_argument("--yiming", action="store_true")
        self.parser.add_argument("--reduced_landmarks", action="store_true")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
