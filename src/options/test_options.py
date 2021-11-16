import argparse


class TestOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument(
            "--data_root", type=str, default="/media/dvl1/SSD_DATA/anonygan-dataset"
        )
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--beta1", type=float, default=0.5)
        self.parser.add_argument("--name", type=str)
        self.parser.add_argument("--lr", type=float, default=2e-4)
        self.parser.add_argument("--ch_input", type=int, default=6)
        self.parser.add_argument("--no_ch_att", action="store_true")
        self.parser.add_argument("--iciap", action="store_true")
        self.parser.add_argument("--ckpt", type=str, default="")
        self.parser.add_argument("--reduced_landmarks", action="store_true")
        self.parser.add_argument("--lfw", action="store_true")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
