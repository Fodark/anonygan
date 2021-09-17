import torch
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

from backbones import get_model


def get_feats(net, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    feat = net(img)

    return feat


@torch.no_grad()
def main(weight, name, orig_dir, mod_dir):

    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

    # orig_dir = "/media/dvl1/SSD_DATA/wildtrack-dataset/quantitative_test/orig/"
    # mod_dir = "/media/dvl1/SSD_DATA/wildtrack-dataset/quantitative_test/gen/"
    print("Starting ID retrieval...")
    total_loss = []
    d = torch.nn.CosineSimilarity()

    for f in tqdm(os.listdir(orig_dir)):
        orig_pic = os.path.join(orig_dir, f)
        mod_pic = os.path.join(mod_dir, f)

        orig_feats, mod_feats = get_feats(net, orig_pic), get_feats(net, mod_pic)

        diff = d(orig_feats, mod_feats).item()

        total_loss.append(diff)

    std, mean = torch.std_mean(torch.tensor(total_loss))

    print("Done. Results:")
    print(f"{mean.item()} +- {std.item()}")
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ArcFace Training")
    parser.add_argument("--network", type=str, default="r50", help="backbone network")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("-s", "--original", type=str, default="")
    parser.add_argument("-g", "--generated", type=str, default="")
    args = parser.parse_args()
    main(args.weight, args.network, args.original, args.generated)
