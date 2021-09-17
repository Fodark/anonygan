import torch
import torch.nn as nn
import numpy as np
from skimage.draw import circle, line_aa, disk
from collections import OrderedDict


def _gradient_penalty(netD, img_att, img_fake):
    # interpolate sample
    bs = img_fake.shape[0]
    alpha = torch.rand(bs, 1, 1, 1).expand_as(img_fake)
    interpolated = torch.autograd.Variable(
        alpha * img_att + (1 - alpha) * img_fake, requires_grad=True
    )
    pred_interpolated = netD(interpolated)
    pred_interpolated = pred_interpolated[-1]

    # compute gradients
    grad = torch.autograd.grad(
        outputs=pred_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(pred_interpolated.size()),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]

    # penalize gradients
    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    loss_d_gp = torch.mean((grad_l2norm - 1) ** 2)

    return loss_d_gp


class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        @notice: avoid in-place ops.
        https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(SpecificNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cuda()
        self.mean = self.mean.view([1, 3, 1, 1])

        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cuda()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])

        x = (x - mean) / std

        return x


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def get_current_visuals(
    input_P1, input_P2, masked_P2, input_BP1, input_BP2, fake_p2, generated_p2
):
    height, width = input_P1.size(2), input_P1.size(3)
    input_P1 = tensor2im(input_P1.data)
    input_P2 = tensor2im(input_P2.data)

    input_BP1 = draw_pose_from_map(input_BP1.data)[0]
    input_BP2 = draw_pose_from_map(input_BP2.data)[0]

    # generated_p2 = tensor2im(self.generated_p2.data)
    masked_P2 = tensor2im(masked_P2.data)
    fake_p2 = tensor2im(fake_p2.data)
    generated_p2 = tensor2im(generated_p2.data)

    vis = np.zeros((height, width * 7, 3)).astype(np.uint8)  # h, w, c
    vis[:, :width, :] = input_P1
    vis[:, width : width * 2, :] = input_BP1
    vis[:, width * 2 : width * 3, :] = input_P2
    vis[:, width * 3 : width * 4, :] = masked_P2
    vis[:, width * 4 : width * 5, :] = input_BP2
    vis[:, width * 5 : width * 6, :] = fake_p2
    vis[:, width * 6 : width * 7, :] = generated_p2
    # vis[:, width * 5 :, :] = fake_p2

    # ret_visuals = OrderedDict([("vis", vis)])

    return vis


# draw pose img
LIMB_SEQ = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [17, 18],
    [18, 19],
    [19, 20],
    [20, 21],
    [22, 23],
    [23, 24],
    [24, 25],
    [25, 26],
    [27, 28],
    [28, 29],
    [29, 30],
    [31, 32],
    [32, 33],
    [33, 34],
    [34, 35],
    [36, 37],
    [37, 38],
    [38, 39],
    [39, 40],
    [40, 41],
    [36, 41],
    [42, 43],
    [43, 44],
    [44, 45],
    [45, 46],
    [46, 47],
    [42, 47],
    [48, 49],
    [49, 50],
    [50, 51],
    [51, 52],
    [53, 54],
    [54, 55],
    [55, 56],
    [56, 57],
    [57, 58],
    [58, 59],
    [48, 59],
    [60, 61],
    [61, 62],
    [62, 63],
    [63, 64],
    [64, 65],
    [65, 66],
    [66, 67],
    [60, 67],
]

COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]

BP = 68
MISSING_VALUE = -1


def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(BP)]
    pose_map = pose_map[..., :BP]

    y, x, z = np.where(
        np.logical_and(pose_map == pose_map.max(axis=(0, 1)), pose_map > threshold)
    )
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(BP):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate(
        [np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1
    )


def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    # CHW -> HCW -> HWC
    pose_map = pose_map[0].cpu().transpose(1, 0).transpose(2, 1).numpy()

    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


# draw pose from map
def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3,), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = (
                pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            )
            to_missing = (
                pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            )
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(
                pose_joints[f][0],
                pose_joints[f][1],
                pose_joints[t][0],
                pose_joints[t][1],
            )
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        # yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        yy, xx = disk((joint[0], joint[1]), radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i % len(COLORS)]
        mask[yy, xx] = True

    return colors, mask
