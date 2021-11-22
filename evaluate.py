from cleanfid import fid
import dlib

_detector = dlib.get_frontal_face_detector()

import os
import cv2
from natsort import natsorted
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
import face_alignment
from skimage.transform import resize
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, file_list):
        self.img_dir = img_dir
        self.file_list = natsorted(file_list)
        self.total_size = len(file_list)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_list[idx] + ".jpg")
        if not os.path.isfile(img_path):
            img_path = img_path[:-4] + ".png"
        img = Image.open(img_path)
        image = transforms.ToTensor()(img)
        return {"images": image}


def lfw_matches(model, in1, in2):
    logits1 = []
    logits2 = []
    # divide the two inputs in smaller chunks
    for minib1, minib2 in zip(in1.chunk(6), in2.chunk(6)):
        logits1.append(model(minib1.cuda()).detach())
        logits2.append(model(minib2.cuda()).detach())
    logits1 = torch.cat(logits1)
    logits2 = torch.cat(logits2)

    # compute the idx corresponding to the highest logit
    indices1, indices2 = torch.max(logits1, dim=1)[1], torch.max(logits2, dim=1)[1]
    # return the number of matches between the two set of ids
    return torch.sum(indices1 == indices2).item()


def compute_reid_lfw(lfw):
    lfw_root = "/media/dvl1/HDD_DATA/iciap/lfw"
    generated_root = lfw
    # print("Computing reidentification on lfw dataset, anonymizing second image")
    # load annotations file for test lfw
    annotations = "/media/dvl1/HDD_DATA/iciap/lfw_annotations/pairs.txt"
    content = open(annotations, "r").readlines()
    # the first line contains number of splits and elements per split
    splits, elems_per_split = content[0].split("\t")
    splits, elems_per_split = int(splits), int(elems_per_split)

    reids_vgg = []
    reids_casia = []

    idx = 1
    # 10 splits of lfw test set
    for split_idx in range(splits):
        # print("\tsplit", split_idx)
        in1_matched = []
        in2_matched = []
        # 300 matched pairs
        for _ in range(elems_per_split):
            # for matched pairs the format is name, id1, id2
            name, id1, id2 = content[idx].split("\t")
            id1, id2 = int(id1), int(id2)
            filename1 = os.path.join(lfw_root, name, f"{name}_{id1:04d}.jpg")
            filename2 = os.path.join(generated_root, f"{name}_{id2:04d}.jpg")

            assert os.path.isfile(filename1), f"Filename 1 not found {filename1}"
            # assert os.path.isfile(filename2), f"Filename 2 not found {filename2}"

            # from [0,255] to [0,1]
            if os.path.isfile(filename2):
                in1_matched.append(read_image(filename1).float() / 255.0)
                in2_matched.append(read_image(filename2).float() / 255.0)

            idx += 1

        in1_matched = torch.stack(in1_matched)
        in2_matched = torch.stack(in2_matched)

        # in_2 should be forwarded in the generator...
        # in_2 = G(in_2, ...)

        in1_unmatched = []
        in2_unmatched = []
        # 300 unmatched pairs
        for _ in range(elems_per_split):
            name1, id1, name2, id2 = content[idx].split("\t")
            id1, id2 = int(id1), int(id2)
            filename1 = os.path.join(lfw_root, name1, f"{name1}_{id1:04d}.jpg")
            filename2 = os.path.join(generated_root, f"{name2}_{id2:04d}.jpg")

            # from [0,255] to [0,1]
            if os.path.isfile(filename2):
                in1_unmatched.append(read_image(filename1).float() / 255.0)
                in2_unmatched.append(read_image(filename2).float() / 255.0)

            idx += 1

        in1_unmatched = torch.stack(in1_unmatched)
        in2_unmatched = torch.stack(in2_unmatched)

        # in_2 should be forwarded in the generator...
        # in_2 = G(in_2, ...)

        # VGG
        model = InceptionResnetV1(pretrained="vggface2", classify=True).cuda().eval()
        reidentified_matched = lfw_matches(model, in1_matched, in2_matched)
        # print(f"\t\tvgg on matched pairs: {reidentified_matched} out of 300")
        reidentified_unmatched = lfw_matches(model, in1_unmatched, in2_unmatched)
        # print(f"\t\tvgg on unmatched pairs: {reidentified_unmatched} out of 300")

        reids_vgg.append((reidentified_matched + reidentified_unmatched) / 600)

        # CASIA
        model = (
            InceptionResnetV1(pretrained="casia-webface", classify=True).cuda().eval()
        )
        reidentified_matched = lfw_matches(model, in1_matched, in2_matched)
        # print(f"\t\tcasia on matched pairs: {reidentified_matched} out of 300")
        reidentified_unmatched = lfw_matches(model, in1_unmatched, in2_unmatched)
        # print(f"\t\tcasia on unmatched pairs: {reidentified_unmatched} out of 300")

        reids_casia.append((reidentified_matched + reidentified_unmatched) / 600)

    vgg_std, vgg_mean = torch.std_mean(torch.Tensor(reids_vgg))
    casia_std, casia_mean = torch.std_mean(torch.Tensor(reids_casia))

    vgg_mean, vgg_std = round(vgg_mean.item(), 4), round(vgg_std.item(), 4)
    casia_mean, casia_std = round(casia_mean.item(), 4), round(casia_std.item(), 4)

    # print(
    #     "VGG mean", vgg_mean, "std", vgg_std, "CASIA mean", casia_mean, "std", casia_std
    # )

    return (vgg_mean, vgg_std), (casia_mean, casia_std)


def compute_reid(original, generated):
    size1 = (256, 256)
    size2 = (256, 256)
    bs = min(64, len(os.listdir(original)))

    file_list1 = list(os.listdir(original))
    file_list1 = set([_[:-4] for _ in file_list1])
    file_list2 = list(os.listdir(generated))
    file_list2 = set([_[:-4] for _ in file_list2])

    file_list = list(file_list1.intersection(file_list2))

    dl1 = DataLoader(CustomImageDataset(original, file_list), batch_size=bs)
    dl2 = DataLoader(CustomImageDataset(generated, file_list), batch_size=bs)

    # print("Computing reidentification using pretrained FaceNet on VGG")

    model = InceptionResnetV1(pretrained="vggface2", classify=True).cuda().eval()

    total_images = len(os.listdir(generated))
    reidentified = 0

    for b1, b2 in zip(dl1, dl2):
        b1, b2 = b1["images"].cuda(), b2["images"].cuda()
        b1 = F.interpolate(b1, size1)
        b2 = F.interpolate(b2, size2)
        # b = torch.cat([b1, b2], dim=0)
        logits1, logits2 = model(b1).detach(), model(b2).detach()
        indices1, indices2 = torch.max(logits1, dim=1)[1], torch.max(logits2, dim=1)[1]
        reidentified += torch.sum(indices1 == indices2)

    p_vggface = round((reidentified.item() / total_images) * 100, ndigits=2)
    # print(
    #    f"vggface reidentified {reidentified} out of {total_images}, {p_vggface}% of the total"
    # )

    # print("Computing reidentification using pretrained FaceNet on Casia")

    model = InceptionResnetV1(pretrained="casia-webface", classify=True).cuda().eval()
    reidentified = 0

    for b1, b2 in zip(dl1, dl2):
        b1, b2 = b1["images"].cuda(), b2["images"].cuda()
        b1 = F.interpolate(b1, size1)
        b2 = F.interpolate(b2, size2)
        # b = torch.cat([b1, b2], dim=0)
        logits1, logits2 = model(b1).detach(), model(b2).detach()
        indices1, indices2 = torch.max(logits1, dim=1)[1], torch.max(logits2, dim=1)[1]
        reidentified += torch.sum(indices1 == indices2)

    p_casia = round((reidentified.item() / total_images) * 100, ndigits=2)
    # print(
    #    f"casia reidentified {reidentified} out of {total_images}, {p_casia}% of the total"
    # )

    return p_vggface, p_casia


def face_detection(root):
    # print(f"Computing face detection in folder {root}")

    mtcnn = MTCNN(select_largest=False, device="cuda")

    image_list = os.listdir(root)
    total_images = len(image_list)
    hits_dlib = 0
    hits_facenet = 0

    # print(f"{total_images} images in directory.")
    # print("Starting")

    for filename in image_list:
        image = cv2.imread(os.path.join(root, filename))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(rgb)
        rects = _detector(rgb, 1)
        _, probs = mtcnn.detect(frame)
        # print(probs)

        if probs[0] is not None:
            hits_facenet += 1

        if len(rects) != 0:
            hits_dlib += 1

    p_dlib = round((hits_dlib / total_images) * 100, ndigits=2)
    p_facenet = round((hits_facenet / total_images) * 100, ndigits=2)

    # print(f"Finished scanning {root}")
    # print(f"dlib detected {hits_dlib} faces, {p_dlib}% of the total")
    # print(f"FaceNet detected {hits_facenet} faces, {p_facenet}% of the total")

    return p_dlib, p_facenet


def compute_fid(orig, gen):
    score = fid.compute_fid(orig, gen)
    return score


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/home/dvl1/projects/ciagan/source/" + "shape_predictor_68_face_landmarks.dat"
)


def compute_kp(path):
    assert os.path.isfile(path), f"{path} is not a file"
    clr = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    clr = cv2.resize(clr, (180, 220))
    height, width, _ = clr.shape
    img = clr.copy()
    img_dlib = np.array(clr[:, :, :], dtype=np.uint8)
    dets = detector(img_dlib, 1)

    for k_it, d in enumerate(dets):
        if k_it != 0:
            continue
        kp = []
        landmarks = predictor(img_dlib, d)

        # f_x, f_y, f_w, f_h = rect_to_bb(d)

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            # x, y = round((x) / width, 4), round((y) / height, 4)
            kp.append([x, y])
        return np.array(kp)


def compute_norm_distance(gt, pred):
    eye_dis = np.linalg.norm(gt[36] - gt[45])
    d = np.linalg.norm(pred - gt, axis=1).mean() / eye_dis
    return d


def compute_pose(gt_root, pred_root):
    # gt_list = natsorted(os.listdir(gt_root))

    file_list1 = list(os.listdir(gt_root))
    file_list1 = set([_[:-4] for _ in file_list1])
    file_list2 = list(os.listdir(pred_root))
    file_list2 = set([_[:-4] for _ in file_list2])

    file_list = list(file_list1.intersection(file_list2))

    pred_list = natsorted(file_list)

    dist = []

    for f in pred_list:
        gt_path = os.path.join(gt_root, f + ".jpg")
        if not os.path.isfile(gt_path):
            gt_path = gt_path[:-4] + ".png"
        pred_path = os.path.join(pred_root, f + ".jpg")
        if not os.path.isfile(pred_path):
            pred_path = pred_path[:-4] + ".png"

        assert os.path.isfile(gt_path), f"{gt_path} is not a file"
        assert os.path.isfile(pred_path), f"{pred_path} is not a file"

        kp_gt = compute_kp(gt_path)
        kp_pred = compute_kp(pred_path)

        if kp_pred is None or kp_gt is None:
            # print(f"Skipping {pred}")
            continue

        dist.append(compute_norm_distance(kp_gt, kp_pred))

    return np.array(dist).mean()


def compute_pose_bis(orig, gen):
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False
    )

    file_list1 = list(os.listdir(orig))
    file_list1 = set([_[:-4] for _ in file_list1])
    file_list2 = list(os.listdir(gen))
    file_list2 = set([_[:-4] for _ in file_list2])

    file_list = list(file_list1.intersection(file_list2))

    pred_list = natsorted(file_list)

    dist = []

    for f in pred_list:
        gt_path = os.path.join(orig, f + ".jpg")
        if not os.path.isfile(gt_path):
            gt_path = gt_path[:-4] + ".png"
        pred_path = os.path.join(gen, f + ".jpg")
        if not os.path.isfile(pred_path):
            pred_path = pred_path[:-4] + ".png"

        assert os.path.isfile(gt_path), f"{gt_path} is not a file"
        assert os.path.isfile(pred_path), f"{pred_path} is not a file"

        # kp_gt = compute_kp(gt_path)
        # kp_pred = compute_kp(pred_path)

        input1 = io.imread(gt_path)
        # input1 = resize(input1, (256, 256))
        preds1 = fa.get_landmarks(input1)

        input2 = io.imread(pred_path)
        # input2 = resize(input2, (256, 256))
        preds2 = fa.get_landmarks(input2)

        if preds1 is None or preds2 is None:
            # print(f"Skipping {pred}")
            continue

        lndmks1 = []

        for (x, y) in preds1[0]:
            tmp = [x, y]
            lndmks1.append(tmp)

        lndmks2 = []

        for (x, y) in preds2[0]:
            tmp = [x, y]
            lndmks2.append(tmp)

        l1 = torch.nn.L1Loss()

        d = l1(torch.tensor(lndmks1), torch.tensor(lndmks2)) / l1(
            torch.tensor(lndmks1[36]), torch.tensor(lndmks1[45])
        )

        dist.append(d)

    return torch.std_mean(torch.tensor(dist))


def compute_metrics(orig, gen):
    # (vgg_mean, vgg_std), (casia_mean, casia_std) = compute_reid_lfw(gen)
    pose = compute_pose_bis(orig, gen)
    fid = compute_fid(orig, gen)
    # det_dblib, det_facenet = face_detection(gen)

    print(f"FID: {fid}")
    # print(f"Det dlib: {det_dblib} - Det FaceNet {det_facenet}")
    # print(f"ReID VGG: {vgg_mean} +- {vgg_std} - ReID CASIA {casia_mean} +- {casia_std}")
    print(f"Pose: {pose}")
