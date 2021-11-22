import os
import torch
import numpy as np
import cv2
from PIL import Image

kp_dir = "/media/dvl1/SSD_DATA/bigraph-dataset-bis/trainK_68"
mask_dir = "/media/dvl1/SSD_DATA/bigraph-dataset-bis/train_mask"

kp_list = os.listdir(kp_dir)

for kp in kp_list:
    # extract basename
    name = kp.split(".")[0]
    kps = torch.load(os.path.join(kp_dir,kp))
    # to work with numpy -> channel last
    kps_npy = kps.numpy().transpose(1,2,0)

    # black base image
    img_msk = np.zeros((218, 178, 3), np.uint8)
    contours = []
    # leftest jaw point, highest left eyebrow point
    first_y, first_x = np.where(kps_npy[:,:,0]==1)[1][0], np.where(kps_npy[:,:,19]==1)[0][0]
    # righest jaw point, highest right eyebrow point
    last_y, last_x = np.where(kps_npy[:,:,16]==1)[1][0], np.where(kps_npy[:,:,24]==1)[0][0]

    contours.append((first_y, first_x))
    # complete jawline
    for p in range(17):
        x, y = np.where(kps_npy[:,:,p]==1)[0][0], np.where(kps_npy[:,:,p]==1)[1][0]
        contours.append((y, x))
    
    contours.append((last_y, last_x))
    # fill the polygon defined by above points of white
    cv2.fillPoly(img_msk, pts=[np.array(contours)], color=(255,255,255))
    result = Image.fromarray(img_msk)
    result.save(os.path.join(mask_dir, name+".jpg"))
    print(f'Done {kp}')