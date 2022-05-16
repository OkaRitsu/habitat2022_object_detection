import cv2
import os
from PIL import Image
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from module import setup_cfg, Detic, OBJECT_GOAL_CATEGORIES


# detic setting
mp.set_start_method("spawn", force=True)
cfg = setup_cfg()
detic = Detic(cfg)

# input image files
input_dir = 'habitat_img'
img_files = [file for file in os.listdir(input_dir) if file.endswith('.png')]

# make output dirs
output_dir = 'output'
for goal in OBJECT_GOAL_CATEGORIES:
    os.makedirs(os.path.join(output_dir, goal), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'nothing'), exist_ok=True)

# create outputs
for img_file in tqdm(img_files):
    image = Image.open(os.path.join(input_dir, img_file))
    image = np.asarray(image)

    has_goals = False
    for idx, goal in enumerate(OBJECT_GOAL_CATEGORIES):
        is_detected, center = detic(image, idx)
        if is_detected:
            # import pdb; pdb.set_trace()
            has_goals = True
            img = cv2.imread(os.path.join(input_dir, img_file))
            cv2.drawMarker(
                img, center.tolist(), (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=30, thickness=5)
            cv2.imwrite(os.path.join(output_dir, goal, img_file), img)
    
    if not has_goals:
        img = cv2.imread(os.path.join(input_dir, img_file))
        cv2.imwrite(os.path.join(output_dir, 'nothing', img_file), img)
