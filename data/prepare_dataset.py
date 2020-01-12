import glob
import os

import numpy as np


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

img_paths = glob.glob(os.path.join('cartoonset10k', '*.png'))
img_lbl_files = [img_path.replace('.png', '.csv') for img_path in img_paths]

img_lbls = []
for img_path, img_lbl_file in zip(img_paths, img_lbl_files):
    img_lbl = np.genfromtxt(img_lbl_file, dtype=int, usecols=1, delimiter=',')

    eye_color = img_lbl[10] if img_lbl[10] <= 4 else -1  # use five types
    face_color = img_lbl[11]if img_lbl[11] <= 4 else -1  # use five types
    hair_color = img_lbl[12]if img_lbl[12] <= 4 else -1  # use five types
    glasses = img_lbl[13]if img_lbl[13] <= 4 else -1  # use five types

    one_hot_mapping = {0: [1, 0, 0, 0, 0],
                       1: [0, 1, 0, 0, 0],
                       2: [0, 0, 1, 0, 0],
                       3: [0, 0, 0, 1, 0],
                       4: [0, 0, 0, 0, 1],
                       -1: [0, 0, 0, 0, 0]}
    eye_color = one_hot_mapping[eye_color]
    face_color = one_hot_mapping[face_color]
    hair_color = one_hot_mapping[hair_color]
    glasses = one_hot_mapping[glasses]

    img_lbl = np.concatenate((eye_color, face_color, hair_color, glasses), axis=0)
    img_lbls.append(img_lbl)


def save_lbl(img_paths, img_lbls, save_path):
    with open(save_path, 'w') as f:
        for img_path, img_lbl in zip(img_paths, img_lbls):
            f.write(os.path.split(img_path)[-1])
            for lbl in img_lbl:
                f.write(' %d' % (lbl * 2 - 1))
            f.write('\n')

save_lbl(img_paths[:8000], img_lbls[:8000], 'train_label.txt')
save_lbl(img_paths[8000:9000], img_lbls[8000:9000], 'val_label.txt')
save_lbl(img_paths[9000:], img_lbls[9000:], 'test_label.txt')
