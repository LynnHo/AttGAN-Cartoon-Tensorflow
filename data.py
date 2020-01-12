import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl


ATT_ID = {'eye_color_0': 0, 'eye_color_1': 1, 'eye_color_2': 2, 'eye_color_3': 3, 'eye_color_4': 4,
          'face_color_0': 5, 'face_color_1': 6, 'face_color_2': 7, 'face_color_3': 8, 'face_color_4': 9,
          'hair_color_0': 10, 'hair_color_1': 11, 'hair_color_2': 12, 'hair_color_3': 13, 'hair_color_4': 14,
          'glasses_0': 15, 'glasses_1': 16, 'glasses_2': 17, 'glasses_3': 18, 'glasses_4': 19}
ID_ATT = {v: k for k, v in ATT_ID.items()}


def make_celeba_dataset(img_dir,
                        label_path,
                        att_names,
                        batch_size,
                        load_size=286,
                        crop_size=256,
                        training=True,
                        drop_remainder=True,
                        shuffle=True,
                        repeat=1):
    img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
    img_paths = np.array([py.join(img_dir, img_name) for img_name in img_names])
    labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, len(ATT_ID.keys()) + 1))
    labels = labels[:, np.array([ATT_ID[att_name] for att_name in att_names])]

    if shuffle:
        idx = np.random.permutation(len(img_paths))
        img_paths = img_paths[idx]
        labels = labels[idx]

    if training:
        def map_fn_(img, label):
            img = tf.image.resize(img, [crop_size, crop_size])
            # img = tl.random_rotate(img, 5)
            img = tf.image.random_flip_left_right(img)
            # img = tf.image.random_crop(img, [crop_size, crop_size, 3])
            # img = tl.color_jitter(img, 25, 0.2, 0.2, 0.1)
            # img = tl.random_grayscale(img, p=0.3)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label
    else:
        def map_fn_(img, label):
            img = tf.image.resize(img, [crop_size, crop_size])
            # img = tl.center_crop(img, size=crop_size)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          labels=labels,
                                          drop_remainder=drop_remainder,
                                          map_fn=map_fn_,
                                          shuffle=shuffle,
                                          repeat=repeat)

    if drop_remainder:
        len_dataset = len(img_paths) // batch_size
    else:
        len_dataset = int(np.ceil(len(img_paths) / batch_size))

    return dataset, len_dataset


def check_attribute_conflict(att_batch, att_name, att_names):
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    idx = att_names.index(att_name)

    for att in att_batch:
        if att_name in ['eye_color_0', 'eye_color_1', 'eye_color_2', 'eye_color_3', 'eye_color_4'] and att[idx] == 1:
            for n in ['eye_color_0', 'eye_color_1', 'eye_color_2', 'eye_color_3', 'eye_color_4']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['face_color_0', 'face_color_1', 'face_color_2', 'face_color_3', 'face_color_4'] and att[idx] == 1:
            for n in ['face_color_0', 'face_color_1', 'face_color_2', 'face_color_3', 'face_color_4']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['hair_color_0', 'hair_color_1', 'hair_color_2', 'hair_color_3', 'hair_color_4'] and att[idx] == 1:
            for n in ['hair_color_0', 'hair_color_1', 'hair_color_2', 'hair_color_3', 'hair_color_4']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['glasses_0', 'glasses_1', 'glasses_2', 'glasses_3', 'glasses_4'] and att[idx] == 1:
            for n in ['glasses_0', 'glasses_1', 'glasses_2', 'glasses_3', 'glasses_4']:
                if n != att_name:
                    _set(att, 0, n)

    return att_batch
