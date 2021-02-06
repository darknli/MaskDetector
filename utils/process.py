import numpy as np


def split_dataset(anno_list, shuffle_save_path=None, ratio=0.9, load_shuffle=None):
    anno_list = np.array(anno_list)
    if load_shuffle is None:
        indices = np.arange(0, len(anno_list))
        np.random.shuffle(indices)
    else:
        print("读取{}的打乱索引".format(load_shuffle))
        indices = np.load(load_shuffle)
    image_list = anno_list[indices]

    train_end_idx = int(len(image_list) * ratio)
    train = image_list[:train_end_idx]
    val = image_list[train_end_idx:]

    if shuffle_save_path is not None:
        np.save(shuffle_save_path, indices)
    return train, val
