import cv2
from random import random, randint
import numpy as np


def random_crop_square(image, bboxes):
    oh, ow = image.shape[:2]
    if len(bboxes) > 0:
        minx = np.min(bboxes[:, 0])
        miny = np.min(bboxes[:, 1])
    else:
        minx = ow
        miny = oh

    if oh > ow:
        dw = 0
        span = oh - ow
        span = min(span, miny)
        dh = randint(0, span)
        min_size = ow
    else:
        span = ow - oh
        span = min(span, minx)
        dw = randint(0, span)
        dh = 0
        min_size = oh
    return image[dh: dh + min_size, dw: dw + min_size], (dh, dw, min_size)


def center_is_in_img(bboxes, size):
    center = (bboxes[:, 2:] + bboxes[:, :2]) / 2
    mask = (center[:, 0] < 0) + (center[:, 1] < 0) + (center[:, 0] > size) + (center[:, 1] > size)
    return mask


def abandon_min_face(bboxes, min_area):
    mask = np.prod(bboxes[:, 2:] - bboxes[:, :2], -1) > min_area
    return mask


class CropResize:
    def __init__(self, size, crop_prob=0.5):
        self.size = size
        self.crop_prob = crop_prob

    def __call__(self, data):
        image = data["image"]
        bboxes, conf = data["bboxes"][:, :4], data["bboxes"][:, 4:]
        np.random.uniform()
        oh, ow = image.shape[:2]
        hw_ratio = oh / ow
        # cv2.imshow("o", image)
        if 0.8 < hw_ratio < 1.25 and random() > self.crop_prob:
            hr, wr = self.size / oh, self.size / ow
            bboxes[:, ::2] = bboxes[:, ::2] * wr
            bboxes[:, 1::2] = bboxes[:, 1::2] * hr
            image = cv2.resize(image, (self.size, self.size))
        else:
            image, (dh, dw, length) = random_crop_square(image, bboxes)
            bboxes[:, ::2] = bboxes[:, ::2] - dw
            bboxes[:, 1::2] = bboxes[:, 1::2] - dh
            mask = (bboxes[:, 0] > length) + (bboxes[:, 2] < 0) + \
                   (bboxes[:, 1] > length) + (bboxes[:, 3] < 0)
            # mask = center_is_in_img(bboxes, length)
            mask = ~mask
            bboxes = bboxes[mask]
            bboxes = np.clip(bboxes, 0, length)
            conf = conf[mask]
            image = cv2.resize(image, (self.size, self.size))
            ratio = self.size / length
            bboxes *= ratio
        ground_truth = np.concatenate([bboxes, conf], -1)
        mask = abandon_min_face(bboxes, 300)

        # for box in bboxes[mask].astype(int):
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        # cv2.imshow("r", image)
        # print(bboxes, mask)
        # cv2.waitKey()

        data["image"] = image
        data["bboxes"] = ground_truth[mask]
        return data


class Normalization:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, data):
        image = data["image"]
        image = image.astype("float32")
        image[:, :] -= self.mean
        image[:, :] /= self.var
        image = image.transpose((2, 0, 1))
        data["image"] = image
        return data
