from xml.dom.minidom import parse
from glob import glob
import os
import numpy as np

def read_xml(file):

    bboxes_list = []

    dom_tree = parse(file)
    root = dom_tree.documentElement
    path = root.getElementsByTagName("filename")[0].firstChild.data
    object = root.getElementsByTagName("object")
    for obj in object:
        label = obj.getElementsByTagName("name")[0].childNodes[0].data.lower()
        label = "1" if "mask" in label else "0"
        bbox = obj.getElementsByTagName("bndbox")[0]
        xmin = bbox.getElementsByTagName("xmin")[0].firstChild.data
        ymin = bbox.getElementsByTagName("ymin")[0].firstChild.data
        xmax = bbox.getElementsByTagName("xmax")[0].firstChild.data
        ymax = bbox.getElementsByTagName("ymax")[0].firstChild.data

        difficulty = obj.getElementsByTagName("difficult")[0].childNodes[0].data
        bboxes_list += [xmin, ymin, xmax, ymax, difficulty, label]
    return path, bboxes_list


def process_mask_data(src, dst,
                      dst_anno_name="trainval.txt", train="train.txt", val="val.txt"):
    files = glob(os.path.join(src, "*.xml"))

    difficulty_set = set()

    if not os.path.exists(dst):
        os.makedirs(dst)
    trainval = os.path.join(dst, dst_anno_name)
    f = open(trainval, "w")
    data = []
    for file in files:
        print("processing [{}]...".format(file))
        path, boxes = read_xml(file)
        if len(boxes) > 0 and boxes[-1] not in difficulty_set:
            difficulty_set.add(boxes[-1])
            print(path)
        anno = [path] + boxes
        line = " ".join(anno) + "\n"
        f.write(line)
        data.append(line)
    np.random.shuffle(data)
    tlen = int(len(data) * 0.8)
    with open(os.path.join(dst, train), "w") as f:
        for line in data[:tlen]:
            f.write(line)

    with open(os.path.join(dst, val), "w") as f:
        for line in data[tlen:]:
            f.write(line)

    f.close()


if __name__ == '__main__':
    process_mask_data(r"D:\AAA\Learn\Dataset\train", "../anno")