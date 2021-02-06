import numpy as np


class Anchor:
    def __init__(self, image_size, num_classes):
        self.stride = [8, 16, 32]
        self.num_classes = num_classes
        self.image_size = image_size
        self.anchor_size = np.array([
            [12, 16], [19, 36], [40, 28],
            [36, 75], [76, 55], [72, 146],
            [142, 110], [192, 243], [459, 401]
        ], dtype=np.float32)
        self.branch_size = np.array([image_size // s for s in self.stride], dtype=int)
        self.anchor_mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.anchor_area = self.anchor_size[:, 0] * self.anchor_size[:, 1]

        self.channel = 4 + 1 + num_classes

    def __call__(self, bboxes):
        if len(bboxes) == 0:
            return [np.zeros((3, self.branch_size[i], self.branch_size[i], self.channel), dtype=np.float32) for i in range(3)]
        center_box = np.concatenate(
            [(bboxes[:, :2] + bboxes[:, 2:4])/2, bboxes[:, 2:4] - bboxes[:, :2], bboxes[:, 4:] + 5], -1)
        best_anchor_idx = self._match_best_anchor(center_box[:, 2:4])
        best_anchor = self.anchor_size[best_anchor_idx]

        all_targets = []

        for i in range(3):
            num_anchor = len(self.anchor_mask[i])
            mask = np.logical_and(i * num_anchor <= best_anchor_idx, best_anchor_idx < (i + 1) * num_anchor)
            branch_target = np.zeros(
                (num_anchor, self.branch_size[i], self.branch_size[i], self.channel), dtype=np.float32
            )

            branch_boxes = center_box[mask]
            branch_anchors = best_anchor[mask]
            try:
                xy = branch_boxes[:, :2] / self.stride[i]
                wh = branch_boxes[:, 2:4] / branch_anchors

                anchor_idx = best_anchor_idx[mask] % 3
                for idx, (x, y), (w, h), cid in zip(anchor_idx, xy, wh, branch_boxes[:, -1]):
                    integer_x, integer_y = int(x), int(y)
                    branch_target[idx, integer_x, integer_y, 0] = x - integer_x
                    branch_target[idx, integer_x, integer_y, 1] = y - int(integer_y)
                    branch_target[idx, integer_x, integer_y, 2] = np.log(w)
                    branch_target[idx, integer_x, integer_y, 3] = np.log(h)
                    branch_target[idx, integer_x, integer_y, 4] = 1
                    branch_target[idx, integer_x, integer_y, int(cid)] = 1
            except:
                print("stop")
            all_targets.append(branch_target)

        return all_targets




    def _match_best_anchor(self, gt_wh):
        """
        找到与gt boxes最佳匹配的anchor
        :param gt_wh: gtboxes的宽高
        :return: 最佳匹配anchor的列表
        """
        right_bottom = np.minimum(gt_wh[:, None, :], self.anchor_size)

        inter_area = right_bottom[..., 0] * right_bottom[..., 1]

        gt_area = gt_wh[..., 0] * gt_wh[..., 1]

        iou = inter_area / (gt_area[:, None] + self.anchor_area - inter_area)
        best_anchor_idx = np.argmax(iou, -1)
        return best_anchor_idx