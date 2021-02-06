import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy, mse_loss
import numpy as np

class NormalLoss:
    def __init__(self, num_classes):
        self.total_show_loss = {
            "obj_loss": 0,
            "xy_loss": 0,
            "wh_loss": 0,
            "cls_loss": 0,
            "n": 0
        }
        self.channel = 4 + 1 + num_classes

    def __call__(self, pred_list, true_list):
        show_loss = {
            "obj_loss": 0,
            "xy_loss": 0,
            "wh_loss": 0,
            "cls_loss": 0,
        }


        for pred, true in zip(pred_list, true_list):

            batchsize = pred.shape[0]
            fsize = pred.shape[-1]
            pred = pred.view(batchsize, 3, self.channel, fsize, fsize)
            pred = pred.permute(0, 1, 3, 4, 2)

            obj_mask = true[..., 4] == 1

            # pred[..., np.r_[:2, 4:]] = torch.sigmoid(pred[..., np.r_[:2, 4:]])


            obj_loss = binary_cross_entropy(torch.sigmoid(pred[..., 4]), true[..., 4])
            show_loss["obj_loss"] += obj_loss
            if obj_mask.sum() == 0:
                continue
            pred, true = pred[obj_mask], true[obj_mask]
            xy_loss = binary_cross_entropy(torch.sigmoid(pred[..., :2]), true[..., :2])
            wh_loss = mse_loss(pred[..., 2: 4], true[..., 2: 4])
            cls_loss = binary_cross_entropy(torch.sigmoid(pred[..., 5:]), true[..., 5:])
            show_loss["xy_loss"] += xy_loss
            show_loss["wh_loss"] += wh_loss
            show_loss["cls_loss"] += cls_loss

        # 12=3x4，3个branch和4种loss
        total_loss = sum([loss for loss in show_loss.values()]) / 12
        for lname, loss in show_loss.items():
            if lname != "n" and loss != 0:
                self.total_show_loss[lname] += loss.item() / 3
        self.total_show_loss["n"] += 1
        return total_loss

    def get_result(self):
        return {ln: loss / self.total_show_loss["n"] for ln, loss in self.total_show_loss.items() if "loss" in ln}

