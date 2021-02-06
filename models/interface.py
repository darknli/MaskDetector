from models.yolov4 import BaseModel


def get_model(model_name, num_classes, is_mobile):
    if model_name == "base_yolo":
        return BaseModel(num_classes, is_mobile)
