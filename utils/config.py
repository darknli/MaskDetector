from easydict import EasyDict as edict


conf = edict()

conf.images_root = "D:/AAA/Learn/Dataset/train"
conf.train_path = "anno/train.txt"
conf.val_path = "anno/val.txt"

conf.num_epochs = 100
conf.batch_size = 6
conf.lr = 0.01
conf.warmup_steps = [10000, 20000]
conf.num_workers = 2

conf.is_mobile = True
conf.num_classes = 2
conf.size = 224
conf.bgr_mean = (104, 117, 123)
