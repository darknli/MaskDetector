from glob import glob
from os.path import join
import shutil

def clear_weights(dir, min_n=1):
    checkpoints_dir_list = glob(join(dir, "*"))
    print(checkpoints_dir_list)
    for checkpoints_dir in checkpoints_dir_list:
        weights = glob(join(checkpoints_dir, "*.pth"))
        if len(weights) < min_n:
            print("delete {}...".format(checkpoints_dir))
            shutil.rmtree(checkpoints_dir)

if __name__ == '__main__':
    clear_weights("checkpoint", 2)