import os
import shutil
import random

data_dir = '/home/lshi/Database/UCF-101/'

classes = sorted(os.listdir(data_dir))
os.chdir(data_dir)
os.mkdir('train')
os.mkdir('val')
for cls in classes:
    train_dir = os.path.join(data_dir, 'train', cls)
    val_dir = os.path.join(data_dir, 'val', cls)
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    cls_dir = os.path.join(data_dir, cls)
    clips = sorted(os.listdir(cls_dir))
    for clip in clips:
        clp_dir = os.path.join(cls_dir, clip)
        tmp = random.randint(0, 5)
        if tmp > 4:
            shutil.move(clp_dir, val_dir)
        else:
            shutil.move(clp_dir, train_dir)
    shutil.rmtree(cls_dir)
