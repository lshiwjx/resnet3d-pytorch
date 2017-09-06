import os
import torch
import torch.utils.data as data
import random
import skimage.data
import skimage.transform
import numpy as np

CLIP_LENGTH = 16
MEAN = [0.485, 0.456, 0.406]  # [101, 97, 90]
RESIZE_SHAPE = (120, 160)
CROP_SHAPE = (112, 112)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [cls for cls in os.listdir(dir) if os.path.isdir(os.path.join(dir, cls))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    clips = []
    dir = os.path.expanduser(dir)
    for cls in sorted(os.listdir(dir)):
        cls_path = os.path.join(dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for clp in sorted(os.listdir(cls_path)):
            clip = []
            imgs = os.path.join(cls_path, clp)
            if not os.path.isdir(imgs):
                continue
            for img in sorted(os.listdir(imgs)):
                if is_image_file(img):
                    path = os.path.join(imgs, img)
                    item = (path, class_to_idx[cls])
                    clip.append(item)
            if len(clip) == 0 :
                print('dir is empty: ', imgs)
            else:
                clips.append(clip)
    return clips


class UCFImageFolder(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, is_train):
        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, class_to_idx)
        print('clips prepare finished')
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders of: " + root +
                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.clips = clips  # path of data
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.is_train = is_train

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        paths = self.clips[index]
        random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(CLIP_LENGTH)])
        clip = []
        label = 0
        # pre processions are same for a clip
        start_train = [random.randint(0, RESIZE_SHAPE[j] - CROP_SHAPE[j]) for j in range(2)]
        start_val = [(RESIZE_SHAPE[j] - CROP_SHAPE[j]) // 2 for j in range(2)]
        tmp = random.randint(0, 2)
        for i in random_list:
            path, label = paths[i]
            # img = Image.open(path)
            img = skimage.data.imread(path)
            if self.is_train:
                img = skimage.transform.resize(img, RESIZE_SHAPE, mode='reflect')

                img = img[start_train[0]:start_train[0] + CROP_SHAPE[0], start_train[1]:start_train[1] + CROP_SHAPE[1], :]
                if tmp == 0:
                    img = img[:, ::-1, :]
                elif tmp == 1:
                    img = img[::-1, :, :]
                else:
                    pass
                img -= MEAN
            else:
                img = skimage.transform.resize(img, RESIZE_SHAPE, mode='reflect')
                img = img[start_val[0]:start_val[0] + CROP_SHAPE[0], start_val[1]:start_val[1] + CROP_SHAPE[1], :]
                img -= MEAN
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)
