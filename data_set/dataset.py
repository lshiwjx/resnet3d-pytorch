import os
import random

import PIL.ImageEnhance as ie
import numpy as np
import pandas as pd
# import skimage.data
# import skimage.transform
import torch.utils.data as data
from PIL import Image  # Replace by accimage when ready
# from torchvision import transforms

from data_set.pre_process import PowerPIL

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(root):
    classes = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, class_to_idx):
    clips = []
    root = os.path.expanduser(root)
    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
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
            if len(clip) == 0:
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

    def __init__(self, root, is_train, args):
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
        self.args = args

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        # print('\rindex: ', index)
        # sys.stdout.flush()
        paths = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        clip = []
        label = 0
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        flip_rand = random.randint(0, 2)
        rotate_rand = random.randint(0, 3)
        flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        contrast = random.randint(5, 10) * 0.1
        sharp = random.randint(5, 15) * 0.1
        bright = random.randint(5, 10) * 0.1
        color = random.randint(5, 10) * 0.1
        for i in random_list:
            path, label = paths[i]
            img = Image.open(path)
            if self.is_train:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                if rotate_rand != 3:
                    img = img.transpose(rotate[rotate_rand])
                if flip_rand != 2:
                    img = img.transpose(flip[flip_rand])
                img = ie.Contrast(img).enhance(contrast)
                img = ie.Color(img).enhance(color)
                img = ie.Brightness(img).enhance(bright)
                img = ie.Sharpness(img).enhance(sharp)
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            else:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_val[0], start_val[1],
                                start_val[0] + self.args.crop_shape[0],
                                start_val[1] + self.args.crop_shape[1]))
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


class UCFImageFolderPlain(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, is_train, args):
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
        self.args = args

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        # print('\rindex: ', index)
        # sys.stdout.flush()
        paths = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        clip = []
        label = 0
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        flip_rand = random.randint(0, 2)
        rotate_rand = random.randint(0, 3)
        flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        for i in random_list:
            path, label = paths[i]
            img = Image.open(path)
            if self.is_train:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                if rotate_rand != 3:
                    img = img.transpose(rotate[rotate_rand])
                if flip_rand != 2:
                    img = img.transpose(flip[flip_rand])
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            else:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_val[0], start_val[1],
                                start_val[0] + self.args.crop_shape[0],
                                start_val[1] + self.args.crop_shape[1]))
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


# with overlap
def make_ego_dataset(root, clip_length):
    data_path = os.path.join(root, 'images')
    label_path = os.path.join(root, 'labels')
    clips = []
    # make clips
    subject_dirs = os.listdir(data_path)
    for subject_dir in subject_dirs:
        subjcet_path = os.path.join(data_path, subject_dir)
        label_subject_path = os.path.join(label_path, subject_dir)
        scene_dirs = os.listdir(subjcet_path)
        for scene_dir in scene_dirs:
            scene_path = os.path.join(subjcet_path, scene_dir)
            label_scene_path = os.path.join(label_subject_path, scene_dir)
            rgb_dirs = sorted(os.listdir(os.path.join(scene_path, 'Color')))
            for rgb_dir in rgb_dirs:
                rgb_path = os.path.join(scene_path, 'Color', rgb_dir)
                label_csv = os.path.join(label_scene_path, 'Group' + rgb_dir[-1] + '.csv')
                # print("now for data dir %s" % rgb_path)
                f = pd.read_csv(label_csv, header=None)
                img_dirs = sorted(os.listdir(rgb_path))
                for i in range(len(f)):
                    clip = []
                    n = 0
                    label, begin, end = f.iloc[i]
                    for j in range((end - begin) // 2):
                        # for img in img_dirs[int(begin):int(end)]:
                        img = img_dirs[begin + j * 2]  # stride=2
                        clip.append(os.path.join(rgb_path, img))
                        n += 1
                        if len(clip) == clip_length:
                            clips.append((clip, int(label) - 1))
                            clip = clip[len(clip) // 2:]  # overlap
                    if len(clip) is not 0:
                        clips.append((clip, int(label) - 1))
    return clips


class EGOImageFolder(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, is_train, args):
        clips = make_ego_dataset(root, args.clip_length)
        print('clips prepare finished for ', root)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders of: " + root +
                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.clips = clips  # path of data
        self.classes = [x for x in range(83)]
        self.is_train = is_train
        self.args = args

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        paths, label = self.clips[index]
        while len(paths) < self.args.clip_length:
            paths += paths
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        clip = []
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        flip_rand = random.randint(0, 2)
        rotate_rand = random.randint(0, 3)
        flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        contrast = random.randint(5, 10) * 0.1
        sharp = random.randint(5, 15) * 0.1
        bright = random.randint(5, 10) * 0.1
        color = random.randint(5, 10) * 0.1
        for i in random_list:
            path = paths[i]
            img = Image.open(path)
            if self.is_train:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                if rotate_rand != 3:
                    img = img.transpose(rotate[rotate_rand])
                if flip_rand != 2:
                    img = img.transpose(flip[flip_rand])
                img = ie.Contrast(img).enhance(contrast)
                img = ie.Color(img).enhance(color)
                img = ie.Brightness(img).enhance(bright)
                img = ie.Sharpness(img).enhance(sharp)
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            else:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_val[0], start_val[1],
                                start_val[0] + self.args.crop_shape[0],
                                start_val[1] + self.args.crop_shape[1]))
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


# with transform
def make_ego_dataset_augment(root):
    data_path = os.path.join(root, 'images')
    label_path = os.path.join(root, 'labels')
    clips = []
    # make clips
    subject_dirs = os.listdir(data_path)
    for subject_dir in subject_dirs:
        subjcet_path = os.path.join(data_path, subject_dir)
        label_subject_path = os.path.join(label_path, subject_dir)
        scene_dirs = os.listdir(subjcet_path)
        for scene_dir in scene_dirs:
            scene_path = os.path.join(subjcet_path, scene_dir)
            label_scene_path = os.path.join(label_subject_path, scene_dir)
            rgb_dirs = sorted(os.listdir(os.path.join(scene_path, 'Color')))
            for rgb_dir in rgb_dirs:
                rgb_path = os.path.join(scene_path, 'Color', rgb_dir)
                label_csv = os.path.join(label_scene_path, 'Group' + rgb_dir[-1] + '.csv')
                # print("now for data dir %s" % rgb_path)
                f = pd.read_csv(label_csv, header=None)
                img_dirs = sorted(os.listdir(rgb_path))
                for i in range(len(f)):
                    clip = []
                    label, begin, end = f.iloc[i]
                    for img in img_dirs[begin:end]:
                        clip.append(os.path.join(rgb_path, img))
                    clips.append((clip, int(label) - 1))
    return clips


class EGOImageFolderAugment(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, is_train, args):
        clips = make_ego_dataset_augment(root)
        print('clips prepare finished for ', root)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders of: " + root +
                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.clips = clips  # path of data
        self.classes = [x for x in range(args.class_num)]
        self.is_train = is_train
        self.args = args
        if is_train:
            self.transform = transforms.Compose([
                transforms.Scale(args.crop_shape),
                # transforms.RandomSizedCrop(args.crop_shape[0]),
                PowerPIL(),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Scale(args.crop_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std)
            ])

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        paths, label = self.clips[index]
        while len(paths) < self.args.clip_length:
            paths += paths
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        clip = []
        seed = np.random.randint(2147483647)
        for i in random_list:
            path = paths[i]
            img = Image.open(path)
            if self.transform is not None:
                random.seed(seed)
                img = self.transform(img)
            clip.append(img.numpy())
        clip = np.array(clip)
        # lchw->clhw
        clip = np.transpose(clip, (1, 0, 2, 3))
        return clip, label

    def __len__(self):
        return len(self.clips)


# with pillow
def make_ego_dataset_pillow(root):
    data_path = os.path.join(root, 'images')
    label_path = os.path.join(root, 'labels')
    clips = []
    # make clips
    subject_dirs = os.listdir(data_path)
    for subject_dir in subject_dirs:
        subjcet_path = os.path.join(data_path, subject_dir)
        label_subject_path = os.path.join(label_path, subject_dir)
        scene_dirs = os.listdir(subjcet_path)
        for scene_dir in scene_dirs:
            scene_path = os.path.join(subjcet_path, scene_dir)
            label_scene_path = os.path.join(label_subject_path, scene_dir)
            rgb_dirs = sorted(os.listdir(os.path.join(scene_path, 'Color')))
            for rgb_dir in rgb_dirs:
                rgb_path = os.path.join(scene_path, 'Color', rgb_dir)
                label_csv = os.path.join(label_scene_path, 'Group' + rgb_dir[-1] + '.csv')
                # print("now for data dir %s" % rgb_path)
                f = pd.read_csv(label_csv, header=None)
                img_dirs = sorted(os.listdir(rgb_path))
                for i in range(len(f)):
                    clip = []
                    label, begin, end = f.iloc[i]
                    for img in img_dirs[begin:end]:
                        clip.append(os.path.join(rgb_path, img))
                    clips.append((clip, int(label) - 1))
    return clips


class EGOImageFolderPillow(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, is_train, args):
        clips = make_ego_dataset_pillow(root)
        print('clips prepare finished for ', root)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders of: " + root +
                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.clips = clips  # path of data
        self.classes = [x for x in range(83)]
        self.is_train = is_train
        self.args = args

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        paths, label = self.clips[index]
        while len(paths) < 150:  # self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        clip = []
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        flip_rand = random.randint(0, 2)
        rotate_rand = random.randint(0, 3)
        flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        contrast = random.randint(5, 10) * 0.1
        sharp = random.randint(5, 15) * 0.1
        bright = random.randint(5, 10) * 0.1
        color = random.randint(5, 10) * 0.1
        for i in random_list:
            path = paths[i]
            img = Image.open(path)
            if self.is_train:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                if rotate_rand != 3:
                    img = img.transpose(rotate[rotate_rand])
                if flip_rand != 2:
                    img = img.transpose(flip[flip_rand])
                img = ie.Contrast(img).enhance(contrast)
                img = ie.Color(img).enhance(color)
                img = ie.Brightness(img).enhance(bright)
                img = ie.Sharpness(img).enhance(sharp)
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            else:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_val[0], start_val[1],
                                start_val[0] + self.args.crop_shape[0],
                                start_val[1] + self.args.crop_shape[1]))
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


# with pillow
def make_cha_dataset_pillow(root, is_train):
    if is_train:
        f = open(os.path.join(root, '../train_list'))
    else:
        f = open(os.path.join(root, '../val_list'))
    lines = f.readlines()
    a = []
    for line in lines:
        a.append(int(line.split(' ')[2][:-1]))
    rgb_dirs = sorted(os.listdir(root))
    clips = []
    # make clips
    for i in range(len(rgb_dirs)):
        rgb_path = os.path.join(root, rgb_dirs[i])
        label = a[i]
        img_dirs = sorted(os.listdir(rgb_path))
        clip = []
        for img in img_dirs:
            clip.append(os.path.join(rgb_path, img))
        clips.append((clip, int(label - 1)))
    return clips


class CHAImageFolderPillow(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, is_train, args):
        clips = make_cha_dataset_pillow(root, is_train)
        print('clips prepare finished for ', root)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders of: " + root +
                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.clips = clips  # path of data
        self.classes = [x for x in range(args.class_num)]
        self.is_train = is_train
        self.args = args

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        paths, label = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        # interval = len(paths) // self.args.clip_length
        # uniform_list = [i * interval for i in range(self.args.clip_length)]
        # random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        clip = []
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        flip_rand = random.randint(0, 2)
        rotate_rand = random.randint(0, 3)
        flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        contrast = random.randint(5, 10) * 0.1
        sharp = random.randint(5, 15) * 0.1
        bright = random.randint(5, 10) * 0.1
        color = random.randint(5, 10) * 0.1
        for i in random_list:
            path = paths[i]
            img = Image.open(path)
            if self.is_train:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                if rotate_rand != 3:
                    img = img.transpose(rotate[rotate_rand])
                if flip_rand != 2:
                    img = img.transpose(flip[flip_rand])
                img = ie.Contrast(img).enhance(contrast)
                img = ie.Color(img).enhance(color)
                img = ie.Brightness(img).enhance(bright)
                img = ie.Sharpness(img).enhance(sharp)
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            else:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_val[0], start_val[1],
                                start_val[0] + self.args.crop_shape[0],
                                start_val[1] + self.args.crop_shape[1]))
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


# with pillow
def make_jester_dataset(jester_root, is_train):
    if is_train:
        f = pd.read_csv(os.path.join(jester_root, 'val.csv'))
    else:
        f = pd.read_csv(os.path.join(jester_root, 'val.csv'))
    clips = []
    # make clips
    for i in range(len(f)):
        img_dirs = os.path.join(jester_root, '20bn-jester-v1', str(f.loc[i, 0]))
        label = f.loc[i, 1]
        imgs = sorted(os.listdir(img_dirs))
        clip = []
        for img in imgs:
            clip.append(os.path.join(img_dirs, img))
        clips.append((clip, int(label)))
    return clips


class JesterImageFolder(data.Dataset):
    """A generic data loader where the clips are arranged in this way: ::

        root/class/clip/xxx.jpg

    Args:
        root (string): Root directory path.

     Attributes:
        clips (list): List of (image path, class_index) tuples
    """

    def __init__(self, is_train, args):
        jester_root = '/home/lshi/Database/Jester'
        clips = make_jester_dataset(jester_root, is_train)
        print('clips prepare finished for ', jester_root)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders of: " + jester_root +
                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = jester_root
        self.clips = clips  # path of data
        self.classes = [x for x in range(args.class_num)]
        self.is_train = is_train
        self.args = args

    def __getitem__(self, index):
        """
        It is a little slow because of the preprocess
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is class_index of the label class.
        """
        paths, label = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        # random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        clip = []
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        flip_rand = random.randint(0, 2)
        rotate_rand = random.randint(0, 3)
        # flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        # rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        for i in random_list:
            path = paths[i]
            img = Image.open(path)
            if self.is_train:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                # if rotate_rand != 3:
                #     img = img.transpose(rotate[rotate_rand])
                # if flip_rand != 2:
                #     img = img.transpose(flip[flip_rand])
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            else:
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_val[0], start_val[1],
                                start_val[0] + self.args.crop_shape[0],
                                start_val[1] + self.args.crop_shape[1]))
                img = np.array(img, dtype=float)
                img -= self.args.mean
                img /= 255
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)
