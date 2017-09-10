import os
import torch
import torch.utils.data as data
import random
import skimage.data
import skimage.transform
import numpy as np

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
        paths = self.clips[index]
        random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        clip = []
        label = 0
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        tmp = random.randint(0, 2)
        for i in random_list:
            path, label = paths[i]
            # img = Image.open(path)
            img = skimage.data.imread(path)
            if self.is_train:
                img = skimage.transform.resize(img, self.args.resize_shape, mode='reflect')

                img = img[start_train[0]:start_train[0] + self.args.crop_shape[0],
                      start_train[1]:start_train[1] + self.args.crop_shape[1], :]
                if tmp == 0:
                    img = img[:, ::-1, :]
                elif tmp == 1:
                    img = img[::-1, :, :]
                else:
                    pass
                img -= self.args.mean
            else:
                img = skimage.transform.resize(img, self.args.resize_shape, mode='reflect')
                img = img[start_val[0]:start_val[0] + self.args.crop_shape[0],
                      start_val[1]:start_val[1] + self.args.crop_shape[1], :]
                img -= self.args.mean
            clip.append(img)
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


def make_ego_dataset(root):
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
                f = open(label_csv)
                img_dirs = sorted(os.listdir(rgb_path))
                for line in f.readlines():
                    clip = []
                    label, begin, end = line.split('\n')[0].split(',')
                    for img in img_dirs[int(begin):int(end)]:
                        clip.append(os.path.join(rgb_path, img))
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
        clips = make_ego_dataset(root)
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
        # interval = len(paths) // self.args.clip_length
        # uniform_list = [i * interval for i in range(self.args.clip_length)]
        # random_list = sorted([uniform_list[i] + random.randint(0, interval-1) for i in range(self.args.clip_length)])
        interval = 3
        uniform_list = [i * interval for i in range(len(paths) // interval)]
        random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(len(paths) // interval)])
        clip = []
        # pre processions are same for a clip
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        tmp = random.randint(0, 2)
        # n = 0
        for i in random_list:
            path = paths[i]
            # img = Image.open(path)
            img = skimage.data.imread(path)
            if self.is_train:
                img = skimage.transform.resize(img, self.args.resize_shape, mode='edge')
                img = img[start_train[0]:start_train[0] + self.args.crop_shape[0],
                      start_train[1]:start_train[1] + self.args.crop_shape[1], :]
                if tmp == 0:
                    img = img[:, ::-1, :]
                elif tmp == 1:
                    img = img[::-1, :, :]
                else:
                    pass
                img -= self.args.mean
            else:
                img = skimage.transform.resize(img, self.args.resize_shape, mode='edge')
                img = img[start_val[0]:start_val[0] + self.args.crop_shape[0],
                      start_val[1]:start_val[1] + self.args.crop_shape[1], :]
                img -= self.args.mean
            clip.append(img)
            # n+=1
            # if n >8
        clip = np.array(clip)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)
