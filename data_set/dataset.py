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
from scipy.stats import truncnorm
import time
from scipy import ndimage


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
                f = pd.read_csv(label_csv, header=None)
                img_dirs = sorted(os.listdir(rgb_path))
                for i in range(len(f)):
                    clip = []
                    label, begin, end = f.iloc[i]
                    for j in range(end - begin):
                        # for img in img_dirs[int(begin):int(end)]:
                        img = img_dirs[begin + j]
                        clip.append(os.path.join(rgb_path, img))
                    clips.append((clip, int(label) - 1))
    return clips


class EGOImageFolder(data.Dataset):
    def __init__(self, mode, args):
        if mode == 'train':
            if args.mode == 'test' or args.mode == 'train_test':
                root = '/opt/Ego/model_test/'
            else:
                root = '/opt/Ego/train/'
        elif mode == 'val':
            if args.mode == 'test' or args.mode == 'train_test':
                root = '/opt/Ego/model_test/'
            else:
                root = '/opt/Ego/val/'
        else:
            root = '/opt/Ego/model_test/'
        clips = make_ego_dataset(root)
        print('clips prepare finished for ', root)

        self.clips = clips  # path of data
        self.mode = mode
        self.args = args
        self.corner = [(0, 0), (30, 0), (0, 20), (30, 20), (15, 10)]
        self.count = 0

    def __getitem__(self, index):
        paths, label = self.clips[index]
        length = self.args.clip_length // 2
        while len(paths) < length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        l = len(paths)
        interval = l / length
        uniform_list = [int(i * interval) for i in range(length)]
        # random_list = sorted(
        #     [uniform_list[i] + random.randint(0, int(interval) - 1) for i in range(self.args.clip_length)])
        # random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        # truncated_norm_list = sorted(
        #     truncnorm(a=-1, b=1, loc=(l - 1) / 2, scale=(l - 1) / 2).rvs(size=32).round().astype(int))
        clip = []
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        # start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        # start_train = self.corner[self.count]
        # self.count += 1
        # if self.count == 5:
        #     self.count = 0
        for i in range(length):
            if self.mode == 'train':
                j = uniform_list[i]
                img = Image.open(paths[j])
                img = img.resize(self.args.resize_shape)
                box = ((start_train[0], start_train[1],
                        start_train[0] + self.args.crop_shape[0],
                        start_train[1] + self.args.crop_shape[1])
                       )
                img = img.crop(box)
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
            else:
                j = uniform_list[i]
                img = Image.open(paths[j])
                img = img.resize(self.args.crop_shape)
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
            clip.append(img)
        # make it 64 frame per clip to detection
        for i in range(length):
            clip.append(clip[-1])

        clip = np.array(clip, dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


def make_cha_dataset(mode, args):
    char_root = '/opt/Char/'
    # char_root = '/home/lshi/Database/CharLearn/'
    if args.mode == 'test' or args.mode == 'train_test':
        f = pd.read_csv(os.path.join(char_root, 'model_test.csv'), header=None)
    elif mode == 'train':
        f = pd.read_csv(os.path.join(char_root, 'train.csv'), header=None)
    elif mode == 'val':
        f = pd.read_csv(os.path.join(char_root, 'val.csv'), header=None)
    else:
        f = pd.read_csv(os.path.join(char_root, 'model_test.csv'), header=None)
    clips = []
    print('clip preparing for ', args.mode, '/', mode)
    for img_dirs, label in f.values:
        imgs = sorted(os.listdir(img_dirs))
        clip = []
        for img in imgs:
            clip.append(os.path.join(img_dirs, img))
        clips.append((clip, int(label)))
    return clips


class CHAImageFolder(data.Dataset):
    def __init__(self, mode, args):
        clips = make_cha_dataset(mode, args)
        self.clips = clips
        self.mode = mode
        self.args = args
        self.corner = [(0, 0), (30, 0), (0, 20), (30, 20), (15, 10)]
        self.count = 0

    def __getitem__(self, index):
        paths, label = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        l = len(paths)
        # interval = len(paths) // self.args.clip_length
        # uniform_list = [i * interval for i in range(self.args.clip_length)]
        # random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        # random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        truncated_norm_list = sorted(
            truncnorm(a=-1, b=1, loc=(l - 1) / 2, scale=(l - 1) / 2).rvs(size=self.args.clip_length).round().astype(
                int))
        clip = []
        # pre processions are same for a clip
        # start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        # start_val = [(self.args.resize_shape[j] - self.args.crop_shape[j]) // 2 for j in range(2)]
        start_train = self.corner[self.count]
        self.count += 1
        if self.count == 5:
            self.count = 0
        for i in range(self.args.clip_length):
            if self.mode == 'train':
                j = truncated_norm_list[i]
                img = Image.open(paths[j])
                img = img.resize(self.args.resize_shape)
                img = img.crop((start_train[0], start_train[1],
                                start_train[0] + self.args.crop_shape[0],
                                start_train[1] + self.args.crop_shape[1]))
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
            else:
                j = truncated_norm_list[i]
                img = Image.open(paths[j])
                img = img.resize(self.args.crop_shape)
                # img = img.crop((start_val[0], start_val[1],
                #                 start_val[0] + self.args.crop_shape[0],
                #                 start_val[1] + self.args.crop_shape[1]))
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
            clip.append(img)
        clip = np.array(clip, dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip, label

    def __len__(self):
        return len(self.clips)


def make_jester_dataset(jester_root, mode):
    if mode == 'train':
        # f = pd.read_csv(os.path.join(jester_root, 'model_test.csv'), header=None)
        f = pd.read_csv(os.path.join(jester_root, 'train.csv'), header=None)
    elif mode == 'val':
        # f = pd.read_csv(os.path.join(jester_root, 'model_test.csv'), header=None)
        f = pd.read_csv(os.path.join(jester_root, 'val.csv'), header=None)
    else:
        f = pd.read_csv(os.path.join(jester_root, 'model_test.csv'), header=None)
    clips = []
    # make clips
    print('clip preparing.....')
    for i in range(len(f)):
        img_dirs = os.path.join(jester_root, '20bn-jester-v1', str(f.loc[i, 0]))
        label = f.loc[i, 1]
        imgs = sorted(os.listdir(img_dirs))
        clip = []
        for img in imgs:
            clip.append(os.path.join(img_dirs, img))
        clips.append((clip, int(label)))
    return clips


# TODO: change the other method the same as this
class JesterImageFolder(data.Dataset):
    def __init__(self, mode, args):
        jester_root = '/home/lshi/Database/Jester/'
        # jester_root = '/opt/'
        clips = make_jester_dataset(jester_root, mode)
        print('clips prepare finished for ', jester_root, '  ', mode)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips"))
        self.clips = clips  # path of data
        self.classes = [x for x in range(args.class_num)]
        self.mode = mode
        self.args = args

    def __getitem__(self, index):
        # s = time.time()
        paths, label = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        interval = len(paths) / self.args.clip_length
        uniform_list = [int(i * interval) for i in range(self.args.clip_length)]
        random_list = sorted(
            [uniform_list[i] + random.randint(0, int(interval) - 1) for i in range(self.args.clip_length)])
        # random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        # truncated_norm_list = sorted(truncnorm(a=(0-16)/16, b=(32-16)/16, loc=16, scale=16).rvs(size=32).round().astype(int)
        clip = []
        start_train_ratio = random.random()
        # s1 = time.time()
        # print(s1-s)
        for i in range(self.args.clip_length):
            if self.mode == 'train':
                j = random_list[i]
                img = Image.open(paths[j])
                # s2 = time.time()
                # print(s2-s1)
                start_train = int((img.width - self.args.crop_shape[1]) * start_train_ratio)
                box = (start_train, 0,
                       start_train + self.args.crop_shape[1],
                       self.args.crop_shape[0]
                       )
                img = img.crop(box)
                # s3 = time.time()
                # print(s3-s2)
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
                # s4=time.time()
                # print(s4-s3)
            else:
                j = uniform_list[i]
                img = Image.open(paths[j])
                start_val = (img.width - self.args.crop_shape[1]) // 2
                box = (start_val, 0, start_val +
                       self.args.crop_shape[1],
                       self.args.crop_shape[0]
                       )
                img = img.crop(box)
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
            clip.append(img)
        # s4=time.time()
        # print(s4-s)
        clip = np.array(clip, dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))
        # s5=time.time()
        # print(s5-s)
        return clip, label

    def __len__(self):
        return len(self.clips)


def make_jester_dataset_lstm(jester_root, mode, clip_length, overlap):
    if mode == 'train':
        f = pd.read_csv(os.path.join(jester_root, 'train.csv'), header=None)
    elif mode == 'val':
        f = pd.read_csv(os.path.join(jester_root, 'val.csv'), header=None)
    else:
        f = pd.read_csv(os.path.join(jester_root, 'model_test.csv'), header=None)
    videos = []
    # make videos
    for i in range(len(f)):
        img_dirs = os.path.join(jester_root, '20bn-jester-v1', str(f.loc[i, 0]))
        label = f.loc[i, 1]
        imgs = sorted(os.listdir(img_dirs))
        while len(imgs) < 70:
            imgs.append(imgs[-1])
        clip = []
        video = []
        for img in imgs:
            clip.append(os.path.join(img_dirs, img))
            if len(clip) == clip_length:
                video.append(clip)
                clip = clip[overlap:]  # overlap
        if len(clip) is not 0:
            for _ in range(clip_length - len(clip)):
                clip.append(clip[-1])
            video.append(clip)
        videos.append((video, int(label)))
    return videos


class JesterImageFolderLstm(data.Dataset):
    def __init__(self, mode, args):
        jester_root = '/home/lshi/Database/Jester/'
        clips = make_jester_dataset_lstm(jester_root, mode, args.clip_length, args.overlap)
        print('clips prepare finished for ', jester_root)
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips in subfolders"))
        self.clips = clips  # path of data
        self.classes = [x for x in range(args.class_num)]
        self.mode = mode
        self.args = args

    def __getitem__(self, index):
        video, label = self.clips[index]
        videos = []
        for imgs in video:
            clip = []
            for path in imgs:
                img = Image.open(path)
                if self.mode == 'train':
                    start_train = random.randint(0, img.width - self.args.crop_shape[1])
                    # img = img.resize(self.args.resize_shape)
                    box = (0, start_train,
                           self.args.crop_shape[0],
                           start_train + self.args.crop_shape[1])
                    img = img.crop(box)
                    img = np.array(img)
                    img = (img / 255 - self.args.mean) / self.args.std
                else:
                    start_val = (img.width - self.args.crop_shape[1]) // 2
                    box = (0, start_val,
                           self.args.crop_shape[0],
                           start_val + self.args.crop_shape[1])
                    img = img.crop(box)
                    img = np.array(img)
                    img = (img / 255 - self.args.mean) / self.args.std
                clip.append(img)
            clip = np.array(clip, dtype=np.float32)
            clip = np.transpose(clip, (3, 0, 1, 2))
            videos.append(clip)
        return np.array(videos), label

    def __len__(self):
        return len(self.clips)


def make_jester_dataset_test(jester_root, f):
    clips = []
    # make clips
    print('clip preparing.....')
    for i in range(len(f)):
        img_dirs = os.path.join(jester_root, '20bn-jester-v1', str(f.loc[i, 0]))
        imgs = sorted(os.listdir(img_dirs))
        clip = []
        for img in imgs:
            clip.append(os.path.join(img_dirs, img))
        clips.append(clip)
    return clips


class JesterImageFolderTest(data.Dataset):
    def __init__(self, f, args):
        jester_root = '/home/lshi/Database/Jester/'
        clips = make_jester_dataset_test(jester_root, f)
        print('clips prepare finished')
        if len(clips) == 0:
            raise (RuntimeError("Found 0 clips"))
        self.clips = clips  # path of data
        self.classes = [x for x in range(args.class_num)]
        self.args = args

    def __getitem__(self, index):
        paths = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
        interval = len(paths) // self.args.clip_length
        uniform_list = [i * interval for i in range(self.args.clip_length)]
        # random_list = sorted([uniform_list[i] + random.randint(0, interval - 1) for i in range(self.args.clip_length)])
        # random_list = sorted([random.randint(0, len(paths) - 1) for _ in range(self.args.clip_length)])
        clip = []
        for i in range(self.args.clip_length):
            j = uniform_list[i]
            img = Image.open(paths[j])
            start_val = (img.width - self.args.crop_shape[1]) // 2
            box = (start_val, 0, start_val +
                   self.args.crop_shape[1],
                   self.args.crop_shape[0]
                   )
            img = img.crop(box)
            img = np.array(img)
            img = (img / 255 - self.args.mean) / self.args.std
            clip.append(img)
        clip = np.array(clip, dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))
        return clip

    def __len__(self):
        return len(self.clips)


def make_ego_mask_dataset(root):
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
            depth_dirs = sorted(os.listdir(os.path.join(scene_path, 'Depth')))
            for i in range(len(rgb_dirs)):
                rgb_path = os.path.join(scene_path, 'Color', rgb_dirs[i])
                depth_path = os.path.join(scene_path, 'Depth', depth_dirs[i])
                label_csv = os.path.join(label_scene_path, 'Group' + rgb_dirs[i][-1] + '.csv')
                # print("now for data dir %s" % rgb_path)
                f = pd.read_csv(label_csv, header=None)
                img_dirs = sorted(os.listdir(rgb_path))
                label_depths = sorted(os.listdir(depth_path))
                for j in range(len(f)):
                    clip = []
                    label_depth = []
                    label, begin, end = f.iloc[j]
                    for k in range(end - begin):
                        # for img in img_dirs[int(begin):int(end)]:
                        img = img_dirs[begin - 2 + k]
                        dep = label_depths[begin - 1 + k]
                        clip.append(os.path.join(rgb_path, img))
                        label_depth.append(os.path.join(depth_path, dep))
                    clips.append((clip, label_depth, int(label) - 1))
    return clips


class EGOImageFolderMask(data.Dataset):
    def __init__(self, mode, args):
        if mode == 'train':
            if args.mode == 'test' or args.mode == 'train_test':
                root = '/opt/Ego/model_test/'
            else:
                root = '/opt/Ego/train/'
        elif mode == 'val':
            if args.mode == 'test' or args.mode == 'train_test':
                root = '/opt/Ego/model_test/'
            else:
                root = '/opt/Ego/val/'
        else:
            root = '/opt/Ego/model_test/'
        clips = make_ego_mask_dataset(root)
        print('clips prepare finished for ', root)

        self.clips = clips  # path of data
        self.mode = mode
        self.args = args

    def __getitem__(self, index):
        paths, label_mses, label_ce = self.clips[index]
        while len(paths) < self.args.clip_length:
            tmp = []
            [tmp.extend([x, x]) for x in paths]
            paths = tmp
            tmp2 = []
            [tmp2.extend([x, x]) for x in label_mses]
            label_mses = tmp2
        l = len(paths)
        truncated_norm_list = sorted(
            truncnorm(a=-1, b=1, loc=(l - 1) / 2, scale=(l - 1) / 2).rvs(size=self.args.clip_length).round().astype(
                int))
        clip = []
        label_mse = []
        start_train = [random.randint(0, self.args.resize_shape[j] - self.args.crop_shape[j]) for j in range(2)]
        for i in range(self.args.clip_length):
            if self.mode == 'train':
                j = truncated_norm_list[i]
                img = Image.open(paths[j])
                img = img.resize(self.args.resize_shape)
                box = ((start_train[0], start_train[1],
                        start_train[0] + self.args.crop_shape[0],
                        start_train[1] + self.args.crop_shape[1])
                       )
                img = img.crop(box)
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std

                dep = Image.open(label_mses[j])
                dep = dep.resize(self.args.resize_shape)
                dep = dep.crop(box)
                dep = np.array(dep)
                dep = ndimage.median_filter(dep, 3)
                dep = dep / 255
                label_mse.append(dep)
            else:
                j = truncated_norm_list[i]
                img = Image.open(paths[j])
                img = img.resize(self.args.crop_shape)
                img = np.array(img)
                img = (img / 255 - self.args.mean) / self.args.std
            clip.append(img)
        clip = np.array(clip, dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))
        if self.mode == 'train':
            label_mse = np.array(label_mse, dtype=np.float32)
            label_mse = np.transpose(label_mse, (3, 0, 1, 2))
            label_mse = np.mean(label_mse, 0, keepdims=True)
            return clip, label_mse, label_ce
        else:
            return clip, label_ce

    def __len__(self):
        return len(self.clips)
