import os
import random
import skimage.data as ski
import numpy as np


def ucf101():
    data_path = '/home/sl/Resource/UCF101/'
    store_path = './list/UCF101/'
    random_num = 4

    # make train.txt and val.txt
    data_dirs = sorted(os.listdir(data_path))
    train_list = []
    val_list = []
    label = -1
    count_for_train = 0
    count_for_val = 0
    mean_list = []
    label_list = []

    for data_dir in data_dirs:
        video_path = os.path.join(data_path, data_dir)
        video_dirs = sorted(os.listdir(video_path))
        label += 1
        label_list.append(data_dir)
        for video_dir in video_dirs:
            print("now for data dir %s\n" % video_dir)
            img_path = os.path.join(video_path, video_dir)
            if not random.randint(0, random_num) == random_num:
                train_list.append((img_path, label))
                count_for_train += 1
            else:
                val_list.append((img_path, label))
                count_for_val += 1
            # calculate mean of img
            imgs = sorted(os.listdir(img_path))
            for img in imgs:
                pic_path = os.path.join(img_path, img)
                pic = ski.imread(pic_path)
                mean_list.append(np.mean(pic, (0, 1)))

    os.chdir(store_path)
    train_file = open('train.txt', 'w')
    for item in train_list:
        train_file.write("%s %s \n" % (item[0], item[1]))
    train_file.close()

    val_file = open('val.txt', 'w')
    for item in val_list:
        val_file.write("%s %s \n" % (item[0], item[1]))
    val_file.close()

    label_file = open('label.txt', 'w')
    for item in label_list:
        label_file.write("%s \n" % item)
    label_file.close()

    mean = np.mean(np.array(mean_list), 0)
    others_file = open('others.txt', 'w')
    others_file.write("mean : %.2f %.2f %.2f\n" % (mean[0], mean[1], mean[2]))
    others_file.write("train num: %d\n" % count_for_train)
    others_file.write("val num: %d\n" % count_for_val)
    print("train num: ", count_for_train)
    print("val num: ", count_for_val)
    print("mean val: ", mean)


def ego_gesture():
    database_path = '/home/lshi/Database/Ego_gesture/train/'
    label_path = os.path.join(database_path, 'labels')
    data_path = os.path.join(database_path, 'images')
    store_path = './list/ego_gesture/'

    train_list = []
    val_list = []
    label_train_list = []
    label_val_list = []
    count_for_train = 0
    count_for_val = 0
    mean_list = []

    random_num = 5
    # make train.txt and val.txt
    subject_dirs = sorted(os.listdir(data_path))
    for subject_dir in subject_dirs:
        subjcet_path = os.path.join(data_path, subject_dir)
        label_subject_path = os.path.join(label_path, subject_dir)
        scene_dirs = sorted(os.listdir(subjcet_path))
        for scene_dir in scene_dirs:
            # if not scene_dir=='Scene6':
            #    continue
            scene_path = os.path.join(subjcet_path, scene_dir)
            label_scene_path = os.path.join(label_subject_path, scene_dir)
            rgb_dirs = sorted(os.listdir(os.path.join(scene_path, 'Color')))
            for rgb_dir in rgb_dirs:
                rgb_path = os.path.join(scene_path, 'Color', rgb_dir)
                label_csv = os.path.join(label_scene_path,
                                         'Group' + rgb_dir[-1] + '.csv')
                print("now for data dir %s" % rgb_dir)
                if not random.randint(0, random_num) == random_num:
                    train_list.append(rgb_path)
                    label_train_list.append(label_csv)
                    count_for_train += 1
                else:
                    val_list.append(rgb_path)
                    label_val_list.append(label_csv)
                    count_for_val += 1
                    # calculate mean of img, only for val, 1st img
                    pic_path = os.path.join(rgb_path, '000001.jpg')
                    pic = ski.imread(pic_path)
                    mean_list.append(np.mean(pic, (0, 1)))

    os.chdir(store_path)
    train_file = open('train.txt', 'w')
    for item in train_list:
        train_file.write("%s\n" % item)
    train_file.close()

    val_file = open('val.txt', 'w')
    for item in val_list:
        val_file.write("%s\n" % item)
    val_file.close()

    label_train_file = open('label_train.txt', 'w')
    for item in label_train_list:
        label_train_file.write("%s\n" % item)
    label_train_file.close()

    label_val_file = open('label_val.txt', 'w')
    for item in label_val_list:
        label_val_file.write("%s\n" % item)
    label_val_file.close()

    mean = np.mean(np.array(mean_list), 0)
    others_file = open('others.txt', 'w')
    others_file.write("mean : %.2f %.2f %.2f\n" % (mean[0], mean[1], mean[2]))
    others_file.write("train num: %d\n" % count_for_train)
    others_file.write("val num: %d\n" % count_for_val)
    print("train num: ", count_for_train)
    print("val num: ", count_for_val)
    print("mean val: ", mean)


def chalearn():
    data_path = '/home/lshi/Database/ChaLearn/val'
    store_path = './list/ChaLearn/'

    data_dirs = sorted(os.listdir(data_path))
    mean_list = []

    for data_dir in data_dirs:
        img_path = os.path.join(data_path, data_dir)
        imgs = sorted(os.listdir(img_path))
        print("now for data dir %s" % img_path)
        for img in imgs:
            pic_path = os.path.join(img_path, img)
            pic = ski.imread(pic_path)
            mean_list.append(np.mean(pic, (0, 1)))

    mean = np.mean(np.array(mean_list), 0)
    print("mean val: ", mean)
    others_file = open(store_path + 'others.txt', 'w')
    others_file.write("mean : %.2f %.2f %.2f\n" % (mean[0], mean[1], mean[2]))


# mean std rows cols clips
def jester():
    data_path = '/home/lshi/Database/Jester/20bn-jester-v1/'
    store_path = '/home/lshi/Database/Jester/'

    data_dirs = sorted(os.listdir(data_path))
    mean_list = []
    std_list = []
    row_list = []
    col_list = []
    clip_len = []
    for data_dir in data_dirs:
        img_path = os.path.join(data_path, data_dir)
        imgs = sorted(os.listdir(img_path))
        clip_len.append(len(imgs))
        print("now for data dir %s" % img_path)
        for img in imgs:
            pic_path = os.path.join(img_path, img)
            pic = ski.imread(pic_path)
            mean_list.append(np.mean(pic, (0, 1)))
            std_list.append(np.std(pic, (0, 1)))
            row_list.append(len(pic))
            col_list.append(len(pic[0]))

    mean = np.mean(np.array(mean_list), 0)
    std = np.mean(np.array(std_list), 0)
    max_row = max(row_list)
    min_row = min(row_list)
    max_col = max(col_list)
    min_col = min(col_list)
    max_clip = max(clip_len)
    min_clip = min(clip_len)
    mean_clip = sum(clip_len) / len(clip_len)
    print("mean val: ", mean)
    others_file = open(store_path + 'others.txt', 'w')
    others_file.write("mean : %.2f %.2f %.2f mean clip: %.2f\n" % (mean[0], mean[1], mean[2], mean_clip))
    others_file.write("max row: %.2f col: %.2f clip: %.2f\n" % (max_row, max_col, max_clip))
    others_file.write("min row: %.2f col: %.2f clip: %.2f\n" % (min_row, min_col, min_clip))


if __name__ == '__main__':
    jester()
