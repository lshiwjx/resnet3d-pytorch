import argparse
import os


# now no crop
def parser_args():
    # params
    parser = argparse.ArgumentParser()
    # resnet3d_18 resnet3d_mask deform_resnet3d_18 simple_model lenet lenet_stn lenet_mask plenet lenet_deform
    parser.add_argument('-model', default='deform_resnet3d_18')
    # ego cifar10 char ego_mask
    parser.add_argument('-data', default='ego')
    # train test train_test
    parser.add_argument('-mode', default='train')
    # cross_entropy mse_ce
    parser.add_argument('-loss', default='cross_entropy')
    # adam sgd
    parser.add_argument('-optimizer', default='adam')
    # 83 10 249
    parser.add_argument('-class_num', default=83)
    parser.add_argument('-batch_size', default=64)
    parser.add_argument('-weight_decay_ratio', default=5e-4)  # 5e-4
    parser.add_argument('-wdr', default=1)
    parser.add_argument('-momentum', default=0.9)
    parser.add_argument('-max_epoch', default=50)

    parser.add_argument('-lr', default=0.001)
    parser.add_argument('-lr_decay_ratio', default=0.1)
    parser.add_argument('-lr_patience', default=3)
    parser.add_argument('-lr_threshold', default=0.01)
    parser.add_argument('-lr_delay', default=0)
    parser.add_argument('-deform_lr_ratio', default=1)

    NAME = './runs/ego_ldevide8_deforml3_51_1lr_1wd_adam_cinit'
    parser.add_argument('-log_dir', default=NAME)
    parser.add_argument('-num_epoch_per_save', default=5)
    parser.add_argument('-model_saved_name', default=NAME)

    parser.add_argument('-use_last_model', default=False)
    parser.add_argument('-last_model', default='./runs/ego_ldevide8_deforml3_51_1lr_0.1wd_sgd-27315.state')

    parser.add_argument('-use_pre_trained_model', default=True)
    parser.add_argument('-pre_class_num', default=27)
    # /opt/model/jes-res18-240-0.92.state /opt/model/cifar10_base_100_0.001_0.0005_p3-20000.state
    parser.add_argument('-pre_trained_model', default='/opt/model/jes-res18-240-0.92.state')
    parser.add_argument('-only_train_classifier', default=False)

    parser.add_argument('-clip_length', default=32)
    parser.add_argument('-resize_shape', default=[160, 120])  # wh
    parser.add_argument('-crop_shape', default=[128, 96])  # must be same for rotate
    # cha[0.486,0.423,0.45] ego[0.45, 0.48, 0.49] ucf[101,97,90] jester[0.45, 0.48, 0.49]
    parser.add_argument('-mean', default=[0.45, 0.48, 0.49])
    parser.add_argument('-std', default=[0.23, 0.24, 0.23])

    # parser.add_argument('-device_id', default=[0,1,2,3])
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4'
    parser.add_argument('-device_id', default=[0, 1])
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,4'
    # parser.add_argument('-device_id', default=[0])
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()
    return args


def parser_cifar_args():
    # params
    parser = argparse.ArgumentParser()
    # resnet3d_18 deform_resnet3d_18 simple_model lenet lenet_stn lenet_mask plenet lenet_deform lenetp
    parser.add_argument('-model', default='lenet_deform')
    # ego cifar10 char
    parser.add_argument('-data', default='cifar10')
    # train test train_test
    parser.add_argument('-mode', default='train')
    # cross_entropy mse_ce
    parser.add_argument('-loss', default='cross_entropy')
    # adam sgd
    parser.add_argument('-optimizer', default='sgd')
    # 83 10 249
    parser.add_argument('-class_num', default=10)
    parser.add_argument('-batch_size', default=200)
    parser.add_argument('-weight_decay_ratio', default=5e-4)  # 5e-4
    parser.add_argument('-wdr', default=0.1)
    parser.add_argument('-momentum', default=0.9)
    parser.add_argument('-max_epoch', default=40)

    parser.add_argument('-lr', default=0.001)
    parser.add_argument('-lr_decay_ratio', default=0.1)
    parser.add_argument('-lr_patience', default=3)
    parser.add_argument('-lr_threshold', default=0.01)
    parser.add_argument('-lr_delay', default=0)
    parser.add_argument('-deform_lr_ratio', default=1)

    NAME = './runs/c_deform_l2_1lr_0.1wd_sgd_nopre'
    parser.add_argument('-log_dir', default=NAME)
    parser.add_argument('-num_epoch_per_save', default=50)
    parser.add_argument('-model_saved_name', default=NAME)

    parser.add_argument('-use_last_model', default=True)
    parser.add_argument('-last_model', default='./runs/c_deform_l2_1lr_0.1wd_sgd_nopre-10000.state')

    parser.add_argument('-use_pre_trained_model', default=False)
    parser.add_argument('-pre_class_num', default=10)
    # /opt/model/jes-res18-240-0.92.state /opt/model/cifar10_base_100_0.001_0.0005_p3-20000.state
    parser.add_argument('-pre_trained_model', default='/opt/model/cifar10_base_100_0.001_0.0005_p3-20000.state')
    parser.add_argument('-only_train_classifier', default=False)

    parser.add_argument('-clip_length', default=32)
    parser.add_argument('-resize_shape', default=[120, 160])
    parser.add_argument('-crop_shape', default=[112, 112])  # must be same for rotate
    # cha[0.486,0.423,0.45] ego[114,123,125] ucf[101,97,90] jester[0.45, 0.48, 0.49]
    parser.add_argument('-mean', default=[0.45, 0.48, 0.49])
    parser.add_argument('-std', default=[0.23, 0.24, 0.23])

    # parser.add_argument('-device_id', default=[0,1,2,3])
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4'
    # parser.add_argument('-device_id', default=[0, 1])t
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5,4'
    parser.add_argument('-device_id', default=[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()
    return args
