from __future__ import print_function, division
from model import resnet3d_18
from model import simple_model
import torch


def model_choose(args):
    if args.mode == 'test':
        m = args.model
        if m == 'resnet3d_18':
            model = resnet3d_18.ResNet3d(args.class_num, args.clip_length, args.crop_shape, 'test')
        elif m == 'deform_resnet3d_18':
            model = resnet3d_18.DeformResNet3d(args.class_num, args.clip_length, args.crop_shape, 'test')
        elif m == 'resnet3d_mask':
            model = resnet3d_18.ResNet3dMask(args.class_num, args.clip_length, args.crop_shape, 'test')
        elif m == 'simple_model':
            model = simple_model.SimpleNet(args.class_num, 'test')
        elif m == 'lenet':
            model = simple_model.LeNet('test')
        elif m == 'lenet_stn':
            model = simple_model.LeNetStn('test')
        elif m == 'lenet_mask':
            model = simple_model.LeNetMask('test')
        elif m == 'lenet_deform':
            model = simple_model.LeNetDeform('test')
        elif m == 'plenet':
            model = simple_model.PLeNet('test')
        elif m == 'lenetp':
            model = simple_model.LeNetP('test')
        else:
            raise (RuntimeError("No modules"))
        print('----Model load finished: ', args.model, '  mode: test')

        return 0, model
    else:
        m = args.model
        if m == 'resnet3d_18':
            model = resnet3d_18.ResNet3d(args.class_num, args.clip_length, args.crop_shape, 'train')
        elif m == 'deform_resnet3d_18':
            model = resnet3d_18.DeformResNet3d(args.class_num, args.clip_length, args.crop_shape, 'train')
        elif m == 'resnet3d_mask':
            model = resnet3d_18.ResNet3dMask(args.class_num, args.clip_length, args.crop_shape, 'train')
        elif m == 'simple_model':
            model = simple_model.SimpleNet(args.class_num, 'train')
        elif m == 'lenet':
            model = simple_model.LeNet('train')
        elif m == 'lenet_stn':
            model = simple_model.LeNetStn('train')
        elif m == 'lenet_mask':
            model = simple_model.LeNetMask('train')
        elif m == 'lenet_deform':
            model = simple_model.LeNetDeform('train')
        elif m == 'plenet':
            model = simple_model.PLeNet('train')
        elif m == 'lenetp':
            model = simple_model.LeNetP('train')
        else:
            raise (RuntimeError("No modules"))
        print('----Model load finished: ', args.model, ' mode: train')

        if args.use_pre_trained_model:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.pre_trained_model)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if args.pre_class_num != args.class_num:
                pretrained_dict.pop('fc.weight')
                pretrained_dict.pop('fc.bias')
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('----Pretrained model load finished: ', args.pre_trained_model)

        if args.only_train_classifier:
            print('----Only train classifier')
            for key, value in model.named_parameters():
                if not key[0:3] == 'stn':
                    value.requires_grad = False

        global_step = 0
        # The name for model must be **_**-$(step).state
        if args.use_last_model:
            model.load_state_dict(torch.load(args.last_model))
            global_step = int(args.last_model[:-6].split('-')[1])
            print('----Training continue, last model load finished, step is ', global_step)

        model.cuda()
        return global_step, model
