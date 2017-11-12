from __future__ import print_function, division
from torch.utils.data import DataLoader
from data_set import dataset
import torchvision
import torchvision.transforms as transforms


def data_choose(args):
    if args.mode == 'test':
        if args.data == 'ego':
            data_set = dataset.EGOImageFolder('model_test', args)
            data_set_loaders = DataLoader(data_set, batch_size=4)
        elif args.data == 'ego_mask':
            data_set = dataset.EGOImageFolderMask('model_test', args)
            data_set_loaders = DataLoader(data_set, batch_size=4)
        elif args.data == 'char':
            data_set = dataset.CHAImageFolder('model_test', args)
            data_set_loaders = DataLoader(data_set, batch_size=4)
        elif args.data == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data_set = torchvision.datasets.CIFAR10(root='/opt/cifar10', train=False, transform=transform)
            data_set_loaders = DataLoader(data_set, batch_size=4)
        else:
            raise (RuntimeError('No data loader'))
    else:
        if args.data == 'ego':
            data_set = {x: dataset.EGOImageFolder(x, args) for x in ['train', 'val']}
            data_set_loaders = {x: DataLoader(data_set[x],
                                              batch_size=args.batch_size,
                                              shuffle=(x == 'train'),
                                              num_workers=8,
                                              drop_last=False,
                                              pin_memory=True)
                                for x in ['train', 'val']}
        elif args.data == 'ego_mask':
            data_set = {x: dataset.EGOImageFolderMask(x, args) for x in ['train', 'val']}
            data_set_loaders = {x: DataLoader(data_set[x],
                                              batch_size=args.batch_size,
                                              shuffle=(x == 'train'),
                                              num_workers=8,
                                              drop_last=False,
                                              pin_memory=True)
                                for x in ['train', 'val']}
        elif args.data == 'char':
            data_set = {x: dataset.CHAImageFolder(x, args) for x in ['train', 'val']}
            data_set_loaders = {x: DataLoader(data_set[x],
                                              batch_size=args.batch_size,
                                              shuffle=(x == 'train'),
                                              num_workers=8,
                                              drop_last=False,
                                              pin_memory=True)
                                for x in ['train', 'val']}
        elif args.data == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data_set = {x: torchvision.datasets.CIFAR10(root='/opt/cifar10',
                                                        train=(x is 'train'), download=False, transform=transform)
                        for x in ['train', 'val']}
            data_set_loaders = {x: DataLoader(data_set[x],
                                              batch_size=args.batch_size,
                                              shuffle=(x == 'train'),
                                              num_workers=2,
                                              drop_last=False,
                                              pin_memory=True)
                                for x in ['train', 'val']}
        else:
            raise (RuntimeError('No data loader'))

    print('----Data load finished: ', args.data)
    return data_set_loaders
