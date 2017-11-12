import torch


def loss_choose(loss):
    if loss == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss(size_average=True)
    elif loss == 'mse_ce':
        loss_function = [torch.nn.MSELoss(), torch.nn.CrossEntropyLoss(size_average=True)]
    else:
        loss_function = torch.nn.CrossEntropyLoss(size_average=True)

    print('----Using loss: ', loss)

    return loss_function
