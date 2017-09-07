import torch
from torch.autograd import Variable
from tensorboard_logger import log_value


def train(model, data_set_loaders, loss_function, optimizer, global_step,
          use_gpu=True, device_id=(0, 1)):
    model.train()  # Set model to training mode
    step = global_step
    for data in data_set_loaders:
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(async=True)), \
                             Variable(labels.cuda(async=True))
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        if use_gpu:
            # use DataParallel to realize multi gpu
            net = torch.nn.DataParallel(model, device_ids=device_id)
            outputs = net(inputs.float())
        else:
            outputs = model(inputs.float())

        loss = loss_function(outputs, labels)
        value, predict_label = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()
        step += 1

        # statistics
        ls = loss.data[0]
        acc = torch.mean((predict_label == labels.data).float())
        log_value('train_loss', ls, step)
        log_value('train_acc', acc, step)
        print('step: {}, loss {:.4f}, acc {:.4f}'.format(step, ls, acc))

    return step


def validate(model, data_set_loaders, loss_function, use_gpu=True, device_id=(0, 1)):
    model.eval()
    val_step = 0
    val_loss = 0
    val_acc = 0
    for data in data_set_loaders:
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(async=True)), \
                             Variable(labels.cuda(async=True))
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        if use_gpu:
            net = torch.nn.DataParallel(model, device_ids=device_id)
            outputs = net(inputs.float())
        else:
            outputs = model(inputs.float())

        # return value and index
        _, predict_label = torch.max(outputs.data, 1)
        loss = loss_function(outputs, labels)

        # statistics
        ls = loss.data[0]
        acc = torch.mean((predict_label == labels.data).float())
        val_loss += ls
        val_acc += acc
        val_step += 1

    return val_loss / val_step, val_acc / val_step
