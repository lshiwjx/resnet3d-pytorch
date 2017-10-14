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
                             Variable(labels.float().cuda(async=True))
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        if use_gpu:
            # use DataParallel to realize multi gpu
            net = torch.nn.DataParallel(model, device_ids=device_id)
            outputs, _ = net(inputs)
        else:
            outputs, _ = model(inputs)

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


def validate(model, data_set_loaders, loss_function, batch_size, use_gpu=True, device_id=(0, 1)):
    model.eval()
    val_step = 0
    val_loss = 0
    right_num = 0
    for data in data_set_loaders:
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(async=True)), \
                             Variable(labels.float().cuda(async=True))
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        if use_gpu:
            net = torch.nn.DataParallel(model, device_ids=device_id)
            outputs, _ = net(inputs)
        else:
            outputs, _ = model(inputs)

        # return value and index
        _, predict_label = torch.max(outputs.data, 1)
        loss = loss_function(outputs, labels)

        # statistics
        ls = loss.data[0]
        right_num += torch.sum(predict_label == labels.data)
        val_loss += ls
        val_step += 1
        print('val step {} loss {:.4f} right num {}'.format(val_step, ls, right_num))
    val_acc = float(right_num) / val_step / batch_size
    val_loss /= val_step
    print('Average val loss: {:.4f}, average val accuracy: {:.4f}'.format(val_loss, val_acc))
    return val_loss, val_acc


def train_lstm(cnnmodel, lstmmodel, data_set_loaders, loss_function, optimizer, global_step):
    cnnmodel.train()  # Set cnnmodel to training mode
    lstmmodel.train()
    step = global_step
    outputs = []
    for data in data_set_loaders:
        video, labels = data
        labels = Variable(labels.cuda(async=True))
        for clip in video:
            clip = Variable(clip.float().cuda(async=True))
            optimizer.zero_grad()
            output, _ = cnnmodel(clip)
            outputs.append(torch.unsqueeze(output, 0))
        outputs = torch.cat(outputs)
        final, _ = lstmmodel(outputs)
        final = torch.mean(final, 0)
        loss = loss_function(final, labels)
        value, predict_label = torch.max(final.data, 1)
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


def validate_lstm(cnnmodel, data_set_loaders, loss_function, batch_size, use_gpu=True, device_id=(0, 1)):
    cnnmodel.eval()
    val_step = 0
    val_loss = 0
    val_acc = 0
    right_num = 0
    for data in data_set_loaders:
        video, labels = data

        # wrap them in Variable
        if use_gpu:
            video, labels = Variable(video.cuda(async=True)), \
                            Variable(labels.cuda(async=True))
        else:
            video, labels = Variable(video), Variable(labels)
        # forward
        if use_gpu:
            net = torch.nn.DataParallel(cnnmodel, device_ids=device_id)
            outputs, _ = net(video.float())
        else:
            outputs, _ = cnnmodel(video.float())

        # return value and index
        _, predict_label = torch.max(outputs.data, 1)
        loss = loss_function(outputs, labels)

        # statistics
        ls = loss.data[0]
        right_num += torch.sum(predict_label == labels.data)
        val_loss += ls
        val_step += 1
        print('val step {} loss {:.4f} right num {}'.format(val_step, ls, right_num))
    val_acc = float(right_num) / val_step / batch_size
    val_loss /= val_step
    print('Average val loss: {:.4f}, average val accuracy: {:.4f}'.format(val_loss, val_acc))
    return val_loss, val_acc
