import torch
from torch.autograd import Variable
from tensorboard_logger import log_value
import time


def train(model, data_set_loaders, loss_function, optimizer, global_step, args):
    model.train()  # Set model to training mode
    step = global_step
    t = time.time()
    if args.loss == 'mse_ce':
        for inputs, labels_mse, labels_ce in data_set_loaders:
            # wrap them in Variable
            s = time.time()
            load_time = s - t
            inputs, labels_mse, labels_ce = \
                Variable(inputs.cuda()), Variable(labels_mse.cuda()), Variable(labels_ce.cuda())

            # use DataParallel to realize multi gpu
            net = torch.nn.DataParallel(model, device_ids=args.device_id)
            outputs_mse, outputs_ce, _ = net(inputs)

            loss_function_mse, loss_funciton_ce = loss_function

            loss_mse = loss_function_mse(outputs_mse, labels_mse)
            loss_ce = loss_funciton_ce(outputs_ce, labels_ce)
            loss = 5 * loss_mse + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # statistics
            value, predict_label = torch.max(outputs_ce.data, 1)
            ls_ce, ls_mse = loss_ce.data[0], loss_mse.data[0]
            acc = torch.mean((predict_label == labels_ce.data).float())
            t = time.time()
            run_time = t - s
            log_value('ce_loss', ls_ce, step)
            log_value('mse_loss', ls_mse, step)
            log_value('train_acc', acc, step)
            # log_value('time', t, step)
            print(
                'step: {}, loss_mse {:.4f}, loss_ce {:.4f}, acc {:.4f}, load time: {:.4f}, run time: {:.4f}'.format(
                    step, ls_mse, ls_ce, acc, load_time, run_time))
    else:
        for inputs, labels in data_set_loaders:
            # wrap them in Variable
            s = time.time()
            load_time = s - t
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # use DataParallel to realize multi gpu
            net = torch.nn.DataParallel(model, device_ids=args.device_id)
            outputs, _ = net(inputs)

            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # statistics
            value, predict_label = torch.max(outputs.data, 1)
            ls = loss.data[0]
            acc = torch.mean((predict_label == labels.data).float())
            t = time.time()
            run_time = t - s
            log_value('train_loss', ls, step)
            log_value('train_acc', acc, step)
            # log_value('time', t, step)
            print(
                'step: {}, loss {:.4f}, acc {:.4f}, load time: {:.4f}, run time: {:.4f}'.format(step, ls, acc,
                                                                                                load_time, run_time))
    return step


def validate(model, data_set_loaders, loss_function, args):
    model.eval()
    val_step = 0
    val_loss = 0
    right_num = 0

    if args.loss == 'mse_ce':
        for inputs, labels in data_set_loaders:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            net = torch.nn.DataParallel(model, device_ids=args.device_id)
            _, outputs, _ = net(inputs)

            # return value and index
            _, predict_label = torch.max(outputs.data, 1)
            loss = loss_function[1](outputs, labels)

            # statistics
            ls = loss.data[0]
            right_num += torch.sum(predict_label == labels.data)
            val_loss += ls
            val_step += 1
            print('val step {} loss {:.4f} right num {}'.format(val_step, ls, right_num))
    else:
        for inputs, labels in data_set_loaders:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            net = torch.nn.DataParallel(model, device_ids=args.device_id)
            outputs, _ = net(inputs)

            # return value and index
            _, predict_label = torch.max(outputs.data, 1)
            loss = loss_function(outputs, labels)

            # statistics
            ls = loss.data[0]
            right_num += torch.sum(predict_label == labels.data)
            val_loss += ls
            val_step += 1
            print('val step {} loss {:.4f} right num {}'.format(val_step, ls, right_num))

    val_acc = float(right_num) / val_step / args.batch_size
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
            clip = Variable(clip.cuda(async=True))
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


def validate_lstm(cnnmodel, data_set_loaders, loss_function, batch_size, device_id=(0, 1)):
    cnnmodel.eval()
    val_step = 0
    val_loss = 0
    val_acc = 0
    right_num = 0
    for data in data_set_loaders:
        video, labels = data

        # wrap them in Variable
        video, labels = Variable(video.cuda(async=True)), \
                        Variable(labels.cuda(async=True))

        net = torch.nn.DataParallel(cnnmodel, device_ids=device_id)
        outputs, _ = net(video.float())

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
