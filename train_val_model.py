import torch
from torch.autograd import Variable
import time
from tensorboard_logger import log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau

MAX_EPOCH = 20
NUM_EPOCH_PER_SAVE = 2
LR_DECAY_RATIO = 0.5


def train(model, data_set_loaders, loss_function, optimizer, global_step,
          use_gpu=True, device_id=(0, 1)):
    model.train()  # Set model to training mode
    for data in data_set_loaders:
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
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
        global_step += 1

        # statistics
        ls = loss.data[0]
        acc = torch.mean((predict_label == labels.data).float())
        log_value('train_loss', ls, global_step)
        log_value('train_acc', acc, global_step)
        print('global_step: {}, loss {:.4f}, acc {:.4f}'.format(global_step, ls, acc))


def validate(model, data_set_loaders, loss_function, use_gpu=True, device_id=(0, 1)):
    model.eval()
    val_step = 0
    val_loss = 0
    val_acc = 0
    for data in data_set_loaders:
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
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


def train_val_model(model, data_set_loaders, loss_function, optimizer,
                    use_gpu=True, global_step=0, device_id=(0, 1)):
    global_step = global_step
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_DECAY_RATIO, patience=4, verbose=True, threshold=0.1,
                                  threshold_mode='abs', cooldown=2)
    for epoch in range(MAX_EPOCH):
        time_start = time.time()
        print('Epoch {}/{}'.format(epoch, MAX_EPOCH - 1))
        print('Training --------- use_gpu: ', use_gpu)
        train(model, data_set_loaders['train'], loss_function, optimizer, global_step, use_gpu, device_id)
        print('Validating --------- use_gpu: ', use_gpu)
        loss, acc = validate(model, data_set_loaders['val'], loss_function, use_gpu, device_id)
        scheduler.step(acc)
        time_elapsed = time.time() - time_start
        print('\nEpoch {} finished, using time: {:.0f}m {:.0f}s, loss: {:.4f} acc: {:.4f} \n'.
              format(epoch, time_elapsed // 60, time_elapsed % 60, loss, acc))

        # save model
        if epoch % NUM_EPOCH_PER_SAVE == 0 and epoch != 0:
            torch.save(model.state_dict(), 'resnet3d_finetuning_18-' + str(global_step) + '.state')
