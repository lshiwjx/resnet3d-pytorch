import torch
from torch.autograd import Variable
import time
from tensorboard_logger import log_value


def train_val_model(model, data_set_loaders, loss_function, optimizer, num_epoch_save=2,
                    use_gpu=True, num_epochs=100, global_step=0, device_id=(0, 1), batch_size=128, lr_decay_ratio=0.1):
    global_step = global_step
    print('-' * 10)
    loss_all = []
    acc_all = []
    for epoch in range(num_epochs):
        time_start = time.time()

        # Each epoch has a training and validation phase
        val_loss = 0.0
        val_acc = 0.0
        val_step = 0
        for phase in ['train', 'val']:
            print('Epoch {}/{}, phase: {} use_gpu: {}'.
                  format(epoch, num_epochs - 1, phase, use_gpu))

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for data in data_set_loaders[phase]:
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if use_gpu:
                    net = torch.nn.DataParallel(model, device_ids=device_id)
                    outputs = net(inputs.float())
                else:
                    outputs = model(inputs.float())

                # return value and index
                _, preds = torch.max(outputs.data, 1)
                loss = loss_function(outputs, labels)

                # backward + update only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                ls = loss.data[0]
                acc = torch.sum(preds == labels.data) / batch_size

                # statistics
                if phase == 'train':
                    global_step += 1
                    log_value('train_loss', ls, global_step)
                    log_value('train_acc', acc, global_step)
                    print('train global_step: {}, loss {:.4f}, acc {:.4f}'.format(global_step, ls, acc))
                else:
                    val_step += 1
                    val_loss += ls
                    val_acc += acc
                    print('val global_step: {}, loss {:.4f}, acc {:.4f}'.format(val_step, ls, acc))

        # mean the val loss and acc
        val_loss /= val_step
        val_acc /= val_step
        log_value('val_loss', val_loss, global_step)
        log_value('val_acc', val_acc, global_step)
        time_elapsed = time.time() - time_start
        print('\nEpoch {} finished, using time: {:.0f}m {:.0f}s, loss: {:.4f} acc: {:.4f} \n'.
              format(epoch, time_elapsed // 60, time_elapsed % 60, val_loss, val_acc))

        # test if the lr need decay
        loss_all.append(val_loss)
        acc_all.append(val_acc)
        if len(acc_all) > 5:
            test = [abs(acc_all[-i] - acc_all[-i - 1]) < 0.1 for i in range(1, 5)]
            if test[0] and test[1] and test[2] and test[3]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_decay_ratio
                    print('set lr: ', param_group['lr'])
                    log_value('lr', param_group['lr'], global_step)

        # save model
        if epoch % num_epoch_save == 0 and epoch != 0:
            torch.save(model.state_dict(), 'resnet3d_finetuning_18-' + str(global_step) + '.state')

    return loss_all, acc_all
