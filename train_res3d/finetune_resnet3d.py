from __future__ import print_function, division

import os
import shutil
import time
import torch
from tensorboard_logger import configure, log_value, Logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_res3d.parser_args import parser_args, parser_cifar_args
from data_set.data_choose import data_choose
from model.model_choose import model_choose
from train_res3d import train_val_model
from train_res3d.optimizer_choose import optimizer_choose
from train_res3d.loss_choose import loss_choose
# params
args = parser_args()
# args = parser_cifar_args()

print(args.log_dir)
configure(args.log_dir)

global_step, model = model_choose(args)

data_set_loaders = data_choose(args)

optimizer = optimizer_choose(model, args)

loss_function = loss_choose(args.loss)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_ratio,
                                 patience=args.lr_patience, verbose=True,
                                 threshold=args.lr_threshold, threshold_mode='abs',
                                 cooldown=args.lr_delay)
print('----lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'
      .format(args.lr, args.lr_decay_ratio, args.lr_patience, args.lr_threshold, args.lr_delay))

# for tensorboard --logdir runs
# if os.path.isdir(args.log_dir) and not args.use_last_model:
#     shutil.rmtree(args.log_dir)
#     print('Dir removed: ', args.log_dir)

# total_time = time.time()
print('----Train and val begin, total epoch: ', args.max_epoch)
for epoch in range(args.max_epoch):
    lr = optimizer.param_groups[0]['lr']
    log_value('lr', lr, global_step)
    log_value('epoch', epoch, global_step)
    time_start = time.time()
    print('Epoch {}/{}'.format(epoch, args.max_epoch - 1))
    print('Train')
    global_step = train_val_model.train(model, data_set_loaders['train'], loss_function, optimizer,
                                        global_step, args)
    print('Validate')
    loss, acc = train_val_model.validate(model, data_set_loaders['val'], loss_function, args)
    log_value('val_loss', loss, global_step)
    log_value('val_acc', acc, global_step)
    lr_scheduler.step(loss)
    time_elapsed = time.time() - time_start
    log_value('epoch_time', time_elapsed, global_step)
    print('validate loss: {:.4f} acc: {:.4f} lr: {}'.format(loss, acc, lr))

    # save model
    if (epoch + 1) % args.num_epoch_per_save == 0:
        torch.save(model.state_dict(), args.model_saved_name + '-' + str(global_step) + '.state')
        print('Save model at step ', global_step)
    print('Epoch {} finished, using time: {:.0f}m {:.0f}s'.
          format(epoch, time_elapsed // 60, time_elapsed % 60))

# print('Finished in time: ', time.time()-total_time)
torch.save(model.cpu().state_dict(), args.model_saved_name + '-' + str(global_step) + '.state')
