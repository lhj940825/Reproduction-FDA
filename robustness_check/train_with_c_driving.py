# Created by Hojun Lim(Media Informatics, 405165) at 31.08.20

import numpy as np
from options.train_options_C_Driving import TrainOptions
from options.train
from utils.timer import Timer
import os
from data import CreateSrcDataLoader_with_C_Driving_Cropsize
from data import CreateTrgC_DrivingLoader, CreateTrgVal_C_DrivingLoader
from model import CreateModel
# import tensorboardX
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
from utils import FDA_source_to_target
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from utils.mIoU_utils import compute_mIoU
import shutil

SRC_IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
SRC_IMG_MEAN = torch.reshape(torch.from_numpy(SRC_IMG_MEAN), (1, 3, 1, 1))
TRG_IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32) # subject to be changed if needed
TRG_IMG_MEAN = torch.reshape(torch.from_numpy(TRG_IMG_MEAN), (1,3,1,1))
CS_weights = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32)
CS_weights = torch.from_numpy(CS_weights)


def main():

    opt = TrainOptions()  # loading train options(arg parser)
    args = opt.initialize()  # get arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time': Timer()}

    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)  # print set options

    sourceloader, targetloader = CreateSrcDataLoader_with_C_Driving_Cropsize(args), CreateTrgC_DrivingLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)

    val_targetloader = CreateTrgVal_C_DrivingLoader(args) # dataloader for the validaion set of target domain

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.cuda()

    # losses to log
    loss = ['loss_seg_src', 'loss_seg_trg']
    loss_train = 0.0
    loss_val = 0.0
    loss_train_list = []
    loss_val_list = []

    src_mean_img, trg_mean_img = torch.zeros(1, 1), torch.zeros(1, 1)
    class_weights = Variable(CS_weights).cuda()

    # prepare directories for saving the training and val-loss log files for tensorboard
    root_log_dir = os.path.join(args.tensorboard_log_dir, 'SingleNode_LB_'+str(args.LB).replace('.', '_') + '_%s_FDA_%s' %(args.weather, args.FDA_mode))
    if os.path.isdir(root_log_dir):
        shutil.rmtree(root_log_dir) # remove the existing log files from previous run

    train_log_dir = os.path.join( root_log_dir,'train(src seg loss)')
    val_log_dir = os.path.join(root_log_dir ,'val(tar seg loss)')
    mIoU_log_dir = os.path.join(root_log_dir,'mIoU(tar loss)')
    os.makedirs(train_log_dir, exist_ok=True), os.makedirs(val_log_dir, exist_ok=True), os.makedirs(mIoU_log_dir, exist_ok=True)
    train_loss_writer, val_loss_writer, mIoU_loss_writer = SummaryWriter(log_dir= train_log_dir), SummaryWriter(log_dir= val_log_dir), SummaryWriter(log_dir=mIoU_log_dir)
    #val_loss_writer = SummaryWriter(log_dir= val_log_dir)

    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)  # adjust learning rate
        optimizer.zero_grad()  # zero grad

        src_img, src_lbl, _, _ = sourceloader_iter.next()  # new batch source
        trg_img, _, _ = targetloader_iter.next()  # new batch target
        scr_img_copy = src_img.clone()

        if src_mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            src_mean_img = SRC_IMG_MEAN.repeat(B, 1, H, W)
            trg_mean_img = TRG_IMG_MEAN.repeat(B, 1, H, W)

        # -------------------------------------------------------------------#

        # 1. source to target, target to target
        if args.FDA_mode == 'on':
            src_in_trg = FDA_source_to_target(src_img, trg_img, L=args.LB)  # src_lbl
            trg_in_trg = trg_img

        elif args.FDA_mode == 'off': # not applying the amplitude-switch(amplitude of source by that of target)
            src_in_trg = src_img
            trg_in_trg = trg_img

        else:
            raise KeyError()

        # 2. subtract mean
        src_img = src_in_trg.clone() - src_mean_img  # src, src_lbl
        trg_img = trg_in_trg.clone() - trg_mean_img  # trg, trg_lbl

        # -------------------------------------------------------------------#

        # evaluate and update params #####
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()  # to gpu
        src_seg_score = model(src_img, lbl=src_lbl, weight=class_weights, ita=args.ita)  # forward pass

        loss_seg_src = model.loss_seg  # get loss
        loss_ent_src = model.loss_ent

        # get target loss, only entropy for backpro
        trg_img = Variable(trg_img).cuda()  # to gpu
        trg_seg_score = model(trg_img, lbl=None, weight=class_weights, ita=args.ita)  # forward pass
        loss_seg_trg = model.loss_seg  # get loss, note that target label is only used to compute the validation loss but not for training.
        loss_ent_trg = model.loss_ent

        triger_ent = 0.0
        if i > args.switch2entropy:
            triger_ent = 1.0

        loss_all = loss_seg_src + triger_ent * args.entW * loss_ent_trg  # loss of seg on src, and ent on s and t

        loss_all.backward()
        optimizer.step()

        loss_train += loss_seg_src.detach().cpu().numpy()
        loss_val += loss_seg_trg.detach().cpu().numpy()

        if (i + 1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)

            print('[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % \
                  (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr'] * 10000, #substitue loss_seg_trg by mIOU
                   _t['iter time'].diff))
            sio.savemat(args.tempdata, {'src_img': src_img.cpu().numpy(), 'trg_img': trg_img.cpu().numpy()})

            loss_train /= args.print_freq
            loss_val /= args.print_freq
            loss_train_list.append(loss_train)
            #loss_val_list.append(loss_val)
            sio.savemat(args.matname, {'loss_train': loss_train_list, 'loss_val': loss_val_list})
            train_loss_writer.add_scalar('loss', loss_train, global_step= (i+1) )
            #val_loss_writer.add_scalar('loss', loss_val, global_step=(i + 1))

            loss_train = 0.0
            loss_val = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

        if (i + 1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '%s_2_%s_LB_%s_%s_FDA_%s_iter_' % (args.source, args.target, str(args.LB).replace('.', '_'), args.weather, args.FDA_mode) + str(i + 1) + '.pth'))

            loss_mIoU19 = compute_mIoU(val_targetloader, TRG_IMG_MEAN, model, args.devkit_dir)  # for every (args.save_pred_every)-iteration, evaluate the network performance(mIoU) with validation set
            mIoU_loss_writer.add_scalar('mIoU19', loss_mIoU19, global_step=(i + 1))
            print('[it %d][trg mIoU19 %.4f]' % (i + 1, loss_mIoU19))

    train_loss_writer.close()
    val_loss_writer.close()


if __name__ == '__main__':

    main()

# command(at /media/data/hlim/FDA/FDA)
# ##  python3 robustness_check/train_with_c_dving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='on'

# command for tensorboard
### tensorboard --logdir=../checkpoints/FDA/"NAME_OF_FOLDER_WHERE_LOG_FILES_ARE"
