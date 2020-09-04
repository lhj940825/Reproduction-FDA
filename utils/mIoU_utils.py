# Created by Hojun Lim(Media Informatics, 405165) at 02.09.20

import numpy as np
import torch
import json
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#TODO put following functions in util python file
def fast_hist(a, b, n): #a: label(flatten), b:prediction, n: num_class
    k = (a>=0) & (a<n) # generate the mask
    #print('s', a[k], np.shape(a[k]), np.shape(a))
    #print('a[k]', max(a[k]), max(b[k]))
    #print('bin', np.bincount(a.astype(int)))
    return np.bincount( n*a[k].astype(int)+b[k], minlength=n**2 ).reshape(n, n)

def per_class_iu(hist):
    #print('hist', hist)
    #print('diag', np.diag(hist))
    #print('hist.sum(1)', hist.sum(1))
    #print('hist.sum(0)', hist.sum(0))
    #print('denominator',hist.sum(1)+hist.sum(0)-np.diag(hist) )

    return np.diag(hist) / ( hist.sum(1)+hist.sum(0)-np.diag(hist) ) # nominator= intersection part(diagonal) of confusion matrix, denominator = union part of confusion matrix, hist here seems to be the confusion matrix

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[ input==mapping[ind][0] ] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def save_predictive_image():
    """
    Store the network output given target domain iamges as png.

    """
    # color coding of semantic classes
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    pass
def compute_mIoU(val_targetloader, TRG_IMG_MEAN, model, devkit_dir):

    # initialize Variables
    with open( os.path.join(devkit_dir, 'info.json'),'r' ) as fp:
        info = json.load(fp) # load the label information(num_class, index mapping table, ..) of cityscape.(since cityscape and c_driving have the same mapping and num_Class)
    num_classes = np.int(info['classes'])
    hist = np.zeros((num_classes, num_classes))

    trg_mean_img = torch.zeros(1,1)
    with torch.no_grad():
        for index, batch in enumerate(val_targetloader):
            #if index % 100 == 0:
            #    print('%d processed of validation set' % index)

            image, label, _, name = batch                        # 1. get image from batch
            if trg_mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                trg_mean_img = TRG_IMG_MEAN.repeat(B,1,H,W)

            image = image.clone() - trg_mean_img                 # 2. normalize the image(no division by std)
            image = Variable(image).cuda()

            # forward target image and get the prediction
            output = model(image)
            output = nn.functional.softmax(output, dim=1)
            output = nn.functional.interpolate(output, (720, 1280), mode='bilinear', align_corners=True).cpu().data[
                0].numpy()                                       # (1280 , 720) is the original resolution of target images
            output = output.transpose(1,2,0)                     # output.shape  = (720,1280, 19) = (height, width, num_class)
            pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8 ) # pred.shape  = (720,1280) = (height, width)

            """
                   if save_figure == True:
                       save_prediction()
                       """


            label = np.asarray(label)

            #print('validaionset target image bincount', np.bincount(pred.flatten()), sum(np.bincount(pred.flatten())))
            #print('validaionset target label bincount', np.bincount(label.flatten()), sum(np.bincount(label.flatten())))
            #assert len(label.flatten()) == len(output.flatten())  # the size of label and output should match
            if len(label.flatten()) != len(pred.flatten()):  # size doesn't match
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                      len(pred.flatten()), name,
                                                                                      name.replace(".jpg", "_train_id.png")))
                continue

            hist += fast_hist(label.flatten(), pred.flatten(), num_classes) # hist.shape = (num_class, num_class), hist is the confusion matrix
        mIoUs = per_class_iu(hist)

        #print('===> mIoU19: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        #print('===> mIoU16: ' + str(
        #    round(np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))
        #print('===> mIoU13: ' + str(round(np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))

    return round(np.nanmean(mIoUs) * 100, 2)
