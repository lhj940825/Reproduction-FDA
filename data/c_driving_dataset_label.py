# Created by Hojun Lim(Media Informatics, 405165) at 31.08.20

import torch
import numpy as np
import os
import os.path as osp
from PIL import Image
from torch.utils import data

class C_DrivingDatasetLabel(data.Dataset): #return target(C_DrivingDataset) dataset with corresponding labels
    def __init__(self, root, list_path, crop_size=(11, 11), mean=(128, 128, 128), max_iters=None, set='val', _type='compound', weather= 'cloudy'):
        self.root = root    # cityscapes
        self.list_path = list_path # list of image names
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int( np.ceil(float(max_iters)/len(self.img_ids)) )

        self.files = []
        self.ignore_label = 255
        self.set = set
        self._type = _type  # for training set, it is either 'compound' or 'open_not_used'
        self.weather = weather

        #TODO note that the label map of c_driving dataset has been already mapped according to the following tabel, and therefore, no need to map it again unlike other datasets(cityscape, bdd100k)
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]   # aachen/aachen_000000_000019_leftImg8bit.png
        image = Image.open( osp.join(self.root, "%s/%s/%s/%s" % (self.set, self._type, self.weather, name)) ).convert('RGB')
        lbname = name.replace(".jpg", "_train_id.png")
        label = Image.open(osp.join(   self.root, "%s/%s/%s/%s" % (self.set,self._type, self.weather, lbname)))
        #print('label path', osp.join(   self.root, "%s/%s/%s/%s" % (self.set,self._type, self.weather, lbname)))
        #print('image path', osp.join(self.root, "%s/%s/%s/%s" % (self.set, self._type, self.weather, name)))

        # resize
        image = image.resize( self.crop_size, Image.BICUBIC )
        label = label.resize( self.crop_size, Image.NEAREST )
        image = np.asarray( image, np.float32 )
        label = np.asarray( label, np.float32 )

        # TODO note that the label map of c_driving dataset has been already mapped according to the following tabel, and therefore, no need to map it again unlike other datasets(cityscape, bdd100k)
        #label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        #for k, v in self.id_to_trainid.items(): #255 denotes non-interesting label, here they ignore a pixel with label 255.
        #    label_copy[label == k] = v
            #print('label', max(label_copy.flatten()))


        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        #return image.copy(), label_copy.copy(), np.array(size), name
        return image.copy(), label.copy(), np.array(size), name

