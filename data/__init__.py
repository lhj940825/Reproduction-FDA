import os
import numpy as np
from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.cityscapes_dataset_SSL import cityscapesDataSetSSL
from data.synthia_dataset import SYNDataSet
from data.bdd100kdataset_label import BDD100kDatasetLabel
from data.bdd100k_dataset import BDD100kDataset
from data.c_driving_dataset_label import C_DrivingDatasetLabel
from data.c_driving_dataset import C_DrivingDataset

IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760), 'bdd100k':(1280, 720), 'c_driving': (1024,512)}
cs_size_test = {'cityscapes': (1344,576), 'bdd100k':(1280, 720), 'c_driving': (1280, 720)}

def CreateSrcDataLoader(args):
    if args.source == 'gta5':
        source_dataset = GTA5DataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'], 
                                      resize=image_sizes['gta5'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    elif args.source == 'synthia':
        source_dataset = SYNDataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'],
                                      resize=image_sizes['synthia'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    source_dataloader = data.DataLoader( source_dataset, 
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         num_workers=args.num_workers, 
                                         pin_memory=True )    
    return source_dataloader

def CreateTrgDataLoader(args):

    if args.set == 'train' or args.set == 'trainval':

        target_dataset = cityscapesDataSetLabel( args.data_dir_target, 
                                                 args.data_list_target, 
                                                 crop_size=image_sizes['cityscapes'], 
                                                 mean=IMG_MEAN, 
                                                 max_iters=args.num_steps * args.batch_size, 
                                                 set=args.set )
    else:
        target_dataset = cityscapesDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['cityscapes'],
                                            mean=IMG_MEAN,
                                            set=args.set )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader



def CreateTrgDataSSLLoader(args):
    target_dataset = cityscapesDataSet( args.data_dir_target, 
                                        args.data_list_target,
                                        crop_size=image_sizes['cityscapes'],
                                        mean=IMG_MEAN, 
                                        set=args.set )
    target_dataloader = data.DataLoader( target_dataset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         pin_memory=True )
    return target_dataloader



def CreatePseudoTrgLoader(args):
    target_dataset = cityscapesDataSetSSL( args.data_dir_target,
                                           args.data_list_target,
                                           crop_size=image_sizes['cityscapes'],
                                           mean=IMG_MEAN,
                                           max_iters=args.num_steps * args.batch_size,
                                           set=args.set,
                                           label_folder=args.label_folder )

    target_dataloader = data.DataLoader( target_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True )

    return target_dataloader

################################################################################################
#TODO functions defined below are implemented to use for the ROBUSTNESS_CHECK Project

def CreateSrcDataLoader_with_C_Driving_Cropsize(args):
    """
    # since original image sizes of the gta5 and C_Driving are the same, upscale the gta5 dataset and conduct the random crop with the size of C_Driving for data augmentation
    """

    if args.source == 'gta5':
        source_dataset = GTA5DataSet(args.data_dir, args.data_list, crop_size=image_sizes['c_driving'],
                                     resize=image_sizes['gta5'], mean=IMG_MEAN,
                                     max_iters=args.num_steps * args.batch_size)

    elif args.source == 'synthia': # actually not used in ROBUSTNESS_CHECK Project
        source_dataset = SYNDataSet(args.data_dir, args.data_list, crop_size=image_sizes['synthia'],
                                    resize=image_sizes['c_driving'], mean=IMG_MEAN,
                                    max_iters=args.num_steps * args.batch_size)
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')

    source_dataloader = data.DataLoader(source_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    return source_dataloader

def CreateTrgC_DrivingLoader(args):
    if args.set == 'train' or args.set == 'trainval': # for training the network with C_Driving dataset we only need target images, no need corresponding labels

        target_dataset = C_DrivingDataset( args.data_dir_target,
                                                 os.path.join(args.data_list_target, '%s_%s.txt' %(args.set, args.weather)),
                                                 crop_size=image_sizes['c_driving'],
                                                 mean=IMG_MEAN,
                                                 max_iters=args.num_steps * args.batch_size,
                                                 set=args.set, _type= args._type , weather= args.weather)
    else:
        target_dataset = C_DrivingDatasetLabel( args.data_dir_target,
                                            os.path.join(args.data_list_target, '%s_%s.txt' %(args.set, args.weather)),
                                            crop_size=image_sizes['c_driving'],
                                            mean=IMG_MEAN,
                                            set=args.set,_type= args._type , weather= args.weather )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             pin_memory=True )

    return target_dataloader


def CreateTrgVal_C_DrivingLoader(args): # return the dataloader which contains image and corresponding label from the validation set
    val_target_dataset =  C_DrivingDatasetLabel( args.data_dir_target,
                                            os.path.join(args.data_list_target, '%s_%s.txt' %('val', args.weather)),
                                            crop_size=cs_size_test['c_driving'],
                                            mean=IMG_MEAN,
                                            set='val', _type= args._type , weather= args.weather )

    val_target_dataloader = data.DataLoader(val_target_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        pin_memory=True)

    return val_target_dataloader

def CreateTrgBDD100kLoader(args):
    if args.set == 'train' or args.set == 'trainval':

        target_dataset = BDD100kDatasetLabel( args.data_dir_target,
                                                 args.data_list_target,
                                                 crop_size=image_sizes['bdd100k'],
                                                 mean=IMG_MEAN,
                                                 max_iters=args.num_steps * args.batch_size,
                                                 set=args.set )
    else:
        target_dataset = BDD100kDataset( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['bdd100k'],
                                            mean=IMG_MEAN,
                                            set=args.set )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             pin_memory=True )

    return target_dataloader