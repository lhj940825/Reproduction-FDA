# Created by Hojun Lim(Media Informatics, 405165) at 04.09.20

from torchvision.utils import make_grid
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from utils.viz_segmask import colorize_mask


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    images = torch.flip(images, [1]) # restore RGB channel from BGR

    grid_image = make_grid(images.clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)

def load_model_and_optimizer(model, optimizer, checkpoint):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items(): # send all variables in optimizer to gpu
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
