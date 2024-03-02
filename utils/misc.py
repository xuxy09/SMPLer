from PIL import Image
import numpy as np
import torch
import random

def imresize(old_image, size=None, ratio=None, mode='bicubic'):
    # old_image needs to be uint8 (0~255) or float (0~1)
    # output is uint8 or float32 depending on the input dtype

    is_uint8 = (old_image.dtype == np.uint8)
    if not is_uint8:
        old_image = (old_image * 255.0).round().clip(0, 255).astype(np.uint8)

    im = Image.fromarray(old_image)
    if size is None:
        if ratio is None:
            raise ValueError(f'size is {size}, ratio is {ratio}')
        else:
            size = tuple((np.array(im.size) * ratio).astype(int))

    mode_dict = {'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    new_image = np.array(im.resize(size, mode_dict[mode]))

    if not is_uint8:
        new_image = new_image.astype(np.float32) / 255.0
    
    return new_image

def set_seed(seed, cudnn_det=False, cudnn_ben=True):
    # Set cudnn_det=True and cudnn_ben=False if you want fully-reproducible results
    # Conversely, cudnn_det=False and cudnn_ben=True will give better running speed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_ben
