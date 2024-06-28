# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/25
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.

import nibabel as nib
import numpy as np
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def nii_loader(path):
    return np.array(nib.load(path).dataobj)


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            return nii_loader(path)
        else:
            return pil_loader(path)
