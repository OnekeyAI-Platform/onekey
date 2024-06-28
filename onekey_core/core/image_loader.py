# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/02/23
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import os
from typing import Optional

import nibabel
import nrrd
import numpy as np


def image_loader_3d(impath: str, root='', index_order='F') -> Optional[np.ndarray]:
    """
    Args:
        impath: image path
        root: Where impath is relative path, use root to concat impath
        index_order: {'C', 'F'}, optional
        Specifies the index order of the resulting data array. Either 'C' (C-order) where the dimensions are ordered from
        slowest-varying to fastest-varying (e.g. (z, y, x)), or 'F' (Fortran-order) where the dimensions are ordered
        from fastest-varying to slowest-varying (e.g. (x, y, z)).

    Returns:

    """
    assert index_order in ['F', 'C']
    impath = os.path.join(root, impath)
    if impath and os.path.exists(impath):
        if impath.endswith('.nrrd'):
            nrrd_data, _ = nrrd.read(impath, index_order=index_order)
            return nrrd_data
        elif impath.endswith('.nii.gz') or impath.endswith('.nii'):
            image = nibabel.load(impath).get_data()
            if index_order == 'C':
                image = np.transpose(image, [2, 1, 0])
            return image
    else:
        return None
